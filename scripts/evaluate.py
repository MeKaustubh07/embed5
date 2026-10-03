"""Score every way of storing the index on what has been embedded so far: cache only, no API calls.

For each set and language, every (index model, question model) pair that is embedded, at every dim, as float,
binary (Hamming), binary_asym (float question vs the page's bits) and int8; plus two-stage search: a 32-byte binary
first pass over every page, then the top n re-scored with a bigger vector. Metrics (src/metrics.py): RCP-nDCG@10
with a shortlist (Cohere's protocol) and without (every page competes), human-graded qrel-nDCG@10 and recall@10/100
over every page.
Writes results/summary.csv (means; results/ is gitignored until Cohere OKs publishing) and cache/eval/<subset>.npz
(per-question values, for confidence intervals). Usage: python scripts/evaluate.py [set ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import metrics  # noqa: E402
import systems as sy  # noqa: E402

RESULTS, PER_QUERY = ROOT / "results", ROOT / "cache" / "eval"
FIRST_PASS_N = (100, 200)                  # pages kept by the 32-byte first pass for re-scoring


def try_cached(texts, model, input_type, dim=2048, types=sy.TYPES):
    try:
        return sy.cached(texts, model, input_type, dim, types)
    except KeyError:
        return None


def systems_for(P: dict, Q: dict, P1k: dict):
    """-> [(row fields, score matrix)] for one set and language."""
    out = []
    for ip in P:
        for qm in Q:
            pair = {"index": ip, "query": qm}
            for d in sy.DIMS:
                for prec in ("float", "binary", "binary_asym"):
                    out.append((pair | {"precision": prec, "dim": d, "stage": "single"},
                                lambda ip=ip, qm=qm, d=d, prec=prec: sy.scores(prec, Q[qm], P[ip], d)))
            out.append((pair | {"precision": "int8", "dim": 2048, "stage": "single"},
                        lambda ip=ip, qm=qm: sy.scores("int8", Q[qm], P[ip], 2048)))
            if ip in P1k:
                out.append((pair | {"precision": "int8", "dim": 1024, "stage": "single"},
                            lambda ip=ip, qm=qm: sy.scores("int8", Q[qm], P1k[ip], 1024)))
            rescorers = {"float2048": lambda ip=ip, qm=qm: sy.scores("float", Q[qm], P[ip], 2048),
                         "int8_2048": lambda ip=ip, qm=qm: sy.scores("int8", Q[qm], P[ip], 2048)}
            if ip in P1k:
                rescorers["int8_1024"] = lambda ip=ip, qm=qm: sy.scores("int8", Q[qm], P1k[ip], 1024)
            for n in FIRST_PASS_N:
                for rname, rescore in rescorers.items():
                    out.append((pair | {"precision": "binary", "dim": 256, "stage": f"then {rname} on top {n}"},
                                lambda ip=ip, qm=qm, n=n, rescore=rescore:
                                sy.two_stage(sy.scores("binary", Q[qm], P[ip], 256), rescore(), n)))
    return out


def run_set(name: str) -> list[dict]:
    d = data.load(name)
    pages = d.pages().text.tolist()
    P = {m: v for m in sy.MODELS if (v := try_cached(pages, m, "search_document")) is not None}
    P1k = {m: v for m in sy.MODELS if (v := try_cached(pages, m, "search_document", 1024, ("int8",))) is not None}
    if not P:
        print(f"{name}: no pages embedded yet")
        return []
    rows = []
    for lang in data.LANGS:
        qs = metrics.QuerySet.load(name, lang, d)
        qtext = d.queries[d.queries.language == lang].text.tolist()
        Q = {m: v for m in sy.MODELS if (v := try_cached(qtext, m, "search_query")) is not None}
        per_query = {}
        for fields, build in systems_for(P, Q, P1k):
            m = metrics.evaluate(build(), qs)
            key = "|".join(str(fields[k]) for k in ("index", "query", "precision", "dim", "stage"))
            per_query |= {f"{key}|{k}": v for k, v in m.items()}
            rows.append({"set": name, "lang": lang, "subset": qs.subset, **fields,
                         "bytes": sy.BYTES[fields["precision"]](fields["dim"]), "n_queries": len(qs.query_ids)}
                        | {k: float(np.nanmean(v)) for k, v in m.items()})
        PER_QUERY.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PER_QUERY / f"{qs.subset}.npz", **per_query)
        print(f"  {qs.subset:28s} {len(Q)} question model(s) x {len(P)} index model(s): {len(per_query) // 5} systems",
              flush=True)
    return rows


def main(sets: list[str]) -> None:
    t0 = time.monotonic()
    rows = [r for s in sets for r in run_set(s)]
    if not rows:
        return
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "summary.csv"
    if out.exists():                       # keep the other sets' rows
        old = pd.read_csv(out)
        df = pd.concat([old[~old.set.isin(df.set.unique())], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"{len(rows)} rows -> {out} ({time.monotonic() - t0:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1:] or list(data.DOMAINS))
