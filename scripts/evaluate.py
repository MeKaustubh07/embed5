"""Score every way of storing the index on what has been embedded so far: cache only, no API calls.

For each set and language, every embedded (index model, question model) pair:
  single     the index as float / binary (Hamming) / binary_asym (float question vs the page's bits) at every dim,
             and as the API's int8 at 2048 and 1024 (float question vs int8 page)
  two-stage  a first pass over every page (32 B Hamming, 32 B binary_asym with the float or the 8-bit question, 128 B
             Hamming) keeps the top `depth` (ties at the cut in random order, averaged over TIE_SEEDS orders)
             pages, which a bigger vector re-scores (float 2048, int8 2048, int8 1024). Depths are page counts and
             shares of the set (the share is what carries to a bigger collection)
Metrics (src/metrics.py): RCP- and qrel-nDCG@10 with the shortlist and over the corpus, never-judged pages in the
corpus top 10, recall@10/100. Translated questions are also scored against their English original's judged pool
(judged_by=english): each language's own pool was built partly by BM25, which barely works across languages.
Writes results/summary.csv (gitignored until Cohere OKs publishing) and cache/eval/<subset>.npz (per question, for
the clustered confidence intervals in scripts/stats.py). Usage: python scripts/evaluate.py [set ...]
"""
import sys
import time
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import metrics  # noqa: E402
import systems as sy  # noqa: E402

RESULTS, PER_QUERY = ROOT / "results", ROOT / "cache" / "eval"
FIRST_PASSES = {"binary256": ("binary", 256), "binary_asym256": ("binary_asym", 256),
                "binary_asym8_256": ("binary_asym8", 256), "binary1024": ("binary", 1024)}
RESCORE_BYTES = {"float2048": 8192, "int8_2048": 2048, "int8_1024": 1024}
DEPTHS = (10, 20, 50, 100, 200)
DEPTH_SHARES = (0.01, 0.02, 0.05, 0.10)
TIE_SEEDS = 3                              # two-stage: per-question metrics averaged over random tie orders at the cut


def try_cached(texts, model, input_type, dim=2048, types=sy.TYPES):
    try:
        return sy.cached(texts, model, input_type, dim, types)
    except KeyError:
        return None


def systems_for(P: dict, Q: dict, P1k: dict, n_pages: int, two_stage: bool = True):
    """-> [(row fields, build(seed))] for one set and language; builds share their score matrices, and only a
    two-stage build depends on the seed (its random tie order at the cut)."""
    @cache
    def single(ip, qm, prec, d):
        if prec == "int8" and d == 1024:
            return sy.scores("int8", Q[qm], P1k[ip], 1024)
        return sy.scores(prec, Q[qm], P[ip], d)

    rescorers = {"float2048": ("float", 2048), "int8_2048": ("int8", 2048)}
    depths = sorted(set(DEPTHS) | {max(1, round(f * n_pages)) for f in DEPTH_SHARES})
    out = []
    for ip in P:
        for qm in Q:
            base = {"index": ip, "query": qm, "rescorer": "", "disk_bytes": 0, "depth": 0}
            cells = [(prec, d) for d in sy.DIMS for prec in ("float", "binary", "binary_asym")] + [("int8", 2048)]
            cells += [("binary_asym8", 256)]                   # what the GPU's bit-serial kernel computes
            cells += [("int8", 1024)] if ip in P1k else []
            for prec, d in cells:
                out.append((base | {"kind": "single", "precision": prec, "dim": d, "ram_bytes": sy.BYTES[prec](d)},
                            lambda seed, ip=ip, qm=qm, prec=prec, d=d: single(ip, qm, prec, d)))
            if not two_stage:
                continue
            rs = rescorers | ({"int8_1024": ("int8", 1024)} if ip in P1k else {})
            for fname, (fprec, fd) in FIRST_PASSES.items():
                for rname, (rprec, rd) in rs.items():
                    for n in depths:
                        out.append(({"index": ip, "query": qm, "kind": "two-stage", "precision": fprec, "dim": fd,
                                     "ram_bytes": sy.BYTES[fprec](fd), "rescorer": rname,
                                     "disk_bytes": RESCORE_BYTES[rname], "depth": n},
                                    lambda seed, ip=ip, qm=qm, fprec=fprec, fd=fd, rprec=rprec, rd=rd, n=n:
                                    sy.two_stage(single(ip, qm, fprec, fd), single(ip, qm, rprec, rd), n, seed)))
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
        qtext = d.queries[d.queries.language == lang].text.tolist()
        Q = {m: v for m in sy.MODELS if (v := try_cached(qtext, m, "search_query")) is not None}
        for judge in [None] + (["english"] if lang != "english" else []):
            qs = metrics.QuerySet.load(name, lang, d, judged_by=judge)
            per_query = {"__query_ids__": np.array(qs.query_ids), "__en_ids__": np.array(qs.en_ids)}
            for fields, build in systems_for(P, Q, P1k, len(pages), two_stage=judge is None):
                if fields["kind"] == "two-stage":           # averaged over random tie orders at the cut
                    runs = [metrics.evaluate(build(seed), qs) for seed in range(TIE_SEEDS)]
                    m = {k: np.mean([r[k] for r in runs], axis=0) for k in runs[0]}
                    m["rcp_shortlist"][:] = m["qrel_shortlist"][:] = np.nan   # no shortlist protocol for a first pass
                else:
                    m = metrics.evaluate(build(0), qs)
                key = "|".join(str(fields[k]) for k in ("index", "query", "kind", "precision", "dim", "rescorer",
                                                       "depth"))
                per_query |= {f"{key}|{k}": v for k, v in m.items()}
                rows.append({"set": name, "lang": lang, "judged_by": judge or lang, "subset": qs.subset, **fields,
                             "depth_share": fields["depth"] / len(pages), "n_pages": len(pages),
                             "n_queries": len(qs.query_ids)} | {k: float(np.nanmean(v)) if np.isfinite(v).any()
                                                                 else np.nan for k, v in m.items()})
            PER_QUERY.mkdir(parents=True, exist_ok=True)
            tag = qs.subset.replace(" (judged by english)", "__judged_by_english")
            np.savez_compressed(PER_QUERY / f"{tag}.npz", **per_query)
            print(f"  {qs.subset:44s} {len(Q)} x {len(P)} model pair(s): {(len(per_query) - 2) // 7} systems",
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
        if "judged_by" in old:             # an older format is simply replaced
            df = pd.concat([old[~old.set.isin(df.set.unique())], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"{len(rows)} rows -> {out} ({time.monotonic() - t0:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1:] or list(data.DOMAINS))
