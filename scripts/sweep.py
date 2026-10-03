"""How deep must a 32-byte first pass's re-check go as the collection grows? (cache only, no API calls)

For each question (English, Pro index and Pro questions), a collection of N pages: its human-relevant pages plus
random other pages. Up to the size of its own set the others come from that set (similar documents: the hard
case); beyond it, pages of the other sets are added (different topics: easier distractors). At each N, for each
first pass (32 B Hamming, 32 B against the float question and against its 8-bit version) and depth n: the human-graded nDCG@10 after re-scoring
the top n with int8 2048, against full float on the same collection, and the first pass's recall@n. Five random
collections per size, the same ones for every system. Writes results/sweep.csv. Usage: python scripts/sweep.py
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

RESULTS = ROOT / "results"
REPEATS, SEED = 5, 0
WITHIN = (100, 200, 400, 800, 1600)                 # sizes below the set's own size, plus the whole set
DEPTH_SHARES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
DEPTHS = (10, 20, 50, 100, 200, 400)


def pages_of(name):
    d = data.load(name)
    texts = d.pages().text.tolist()
    v = sy.cached(texts, "pro", "search_document")
    return d, v


def run_set(name: str, others: list[str]) -> list[dict]:
    d, P = pages_of(name)
    pad = [pages_of(o)[1] for o in others]
    allP = {e: np.concatenate([P[e]] + [p[e] for p in pad]) for e in sy.TYPES}
    n_own, n_all = len(P["float"]), len(allP["float"])
    qs = metrics.QuerySet.load(name, "english", d)
    Q = sy.cached(d.queries[d.queries.language == "english"].text.tolist(), "pro", "search_query")
    full = {"float": sy.cosine(Q["float"], allP["float"], 2048), "int8": sy.scores("int8", Q, allP, 2048),
            "binary256": sy.hamming(Q["ubinary"], allP["ubinary"], 256),
            "binary_asym256": sy.binary_asym(Q["float"], allP["ubinary"], 256),
            "binary_asym8_256": sy.binary_asym8(Q["float"], allP["ubinary"], 256)}
    sizes = [n for n in WITHIN if n < n_own] + [n_own] + [n for n in (2 * n_own, 4 * n_own) if n < n_all] + [n_all]
    sizes = sorted(set(s for s in sizes if s > 0))
    rng = np.random.default_rng(SEED)
    rows = []
    for N in sizes:
        for rep in range(REPEATS if N < n_all and N != n_own else 1):     # the full set and everything: one draw
            acc = {}
            for i in range(len(qs.query_ids)):
                gfull = np.concatenate([qs.grade[i], np.zeros(n_all - n_own)])   # other sets' pages: grade 0
                rel = np.flatnonzero(gfull > 0)
                if N < n_own:
                    rest = np.setdiff1d(np.arange(n_own), rel)
                    pick = np.concatenate([rel, rng.choice(rest, N - len(rel), replace=False)])
                elif N < n_all:
                    pick = np.concatenate([np.arange(n_own), rng.choice(np.arange(n_own, n_all), N - n_own,
                                                                        replace=False)])
                else:
                    pick = np.arange(n_all)
                pick = np.sort(pick)          # page order, not relevant-first: ties must not favour relevant pages
                g = gfull[pick]
                ideal = g[g > 0]
                f = full["float"][i, pick]
                acc.setdefault(("float", 0), []).append(metrics.ndcg(f, g, ideal))
                r8 = full["int8"][i, pick]
                for fp in ("binary256", "binary_asym256", "binary_asym8_256"):
                    first = full[fp][i, pick]
                    order = np.argsort(-first, kind="stable")
                    for n in sorted(set(DEPTHS) | {max(1, round(s * N)) for s in DEPTH_SHARES}):
                        if n > N:
                            continue
                        keep = order[:n]
                        s2 = first.astype(np.float64) - first.max() - 1.0
                        s2[keep] = r8[keep] - r8[keep].min() + 1.0
                        acc.setdefault((fp, n), []).append(
                            (metrics.ndcg(s2, g, ideal), metrics.recall(first, g > 0, n)))
            fl = float(np.mean(acc.pop(("float", 0))))
            for (fp, n), v in acc.items():
                v = np.array(v)
                rows.append({"set": name, "N": N, "distractors": "same set" if N <= n_own else "other sets",
                             "repeat": rep, "first_pass": fp, "depth": n, "depth_share": n / N,
                             "qrel_float": fl, "qrel_two_stage": float(v[:, 0].mean()),
                             "loss": fl - float(v[:, 0].mean()), "recall_at_depth": float(v[:, 1].mean())})
        print(f"  {name:17s} N={N:5d} done", flush=True)
    return rows


def main() -> None:
    t0 = time.monotonic()
    ready = []
    for s in data.DOMAINS:
        try:
            pages_of(s)
            ready.append(s)
        except KeyError:
            print(f"{s}: not fully embedded yet, skipped")
    rows = [r for s in ready for r in run_set(s, [o for o in ready if o != s])]
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "sweep.csv", index=False)
    print(f"{len(rows)} rows -> {RESULTS / 'sweep.csv'} ({time.monotonic() - t0:.0f} s)")


if __name__ == "__main__":
    main()
