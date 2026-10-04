"""How deep must a 32-byte first pass's re-check go as the collection grows? (cache only, no API calls)

For each question (English, Pro index and Pro questions), a collection of N pages: its human-relevant pages plus
random other pages. Up to the size of its own set the others come from that set (similar documents: the hard
case); beyond it, pages of the other sets are added (different topics: easier distractors). Five random collections
per size, the same ones for every first pass. For each first pass (32 B Hamming, 32 B against the float question and
against its 8-bit version, and full float as the reference) and depth n: the first pass's recall@n of the
human-relevant pages, and the human-graded nDCG@10 after re-scoring its top n with int8 2048, against full float on
the same collection. Ties at the cut are broken at random (page order is not neutral).

The depth for 95% recall at each size comes with a 95% bootstrap interval over questions.
Writes results/sweep.csv (means), results/sweep_depth95.csv and cache/sweep/<set>.npz (per-question recall).
Usage: python scripts/sweep.py
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

RESULTS, PER_QUESTION = ROOT / "results", ROOT / "cache" / "sweep"
REPEATS, SEED, BOOT = 5, 0, 1000
WITHIN = (100, 200, 400, 800, 1600)                 # sizes below the set's own size, plus the whole set
DEPTH_SHARES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
DEPTHS = (10, 20, 50, 100, 200, 400)
FIRST_PASSES = ("float2048", "binary256", "binary_asym256", "binary_asym8_256")


def pages_of(name):
    d = data.load(name)
    texts = d.pages().text.tolist()
    v = sy.cached(texts, "pro", "search_document")
    return d, v


def depth_at(depths: np.ndarray, recall: np.ndarray, target: float = 0.95) -> float:
    """The depth where mean recall reaches target, interpolated log-linearly between grid depths (NaN if never)."""
    if recall.max() < target:
        return np.nan
    j = int(np.argmax(recall >= target))
    if j == 0:
        return float(depths[0])
    t = (target - recall[j - 1]) / (recall[j] - recall[j - 1])
    return float(np.exp(np.log(depths[j - 1]) + t * (np.log(depths[j]) - np.log(depths[j - 1]))))


def run_set(name: str, others: list[str]) -> tuple[list[dict], dict]:
    d, P = pages_of(name)
    pad = [pages_of(o)[1] for o in others]
    allP = {e: np.concatenate([P[e]] + [p[e] for p in pad]) for e in sy.TYPES}
    n_own, n_all = len(P["float"]), len(allP["float"])
    qs = metrics.QuerySet.load(name, "english", d)
    Q = sy.cached(d.queries[d.queries.language == "english"].text.tolist(), "pro", "search_query")
    full = {"float2048": sy.cosine(Q["float"], allP["float"], 2048), "int8": sy.scores("int8", Q, allP, 2048),
            "binary256": sy.hamming(Q["ubinary"], allP["ubinary"], 256),
            "binary_asym256": sy.binary_asym(Q["float"], allP["ubinary"], 256),
            "binary_asym8_256": sy.binary_asym8(Q["float"], allP["ubinary"], 256)}
    sizes = [n for n in WITHIN if n < n_own] + [n_own] + [n for n in (2 * n_own, 4 * n_own) if n < n_all] + [n_all]
    sizes = sorted(set(s for s in sizes if s > 0))
    rng, ties = np.random.default_rng(SEED), np.random.default_rng(SEED + 1)
    rows, per_q = [], {}
    for N in sizes:
        grid = sorted(n for n in set(DEPTHS) | {max(1, round(s * N)) for s in DEPTH_SHARES} if n <= N)
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
                pick = np.sort(pick)
                g = gfull[pick]
                ideal = g[g > 0]
                acc.setdefault(("float", 0), []).append(metrics.ndcg(full["float2048"][i, pick], g, ideal))
                r8 = full["int8"][i, pick]
                for fp in FIRST_PASSES:
                    first = full[fp][i, pick]
                    order = np.lexsort((ties.random(len(first)), -first))      # ties at the cut in random order
                    for n in grid:
                        keep = order[:n]
                        s2 = first - first.max() - 1.0
                        s2[keep] = r8[keep] - r8[keep].min() + 1.0
                        acc.setdefault((fp, n), []).append(
                            (metrics.ndcg(s2, g, ideal), metrics.recall(first, g > 0, n)))
            fl = float(np.mean(acc.pop(("float", 0))))
            for (fp, n), v in acc.items():
                v = np.array(v)
                per_q[f"{N}|{rep}|{fp}|{n}"] = v[:, 1].astype(np.float32)
                rows.append({"set": name, "N": N, "distractors": "same set" if N <= n_own else "other sets",
                             "repeat": rep, "first_pass": fp, "depth": n, "depth_share": n / N,
                             "qrel_float": fl, "qrel_two_stage": float(v[:, 0].mean()),
                             "loss": fl - float(v[:, 0].mean()), "recall_at_depth": float(v[:, 1].mean())})
        print(f"  {name:17s} N={N:5d} done", flush=True)
    return rows, per_q


def depth95(name: str, per_q: dict, rows: list[dict]) -> list[dict]:
    """Depth for 95% recall per (N, first pass), with a 95% bootstrap interval: questions resampled with replacement
    (the same resample for every repeat and depth), recall averaged over the repeats."""
    meta = pd.DataFrame(rows)[["N", "distractors"]].drop_duplicates().set_index("N").distractors
    keys = pd.DataFrame([k.split("|") for k in per_q], columns=["N", "rep", "fp", "depth"]).astype(
        {"N": int, "rep": int, "depth": int})
    nq = len(next(iter(per_q.values())))
    boot = np.random.default_rng(SEED + 2).integers(0, nq, (BOOT, nq))
    out = []
    for (N, fp), k in keys.groupby(["N", "fp"]):
        depths = np.array(sorted(k.depth.unique()))
        # [depth, question]: per-question recall averaged over the repeats
        R = np.stack([np.mean([per_q[f"{N}|{r}|{fp}|{n}"] for r in sorted(k.rep.unique())], axis=0) for n in depths])
        est = depth_at(depths, R.mean(1))
        bs = np.array([depth_at(depths, R[:, b].mean(1)) for b in boot])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(bs).any() else (np.nan, np.nan)
        out.append({"set": name, "N": N, "distractors": meta[N], "first_pass": fp, "depth95": est,
                    "ci_low": float(lo), "ci_high": float(hi), "share": est / N})
    return out


def main() -> None:
    t0 = time.monotonic()
    ready = []
    for s in data.DOMAINS:
        try:
            pages_of(s)
            ready.append(s)
        except KeyError:
            print(f"{s}: not fully embedded yet, skipped")
    rows, d95 = [], []
    PER_QUESTION.mkdir(parents=True, exist_ok=True)
    for s in ready:
        r, per_q = run_set(s, [o for o in ready if o != s])
        np.savez_compressed(PER_QUESTION / f"{s}.npz", **per_q)
        rows += r
        d95 += depth95(s, per_q, r)
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "sweep.csv", index=False)
    pd.DataFrame(d95).to_csv(RESULTS / "sweep_depth95.csv", index=False)
    print(f"{len(rows)} rows -> {RESULTS / 'sweep.csv'}, {len(d95)} -> sweep_depth95.csv ({time.monotonic() - t0:.0f} s)")


if __name__ == "__main__":
    main()
