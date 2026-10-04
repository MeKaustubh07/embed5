"""Every number results/RESULTS_SHEET.md cites, recomputed from the scored data (cache/eval, cache/sweep and
results/*.csv; no API calls), so the sheet can be checked and regenerated. Points = qrel-nDCG@10 x 100 unless a key
says rcp. Intervals: paired bootstrap over questions, a question and its translations one cluster (10,000 resamples,
seed 0), two-sided bootstrap p-values, Holm across the 6-language contrasts the sheet reports.
Writes results/sheet_numbers.json. Usage: python scripts/sheet_numbers.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import metrics  # noqa: E402
import systems as sy  # noqa: E402

EVAL, SWEEP, RESULTS = ROOT / "cache" / "eval", ROOT / "cache" / "sweep", ROOT / "results"
B, SEED = 10_000, 0
SETS = data.DOMAINS
SINGLE = lambda i, q, prec, d: f"{i}|{q}|single|{prec}|{d}||0"
TWO = lambda i, q, prec, d, r, n: f"{i}|{q}|two-stage|{prec}|{d}|{r}|{n}"


def load(s, lang, english_pools=False):
    tag = data.subset(s, lang) + ("__judged_by_english" if english_pools and lang != "english" else "")
    with np.load(EVAL / f"{tag}.npz") as z:
        return {k: z[k] for k in z.files}


def boot(values: dict, clusters: dict) -> dict:
    """values/clusters: language -> per-question array. -> estimate, 95% CI, two-sided p (x100 points)."""
    df = pd.concat([pd.DataFrame({"c": clusters[l], "v": v}) for l, v in values.items()]).dropna()
    per = df.groupby("c").v.mean().to_numpy()
    bs = per[np.random.default_rng(SEED).integers(0, len(per), (B, len(per)))].mean(1)
    p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"est": round(100 * per.mean(), 3), "lo": round(100 * lo, 3), "hi": round(100 * hi, 3), "p": p,
            "clusters": len(per)}


def contrast(s, f, langs=data.LANGS, english_pools=False):
    """f(z) -> per-question contrast for one language's arrays; bootstrapped over the given languages."""
    zs = {l: load(s, l, english_pools) for l in langs}
    return boot({l: f(z) for l, z in zs.items()}, {l: z["__en_ids__"] for l, z in zs.items()})


def diff(a, b, m="qrel"):
    return lambda z: z[f"{a}|{m}_corpus"] - z[f"{b}|{m}_corpus"]


def gap(k, m="qrel"):
    return lambda z: z[f"{k}|{m}_shortlist"] - z[f"{k}|{m}_corpus"]


def both(s, f, english_pools=False):
    return {"6L": contrast(s, f, english_pools=english_pools), "EN": contrast(s, f, ("english",))}


def depth_at(depths, recall, target=0.95):
    if recall.max() < target:
        return np.nan
    j = int(np.argmax(recall >= target))
    if j == 0:
        return float(depths[0])
    t = (target - recall[j - 1]) / (recall[j] - recall[j - 1])
    return float(np.exp(np.log(depths[j - 1]) + t * (np.log(depths[j]) - np.log(depths[j - 1]))))


def sweep_fits(s) -> dict:
    """Linear fit depth95 = a + b N over the same-set sizes, per first pass, with a bootstrap over questions (the
    same resample at every size); the 8-bit/float question's depth relative to Hamming's; full float's share."""
    with np.load(SWEEP / f"{s}.npz") as z:
        per_q = {k: z[k] for k in z.files}
    keys = pd.DataFrame([k.split("|") for k in per_q], columns=["N", "rep", "fp", "depth"]).astype(
        {"N": int, "rep": int, "depth": int})
    n_own = len(data.load(s).pages())
    sizes = sorted(n for n in keys.N.unique() if n <= n_own)
    nq = len(next(iter(per_q.values())))
    idx = np.random.default_rng(SEED).integers(0, nq, (1000, nq))
    curves = {}
    for fp in keys.fp.unique():
        for N in sorted(keys.N.unique()):
            k = keys[(keys.fp == fp) & (keys.N == N)]
            depths = np.array(sorted(k.depth.unique()))
            curves[(fp, N)] = (depths, np.stack([np.mean([per_q[f"{N}|{r}|{fp}|{n}"] for r in sorted(k.rep.unique())],
                                                         axis=0) for n in depths]))

    def depth(fp, N, rows=None):
        d, R = curves[(fp, N)]
        return depth_at(d, (R if rows is None else R[:, rows]).mean(1))

    def slope(fp, rows=None):
        y = np.array([depth(fp, N, rows) for N in sizes])
        b, a = np.polyfit(sizes, y, 1)
        r2 = 1 - ((y - (a + b * np.array(sizes))) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return a, b, r2

    out = {}
    for fp in keys.fp.unique():
        a, b, r2 = slope(fp)
        bs = np.array([slope(fp, r)[1] for r in idx])
        out[fp] = {"a": round(a, 1), "b_pct": round(100 * b, 2), "b_lo": round(100 * np.nanpercentile(bs, 2.5), 2),
                   "b_hi": round(100 * np.nanpercentile(bs, 97.5), 2), "r2": round(r2, 4),
                   "share_full_set_pct": round(100 * depth(fp, n_own) / n_own, 2),
                   "depth_full_set": round(depth(fp, n_own), 1)}
        n_all = max(keys.N.unique())
        if n_all > n_own:
            out[fp]["other_sets_increase_pct"] = round(100 * (depth(fp, n_all) / depth(fp, n_own) - 1), 1)
    for fp in ("binary_asym256", "binary_asym8_256", "float2048"):
        less = 1 - out[fp]["b_pct"] / out["binary256"]["b_pct"]
        bs = np.array([1 - slope(fp, r)[1] / slope("binary256", r)[1] for r in idx[:300]])
        out[fp]["slope_vs_hamming_less_pct"] = round(100 * less, 1)
        out[fp]["less_lo"], out[fp]["less_hi"] = (round(100 * np.nanpercentile(bs, q), 1) for q in (2.5, 97.5))
    out["hamming_vs_float_full_set_x"] = round(out["binary256"]["depth_full_set"] / out["float2048"]["depth_full_set"], 2)
    return out


def translated_depth(s) -> dict:
    """Depth for 95% recall of a 32 B Hamming first pass over the whole set, per question language (no sampling)."""
    d = data.load(s)
    P = sy.cached(d.pages().text.tolist(), "pro", "search_document")
    grid = np.unique(np.round(np.geomspace(5, len(P["float"]), 40)).astype(int))
    out = {}
    for lang in data.LANGS:
        qs = metrics.QuerySet.load(s, lang, d)
        Q = sy.cached(d.queries[d.queries.language == lang].text.tolist(), "pro", "search_query")
        S = sy.hamming(Q["ubinary"], P["ubinary"], 256)
        rec = np.array([[metrics.recall(S[i], qs.grade[i] > 0, n) for n in grid] for i in range(len(S))]).mean(0)
        out[lang] = round(depth_at(grid, rec), 1)
    en = out["english"]
    out["more_than_english_pct"] = {l: round(100 * (v / en - 1), 1) for l, v in out.items() if l != "english"}
    return out


def main() -> None:
    R = {}
    for s in SETS:
        r = R[s] = {}
        F = SINGLE("pro", "pro", "float", 2048)
        r["F1_gap_binary256"] = both(s, gap(SINGLE("pro", "pro", "binary", 256)))
        r["F1_gap_binary256_english_pools_6L"] = contrast(s, gap(SINGLE("pro", "pro", "binary", 256)),
                                                         english_pools=True)
        r["F1_gap_float2048"] = both(s, gap(F))
        r["F1_hamming256_vs_float"] = both(s, diff(SINGLE("pro", "pro", "binary", 256), F))
        r["F2_int8_2048_vs_float_2048"] = both(s, diff(SINGLE("pro", "pro", "int8", 2048), F))
        r["F2_int8_1024_vs_float_1024"] = both(s, diff(SINGLE("pro", "pro", "int8", 1024),
                                                       SINGLE("pro", "pro", "float", 1024)))
        r["F2_int8_1024_vs_float_2048"] = both(s, diff(SINGLE("pro", "pro", "int8", 1024), F))
        r["F3_8bit_vs_float_question_32B"] = both(s, diff(SINGLE("pro", "pro", "binary_asym8", 256),
                                                          SINGLE("pro", "pro", "binary_asym", 256)))
        r["F3_8bit_vs_float_question_after_int8_1024_top200"] = both(s, diff(
            TWO("pro", "pro", "binary_asym8", 256, "int8_1024", 200), TWO("pro", "pro", "binary_asym", 256, "int8_1024", 200)))
        I1 = SINGLE("pro", "pro", "int8", 1024)
        r["F4_8bit_int8_1024_top200_vs_rescorer"] = both(s, diff(TWO("pro", "pro", "binary_asym8", 256, "int8_1024", 200), I1))
        r["F4_8bit_int8_1024_top200_vs_float"] = both(s, diff(TWO("pro", "pro", "binary_asym8", 256, "int8_1024", 200), F))
        r["F4_hamming_int8_1024_top200_vs_rescorer"] = both(s, diff(TWO("pro", "pro", "binary", 256, "int8_1024", 200), I1))
        r["F4_8bit_int8_2048_top200_vs_float"] = both(s, diff(TWO("pro", "pro", "binary_asym8", 256, "int8_2048", 200), F))
        r["F4_128B_int8_1024_top100_vs_rescorer"] = both(s, diff(TWO("pro", "pro", "binary", 1024, "int8_1024", 100), I1))
        zs = {l: load(s, l) for l in data.LANGS}
        per = pd.concat([pd.DataFrame({"c": z["__en_ids__"], "v": diff(TWO("pro", "pro", "binary_asym8", 256, "int8_1024", 200), I1)(z)})
                         for z in zs.values()]).groupby("c").v.mean()
        neg = per[per < 0].sort_values()
        r["F4_top3_share_of_8bit_top200_loss_pct"] = round(100 * neg.iloc[:3].sum() / per.sum(), 1) if per.sum() < 0 else None
        from scipy.stats import ttest_1samp
        r["F4_8bit_top200_t_test_p"] = float(ttest_1samp(per, 0).pvalue)
        r["F4_depth200_share_pct"] = round(100 * 200 / len(data.load(s).pages()), 1)
        for ip, qm in (("pro", "fast"), ("fast", "pro"), ("fast", "fast")):
            for prec, d in (("float", 2048), ("int8", 2048), ("binary", 256), ("binary_asym8", 256)):
                r[f"F6_{ip}_index_{qm}_question_{prec}{d}"] = both(s, diff(SINGLE(ip, qm, prec, d), SINGLE("pro", "pro", prec, d)))
        mix = lambda prec, d: diff(SINGLE("pro", "fast", prec, d), SINGLE("pro", "pro", prec, d))
        r["F6_pro_fast_excess_32B_over_float"] = both(s, lambda z: mix("binary", 256)(z) - mix("float", 2048)(z))
        r["F6_pro_fast_32B_rcp"] = both(s, diff(SINGLE("pro", "fast", "binary", 256), SINGLE("pro", "pro", "binary", 256), "rcp"))
        r["F6_pro_fast_32B_english_pools_6L"] = contrast(s, mix("binary", 256), english_pools=True)
        r["F6_pro_fast_32B_per_language"] = {l: contrast(s, mix("binary", 256), (l,)) for l in data.LANGS}
        r["F6_pro_fast_after_hamming_int8_1024_top200"] = both(s, diff(
            TWO("pro", "fast", "binary", 256, "int8_1024", 200), TWO("pro", "pro", "binary", 256, "int8_1024", 200)))
        r["F5_sweep"] = sweep_fits(s)
        r["F5_translated_depth95_hamming"] = translated_depth(s)
        print(f"{s}: done", flush=True)
    # Holm across the 6-language contrasts the sheet reports
    fam = {f"{s}:{k}": v["6L"] for s in SETS for k, v in R[s].items() if isinstance(v, dict) and "6L" in v
           and k.split("_")[0] in ("F1", "F2", "F3", "F4", "F6")}
    order = sorted(fam, key=lambda k: fam[k]["p"])
    m, running = len(order), 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * fam[k]["p"]))
        fam[k]["p_holm"] = running
    R["holm"] = {"family_size": m, "surviving_0.05": [k for k in order if fam[k]["p_holm"] < 0.05]}
    (RESULTS / "sheet_numbers.json").write_text(json.dumps(R, indent=1, default=float))
    print(f"-> {RESULTS / 'sheet_numbers.json'}; Holm family {m}, surviving {len(R['holm']['surviving_0.05'])}")


if __name__ == "__main__":
    main()
