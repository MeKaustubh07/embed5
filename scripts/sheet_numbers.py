"""Every number results/RESULTS_SHEET.md cites, recomputed from the scored data (cache/eval, cache/sweep and
results/*.csv; no API calls), so the sheet can be checked and regenerated. Points = qrel-nDCG@10 x 100 unless a key
says rcp. Intervals: paired bootstrap over questions, a question and its translations one cluster (10,000 resamples,
seed 0, as scripts/stats.py); two-sided bootstrap p-values from 100,000 resamples (10,000 is too coarse near a Holm
threshold); Holm across the 6-language contrasts the sheet reports. Two-stage numbers are the expectation over random
tie orders at the cut (20 orders; scripts/evaluate.py averages 3). Values are stored unrounded: round once, when
writing the sheet.
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
B, B_P, SEED, TIE_ORDERS = 10_000, 100_000, 0, 20
SETS = data.DOMAINS
SINGLE = lambda i, q, prec, d: f"{i}|{q}|single|{prec}|{d}||0"


def load(s, lang, english_pools=False):
    tag = data.subset(s, lang) + ("__judged_by_english" if english_pools and lang != "english" else "")
    with np.load(EVAL / f"{tag}.npz") as z:
        return {k: z[k] for k in z.files}


def boot(values: dict, clusters: dict) -> dict:
    """values/clusters: language -> per-question array. -> estimate, 95% CI, two-sided p (x100 points)."""
    df = pd.concat([pd.DataFrame({"c": clusters[l], "v": v}) for l, v in values.items()]).dropna()
    per = df.groupby("c").v.mean().to_numpy()
    rng = np.random.default_rng(SEED)
    bs = per[rng.integers(0, len(per), (B, len(per)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    bp = np.concatenate([bs] + [per[rng.integers(0, len(per), (B, len(per)))].mean(1) for _ in range(B_P // B - 1)])
    p = max(2 * min((bp <= 0).mean(), (bp >= 0).mean()), 1 / B_P)
    return {"est": 100 * per.mean(), "lo": 100 * lo, "hi": 100 * hi, "p": p, "clusters": len(per)}


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


class Vectors:
    """One set's cached vectors and question sets, for numbers computed straight from the scores."""

    def __init__(self, s):
        self.d = data.load(s)
        pages = self.d.pages().text.tolist()
        self.P = {m: sy.cached(pages, m, "search_document") for m in sy.MODELS}
        self.P1k = {"pro": sy.cached(pages, "pro", "search_document", 1024, ("int8",))}
        self.Q = {l: {m: sy.cached(self.d.queries[self.d.queries.language == l].text.tolist(), m, "search_query")
                      for m in sy.MODELS} for l in data.LANGS}
        self.qs = {l: metrics.QuerySet.load(s, l, self.d) for l in data.LANGS}
        self.memo = {}

    def scores(self, lang, ip, qm, prec, d):
        if prec == "int8" and d == 1024:
            return sy.scores("int8", self.Q[lang][qm], self.P1k[ip], 1024)
        return sy.scores(prec, self.Q[lang][qm], self.P[ip], d)

    def two_stage(self, lang, ip, qm, fprec, fd, rescorer, n):
        """Per-question qrel-nDCG@10 of a two-stage search, averaged over TIE_ORDERS random tie orders at the cut."""
        key = (lang, ip, qm, fprec, fd, rescorer, n)
        if key not in self.memo:
            first = self.scores(lang, ip, qm, fprec, fd)
            rprec, rd = rescorer.split("_")[0], int(rescorer.split("_")[1].replace("float", ""))
            res = self.scores(lang, ip, qm, rprec, rd)
            g = self.qs[lang].grade
            self.memo[key] = np.mean([[metrics.ndcg(S[i], g[i], g[i][g[i] > 0]) for i in range(len(S))]
                                      for S in (sy.two_stage(first, res, n, seed) for seed in range(TIE_ORDERS))], axis=0)
        return self.memo[key]

    def unjudged_float_rank(self, langs):
        """Where full float ranks the never-judged pages that 32 B Hamming puts in its top 10 (median over all)."""
        ranks, rng = [], np.random.default_rng(SEED)
        for l in langs:
            H, F = self.scores(l, "pro", "pro", "binary", 256), self.scores(l, "pro", "pro", "float", 2048)
            for i in range(len(H)):
                top = np.lexsort((rng.random(H.shape[1]), -H[i]))[:10]
                out = top[~np.isin(top, self.qs[l].pool[i])]
                ranks += [1 + int((F[i] > F[i, j]).sum()) for j in out]
        return float(np.median(ranks)) if ranks else None


def tboot(s, a, b, langs=data.LANGS):
    """Contrast of two per-language arrays (dicts lang -> per-question values, x1) with the clustered bootstrap."""
    zs = {l: load(s, l) for l in langs}
    return boot({l: a[l] - b[l] for l in langs}, {l: zs[l]["__en_ids__"] for l in langs})


def tboth(s, a, b):
    return {"6L": tboot(s, a, b), "EN": tboot(s, a, b, ("english",))}


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
        out[fp] = {"a": a, "b_pct": 100 * b, "b_lo": 100 * np.nanpercentile(bs, 2.5),
                   "b_hi": 100 * np.nanpercentile(bs, 97.5), "r2": r2,
                   "share_full_set_pct": 100 * depth(fp, n_own) / n_own, "depth_full_set": depth(fp, n_own)}
        n_all = max(keys.N.unique())
        if n_all > n_own:
            out[fp]["other_sets_increase_pct"] = 100 * (depth(fp, n_all) / depth(fp, n_own) - 1)
    for fp in ("binary_asym256", "binary_asym8_256", "float2048"):
        less = 1 - slope(fp)[1] / slope("binary256")[1]
        bs = np.array([1 - slope(fp, r)[1] / slope("binary256", r)[1] for r in idx[:300]])
        out[fp]["slope_vs_hamming_less_pct"] = 100 * less
        out[fp]["less_lo"], out[fp]["less_hi"] = (100 * np.nanpercentile(bs, q) for q in (2.5, 97.5))
    out["hamming_vs_float_full_set_x"] = out["binary256"]["depth_full_set"] / out["float2048"]["depth_full_set"]
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
        out[lang] = depth_at(grid, rec)
    en = out["english"]
    out["more_than_english_pct"] = {l: 100 * (v / en - 1) for l, v in out.items() if l != "english"}
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
        V = Vectors(s)
        T = lambda ip, qm, fprec, fd, rs, n: {l: V.two_stage(l, ip, qm, fprec, fd, rs, n) for l in data.LANGS}
        S1 = lambda k: {l: load(s, l)[f"{k}|qrel_corpus"] for l in data.LANGS}
        a8_1k = T("pro", "pro", "binary_asym8", 256, "int8_1024", 200)
        r["F3_8bit_vs_float_question_after_int8_1024_top200"] = tboth(s, a8_1k, T("pro", "pro", "binary_asym", 256, "int8_1024", 200))
        I1, FL = S1(SINGLE("pro", "pro", "int8", 1024)), S1(F)
        r["F4_8bit_int8_1024_top200_vs_rescorer"] = tboth(s, a8_1k, I1)
        r["F4_8bit_int8_1024_top200_vs_float"] = tboth(s, a8_1k, FL)
        ham_1k = T("pro", "pro", "binary", 256, "int8_1024", 200)
        r["F4_hamming_int8_1024_top200_vs_rescorer"] = tboth(s, ham_1k, I1)
        r["F4_8bit_int8_2048_top200_vs_float"] = tboth(s, T("pro", "pro", "binary_asym8", 256, "int8_2048", 200), FL)
        r["F4_128B_int8_1024_top100_vs_rescorer"] = tboth(s, T("pro", "pro", "binary", 1024, "int8_1024", 100), I1)
        r["F1_unjudged_in_hamming_top10_float_rank_median"] = {"6L": V.unjudged_float_rank(data.LANGS),
                                                               "EN": V.unjudged_float_rank(("english",))}
        zs = {l: load(s, l) for l in data.LANGS}
        per = pd.concat([pd.DataFrame({"c": zs[l]["__en_ids__"], "v": a8_1k[l] - I1[l]})
                         for l in data.LANGS]).groupby("c").v.mean()
        neg = per[per < 0].sort_values()
        r["F4_top3_share_of_8bit_top200_loss_pct"] = 100 * neg.iloc[:3].sum() / per.sum() if per.sum() < 0 else None
        from scipy.stats import ttest_1samp
        r["F4_8bit_top200_t_test_p"] = float(ttest_1samp(per, 0).pvalue)
        r["F4_depth200_share_pct"] = 100 * 200 / len(data.load(s).pages())
        for ip, qm in (("pro", "fast"), ("fast", "pro"), ("fast", "fast")):
            for prec, d in (("float", 2048), ("int8", 2048), ("binary", 256), ("binary_asym8", 256)):
                r[f"F6_{ip}_index_{qm}_question_{prec}{d}"] = both(s, diff(SINGLE(ip, qm, prec, d), SINGLE("pro", "pro", prec, d)))
        mix = lambda prec, d: diff(SINGLE("pro", "fast", prec, d), SINGLE("pro", "pro", prec, d))
        r["F6_pro_fast_excess_32B_over_float"] = both(s, lambda z: mix("binary", 256)(z) - mix("float", 2048)(z))
        r["F6_pro_fast_32B_rcp"] = both(s, diff(SINGLE("pro", "fast", "binary", 256), SINGLE("pro", "pro", "binary", 256), "rcp"))
        r["F6_pro_fast_32B_english_pools_6L"] = contrast(s, mix("binary", 256), english_pools=True)
        r["F6_pro_fast_32B_per_language"] = {l: contrast(s, mix("binary", 256), (l,)) for l in data.LANGS}
        r["F6_pro_fast_after_hamming_int8_1024_top200"] = tboth(s, T("pro", "fast", "binary", 256, "int8_1024", 200), ham_1k)
        r["F5_sweep"] = sweep_fits(s)
        r["F5_translated_depth95_hamming"] = translated_depth(s)
        print(f"{s}: done", flush=True)
    # Holm across the 6-language contrasts the sheet reports
    fam = {f"{s}:{k}": v["6L"] for s in SETS for k, v in R[s].items() if isinstance(v, dict)
           and isinstance(v.get("6L"), dict) and "p" in v["6L"] and k.split("_")[0] in ("F1", "F2", "F3", "F4", "F6")}
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
