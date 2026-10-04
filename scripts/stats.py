"""95% confidence intervals for the contrasts "32 Bytes" reports, from the per-question values of
scripts/evaluate.py (cache/eval/*.npz). Paired bootstrap, 10,000 resamples, fixed seed. A question and its five
translations are one cluster (en_id): resampling them as independent rows would make the intervals 1.4-2x too
narrow. "all languages" averages each cluster over its languages, then the clusters. "Exact" below means exact
relative to ViDoRe's labels (VLM-pre-filtered, then human-reviewed; not exhaustive).

Contrasts (per set):
  gap        shortlist minus corpus nDCG@10 of one index (how much the shortlist flatters it); the corpus RCP counts
             never-judged pages as 0, so the RCP gap is an upper bound and the qrel gap (human grades) is exact
  gap_excess the gap minus the full-float gap: the part that comes from the compression
  vs_float   a system's corpus nDCG@10 minus the full-float index's (two-stage and binary_asym rows)
  vs_rescorer a two-stage system minus its re-scorer scoring every page: the cost of the first pass alone
  mix        an (index model, question model) pair minus Pro index + Pro question at the same storage
  paired     int8 vs float at the same length; the 8-bit vs the float question (alone and before each re-check)
Writes results/stats.csv. Usage: python scripts/stats.py [set ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402

PER_QUERY, RESULTS = ROOT / "cache" / "eval", ROOT / "results"
RESAMPLES, SEED = 10_000, 0
FLOAT = "pro|pro|single|float|2048||0"


def load(subset: str) -> dict:
    with np.load(PER_QUERY / f"{subset}.npz") as z:
        return {k: z[k] for k in z.files}


def cluster_ci(values: dict[str, np.ndarray], en_ids: dict[str, np.ndarray]) -> tuple[float, float, float, int]:
    """values: language -> per-question contrast; en_ids: language -> each question's cluster. -> (estimate,
    low, high, clusters): each cluster's mean over its languages, then a bootstrap over clusters."""
    df = pd.concat([pd.DataFrame({"c": en_ids[l], "v": v}) for l, v in values.items()]).dropna()
    per = df.groupby("c").v.mean().to_numpy()
    rng = np.random.default_rng(SEED)
    boot = per[rng.integers(0, len(per), (RESAMPLES, len(per)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(per.mean()), float(lo), float(hi), len(per)


def contrasts(z: dict) -> dict[str, dict[str, np.ndarray]]:
    """-> {contrast name: {metric: per-question values}} for one subset (Pro index, Pro questions)."""
    out = {}
    f = {m: z[f"{FLOAT}|{m}"] for m in ("qrel_shortlist", "qrel_corpus", "rcp_shortlist", "rcp_corpus")}
    for prec, d in (("float", 2048), ("int8", 2048), ("int8", 1024), ("binary", 2048), ("binary", 1024),
                    ("binary", 512), ("binary", 256), ("binary_asym", 256), ("binary_asym8", 256)):
        k = f"pro|pro|single|{prec}|{d}||0"
        for m in ("qrel", "rcp"):
            gap = z[f"{k}|{m}_shortlist"] - z[f"{k}|{m}_corpus"]
            out.setdefault(f"gap {prec} {d}", {})[m] = gap
            out.setdefault(f"gap_excess {prec} {d}", {})[m] = gap - (f[f"{m}_shortlist"] - f[f"{m}_corpus"])
            out.setdefault(f"vs_float {prec} {d}", {})[m] = z[f"{k}|{m}_corpus"] - f[f"{m}_corpus"]
    # int8 against float at the same length, and the 8-bit question against the float question (paired, same pages)
    pairs = [("int8 1024 vs float 1024", "pro|pro|single|int8|1024||0", "pro|pro|single|float|1024||0"),
             ("int8 2048 vs float 2048", "pro|pro|single|int8|2048||0", FLOAT),
             ("8-bit question vs float question, 32 B", "pro|pro|single|binary_asym8|256||0",
              "pro|pro|single|binary_asym|256||0")]
    for name, a, b in pairs:
        if f"{a}|qrel_corpus" in z:
            out[name] = {m: z[f"{a}|{m}_corpus"] - z[f"{b}|{m}_corpus"] for m in ("qrel", "rcp")}
    # Pro/Fast mixing: each (index, question) model pair minus Pro index + Pro question, at the same storage
    for prec, d in (("float", 2048), ("int8", 2048), ("float", 256), ("binary", 2048), ("binary", 1024),
                    ("binary", 256), ("binary_asym8", 256)):
        pp = f"pro|pro|single|{prec}|{d}||0"
        for ip, qm in (("pro", "fast"), ("fast", "pro"), ("fast", "fast")):
            k = f"{ip}|{qm}|single|{prec}|{d}||0"
            if f"{k}|qrel_corpus" in z:
                out[f"mix {ip} index, {qm} question, {prec} {d}"] = {
                    m: z[f"{k}|{m}_corpus"] - z[f"{pp}|{m}_corpus"] for m in ("qrel", "rcp")}
    alone = {"float2048": "pro|pro|single|float|2048||0", "int8_2048": "pro|pro|single|int8|2048||0",
             "int8_1024": "pro|pro|single|int8|1024||0"}
    two_stage = [k[:-len("|qrel_corpus")] for k in z if k.endswith("|qrel_corpus") and k.startswith("pro|pro|two-stage|")]
    for key in two_stage:                     # Pro index + Pro questions only (the names carry no model pair)
        _, _, _, prec, d, rescorer, depth = key.split("|")
        name = f"{prec}{'_' if prec[-1].isdigit() else ''}{d} -> {rescorer} top {depth}"   # binary_asym8_256
        out[f"vs_float {name}"] = {m: z[f"{key}|{m}_corpus"] - f[f"{m}_corpus"] for m in ("qrel", "rcp")}
        # against the re-scorer scoring every page: what the first pass alone costs
        out[f"vs_rescorer {name}"] = {m: z[f"{key}|{m}_corpus"] - z[f"{alone[rescorer]}|{m}_corpus"]
                                      for m in ("qrel", "rcp")}
        if prec == "binary_asym8":                 # the 8-bit question against the float question, same re-check
            twin = key.replace("|binary_asym8|", "|binary_asym|")
            out[f"8-bit vs float question, {name}"] = {m: z[f"{key}|{m}_corpus"] - z[f"{twin}|{m}_corpus"]
                                                       for m in ("qrel", "rcp")}
    return out


def run_set(name: str) -> list[dict]:
    zs = {lang: load(data.subset(name, lang)) for lang in data.LANGS}
    cs = {lang: contrasts(z) for lang, z in zs.items()}
    en_ids = {lang: z["__en_ids__"] for lang, z in zs.items()}
    rows = []
    for c in cs["english"]:
        for m in cs["english"][c]:
            for scope, langs in (("english", ["english"]), ("all languages", list(data.LANGS))):
                est, lo, hi, n = cluster_ci({l: cs[l][c][m] for l in langs}, {l: en_ids[l] for l in langs})
                rows.append({"set": name, "contrast": c, "metric": m, "scope": scope, "estimate": est,
                             "ci_low": lo, "ci_high": hi, "clusters": n})
    return rows


def main(sets: list[str]) -> None:
    rows = [r for s in sets if (PER_QUERY / f"{data.subset(s, 'english')}.npz").exists() for r in run_set(s)]
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "stats.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old.set.isin(df.set.unique())], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(data.DOMAINS))
