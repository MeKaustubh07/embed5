"""src/metrics.py against Cohere's rcp-ndcg: (1) query by query against rcp_ndcg_core.protocol.score_query on
random cases, including the heavy ties of binary (Hamming) scores; (2) Cohere's stored runs reproduce the numbers
rcp-ndcg's CLI reports. Part 2 reads the stored runs from the Hugging Face cache (downloads ~18 MB once).
Run: .venv/bin/python tests/test_metrics.py"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcp_ndcg_core.protocol import score_query  # noqa: E402

import data  # noqa: E402
import metrics  # noqa: E402

passed = 0


def check(cond, what):
    global passed
    assert cond, what
    passed += 1
    print(f"  ok  {what}")


print("query by query against rcp_ndcg_core.protocol.score_query")
rng = np.random.default_rng(0)
worst = {"shortlist": 0.0, "corpus": 0.0, "qrel": 0.0, "recall": 0.0}
for case in range(1200):
    n = int(rng.integers(150, 600))
    ids = [f"corpus-test-{i}" for i in range(n)]
    pool = rng.choice(n, 150, replace=False)
    pool_gain = rng.random(150) ** 3
    if case % 3 == 0:                                   # Hamming-like integer scores: many ties
        s = -rng.integers(80, 140, n).astype(float)
    elif case % 3 == 1:                                 # a few distinct values: huge tie groups
        s = rng.integers(0, 4, n).astype(float)
    else:
        s = rng.standard_normal(n)
    grade = np.zeros(n)
    pos = rng.choice(pool, int(rng.integers(1, 8)), replace=False)
    grade[pos] = rng.integers(1, 3, len(pos))
    k = int(rng.choice([1, 5, 10, 20]))
    scores = dict(zip(ids, s))
    gains = {ids[p]: g for p, g in zip(pool, pool_gain)}
    cand = [ids[p] for p in pool]
    ref_short = score_query(scores, gains, protocol="vidore", k=k, candidates=cand)
    ref_corpus = score_query(scores, gains, protocol="plain", k=k, candidates=cand)
    ref_qrel = score_query(scores, {ids[p]: grade[p] for p in pool}, protocol="plain", k=k, metric="qrel_ndcg",
                           candidates=cand)
    g_all = np.zeros(n)
    g_all[pool] = pool_gain
    worst["shortlist"] = max(worst["shortlist"], abs(metrics.ndcg(s[pool], pool_gain, pool_gain, k) - ref_short))
    worst["corpus"] = max(worst["corpus"], abs(metrics.ndcg(s, g_all, pool_gain, k) - ref_corpus))
    worst["qrel"] = max(worst["qrel"], abs(metrics.ndcg(s, grade, grade[grade > 0], k) - ref_qrel))
    # recall: the expectation over random orders of the straddling tie, by brute force on a small case
    if case < 60:
        rel = grade > 0
        kk = 10
        perms = [np.lexsort((rng.random(n), -s))[:kk] for _ in range(4000)]
        mc = np.mean([rel[p].sum() / rel.sum() for p in perms])
        worst["recall"] = max(worst["recall"], abs(metrics.recall(s, rel, kk) - mc))
check(worst["shortlist"] < 1e-12, f"shortlist (vidore) RCP-nDCG identical (max diff {worst['shortlist']:.1e})")
check(worst["corpus"] < 1e-12, f"corpus (plain) RCP-nDCG identical (max diff {worst['corpus']:.1e})")
check(worst["qrel"] < 1e-12, f"qrel-nDCG identical (max diff {worst['qrel']:.1e})")
check(worst["recall"] < 0.02, f"fractional recall = the expectation over tie orders (max diff {worst['recall']:.3f})")

print("Cohere's stored runs reproduce rcp-ndcg's published numbers")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
from huggingface_hub import hf_hub_download  # noqa: E402

fetch = lambda f: pd.read_parquet(hf_hub_download(data.REPO, f, repo_type="dataset", revision=data.REV))


def stored_matrix(run: pd.DataFrame, qs: metrics.QuerySet) -> np.ndarray:
    """A stored run as a score matrix; pages it did not score get a score below every scored page."""
    col = {p: i for i, p in enumerate(qs.page_ids)}
    row = {q: i for i, q in enumerate(qs.query_ids)}
    run = run[run["query-id"].isin(row) & run["corpus-id"].isin(col)]
    S = np.full((len(row), len(col)), -1e9)
    S[run["query-id"].map(row).to_numpy(), run["corpus-id"].map(col).to_numpy()] = run.score.to_numpy()
    return S


ANCHORS = {"finance_en": 0.7935, "computer_science": 0.7775, "hr": 0.7704}   # rcp-ndcg CLI, Embed v4, native
for dom, want in ANCHORS.items():
    qs = metrics.QuerySet.load(dom, "english")
    fs = fetch(f"provenance/first_stage/{dom}.parquet")
    got = metrics.evaluate(stored_matrix(fs[fs.retriever == "embed_v4"], qs), qs)["rcp_shortlist"].mean()
    check(abs(got - want) < 6e-5, f"Embed v4 on {dom}: {got:.4f} (rcp-ndcg: {want})")
qs = metrics.QuerySet.load("finance_en", "english")
runs = fetch("provenance/runs/finance_en.parquet")
got = metrics.evaluate(stored_matrix(runs[runs.model == "rerank-v4.0-pro"], qs), qs)["rcp_shortlist"].mean()
check(abs(got - 0.8859) < 6e-5, f"Rerank 4 Pro on finance_en: {got:.4f} (rcp-ndcg: 0.8859)")
print(f"{passed} checks passed")
