"""Retrieval metrics for "32 Bytes", vectorised with numpy. RCP-nDCG@k is computed exactly as Cohere's rcp-ndcg
computes it: linear gains, tied pages credited their tie group's mean gain, the ideal DCG from the query's whole
judged pool (tests/test_metrics.py checks every query against rcp_ndcg_core.protocol.score_query).

Two protocols, for the question this project asks:
  shortlist  rcp-ndcg's "vidore" protocol: only the query's 150-page judged pool is ranked. This is what Cohere's
             chart measures: re-ordering a shortlist that other search systems picked.
  corpus     rcp-ndcg's "plain" protocol: every page of the set is ranked, and a page outside the pool has gain 0.
             With no shortlist, a page the pool never saw can take a top-10 slot. For RCP-nDCG that is a lower
             bound (such a page could still be relevant); for qrel-nDCG (human grades) it is exact, because every
             page a human marked relevant is in the pool.
Recall@k counts the human-relevant pages in the top k, a tie straddling the cutoff counted fractionally (the
expected recall over the tie's orders).
"""
from dataclasses import dataclass

import numpy as np

import data


def _discounts(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2))


def ndcg(scores: np.ndarray, gains: np.ndarray, ideal: np.ndarray, k: int = 10) -> float:
    """nDCG@k of one query: scores and gains over the pages that enter the ranking; ideal: the gains the ideal DCG
    sorts (the whole judged pool). Ties are credited their group's mean gain (rcp-ndcg's "group_mean")."""
    if len(scores) == 0:
        return 0.0
    kk = min(k, len(scores))
    t = np.partition(scores, len(scores) - kk)[len(scores) - kk]      # the k-th highest score
    sel = np.flatnonzero(scores >= t)                                   # every tie group reaching the top k, whole
    sel = sel[np.argsort(-scores[sel], kind="stable")]
    s, g = scores[sel], gains[sel]
    group = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])
    mean = np.bincount(group, weights=g) / np.bincount(group)
    dcg = float(mean[group[:kk]] @ _discounts(kk))
    best = np.sort(ideal)[::-1][:k]
    norm = float(best @ _discounts(len(best)))
    return dcg / norm if norm > 0 else 0.0


def recall(scores: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Share of the relevant pages (bool mask) in the top k; a tie straddling the cutoff counts fractionally."""
    n_rel = int(relevant.sum())
    if n_rel == 0:
        return float("nan")
    if k >= len(scores):
        return 1.0
    t = np.partition(scores, len(scores) - k)[len(scores) - k]
    above, at = scores > t, scores == t
    slots = k - int(above.sum())
    return (float(relevant[above].sum()) + float(relevant[at].sum()) * slots / int(at.sum())) / n_rel


@dataclass
class QuerySet:
    """One set in one language: what every system is scored against."""
    subset: str                 # rcp-ndcg's name, e.g. "hr__german"
    page_ids: list[str]         # columns of a score matrix: the set's non-empty pages
    query_ids: list[str]        # rows
    en_ids: list[str]           # each question's English original (rows of a language align through it)
    pool: np.ndarray            # [nq, 150] column index of each judged page
    pool_gain: np.ndarray       # [nq, 150] its RCP gain
    grade: np.ndarray           # [nq, n_pages] human grade (0/1/2; 0 for unjudged pages)

    @classmethod
    def load(cls, domain: str, lang: str, d: "data.Domain | None" = None) -> "QuerySet":
        d = d or data.load(domain)
        pages = d.pages().id.tolist()
        col = {p: i for i, p in enumerate(pages)}
        q = d.queries[d.queries.language == lang]
        qr = d.qrels[d.qrels.language == lang]
        pools = d.pools[d.pools.language == lang].set_index("query-id")["corpus-ids"]
        gain = qr.set_index(["query-id", "corpus-id"]).gain
        n = len(q)
        pool, pool_gain = np.zeros((n, 150), np.int64), np.zeros((n, 150))
        grade = np.zeros((n, len(pages)))
        for i, qid in enumerate(q.id):
            ids = list(pools[qid])
            assert len(ids) == 150 and all(p in col for p in ids), (qid, "pool page missing or empty")
            pool[i] = [col[p] for p in ids]
            pool_gain[i] = gain.loc[[(qid, p) for p in ids]].to_numpy()
        pos = qr[qr.score > 0]
        for qid, p, s in zip(pos["query-id"], pos["corpus-id"], pos.score):
            grade[q.id.tolist().index(qid), col[p]] = s
        return cls(data.subset(domain, lang), pages, q.id.tolist(), q.en_id.tolist(), pool, pool_gain, grade)


def evaluate(S: np.ndarray, qs: QuerySet, k: int = 10, recall_at: tuple[int, ...] = (10, 100)) -> dict:
    """Per-query metrics of a score matrix S [n_queries, n_pages] (higher = more similar):
    rcp_shortlist, rcp_corpus, qrel_corpus (all @k) and recall@n (full corpus), each an array over queries."""
    assert S.shape == (len(qs.query_ids), len(qs.page_ids)), S.shape
    S = S.astype(np.float64)
    out = {m: np.zeros(len(S)) for m in ("rcp_shortlist", "rcp_corpus", "qrel_corpus")}
    out |= {f"recall@{n}": np.zeros(len(S)) for n in recall_at}
    for i, s in enumerate(S):
        pg = qs.pool_gain[i]
        out["rcp_shortlist"][i] = ndcg(s[qs.pool[i]], pg, pg, k)
        g = np.zeros(len(s))
        g[qs.pool[i]] = pg
        out["rcp_corpus"][i] = ndcg(s, g, pg, k)
        grade = qs.grade[i]
        out["qrel_corpus"][i] = ndcg(s, grade, grade[grade > 0], k)
        for n in recall_at:
            out[f"recall@{n}"][i] = recall(s, grade > 0, n)
    return out
