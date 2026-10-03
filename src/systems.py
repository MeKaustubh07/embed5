"""Score matrices [questions, pages] for every way of storing the index, built from the cached Embed 5 vectors (no
API calls). Every size derives from the 2048 vectors (the probe: a d-dim vector is the renormalized first d numbers;
the binary code is the first d/8 bytes of the sign bits). Embed 5's int8 cannot be derived, so it is read at the
size it was requested.

Precisions (bytes per page at dim d):
  float        4d   cosine
  int8         d    the float question against the page's int8 (API int8 at 2048 and 1024 only). A question is
                    never stored, so it stays float (int8_dot below scores int8 x int8 for comparison)
  binary       d/8  minus the Hamming distance between the question's and the page's sign bits (what a GPU scans)
  binary_asym  d/8  the float question against the page's sign bits as +-1: same index, a float query
"""
import numpy as np

import client as ec

MODELS = {"pro": "embed-v5.0-pro", "fast": "embed-v5.0-fast"}
DIMS = (256, 512, 768, 1024, 1536, 2048)
TYPES = ("float", "int8", "ubinary")
BYTES = {"float": lambda d: 4 * d, "int8": lambda d: d, "binary": lambda d: d // 8, "binary_asym": lambda d: d // 8}


def cached(texts: list[str], model: str, input_type: str, dim: int = 2048, types=TYPES,
           cache: ec.Cache | None = None) -> dict[str, np.ndarray]:
    """Vectors from the cache only; a missing text raises (run scripts/embed_all.py for that chunk)."""
    cache = cache or ec.Cache()
    out = {}
    for e in types:
        rows = [cache.get(ec.cache_key(MODELS[model], input_type, dim, e, t), e) for t in texts]
        missing = sum(r is None for r in rows)
        if missing:
            raise KeyError(f"{missing}/{len(texts)} {model} {input_type} {dim} {e} vectors not embedded yet")
        out[e] = np.stack(rows)
    return out


def renorm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cosine(qf: np.ndarray, pf: np.ndarray, d: int) -> np.ndarray:
    return renorm(qf[:, :d]) @ renorm(pf[:, :d]).T


def hamming(qb: np.ndarray, pb: np.ndarray, d: int, rows: int = 64) -> np.ndarray:
    """Minus the Hamming distance between d-bit codes (ubinary bytes): higher = more similar, integers."""
    q, p = qb[:, :d // 8], pb[:, :d // 8]
    out = np.empty((len(q), len(p)), np.float32)
    for i in range(0, len(q), rows):                               # bounds the [rows, pages, bytes] XOR
        # sum into int32: the counts are uint8, and negating an unsigned sum wraps around instead
        out[i:i + rows] = -np.bitwise_count(q[i:i + rows, None, :] ^ p[None, :, :]).sum(-1, dtype=np.int32)
    return out


def binary_asym(qf: np.ndarray, pb: np.ndarray, d: int) -> np.ndarray:
    """The float question (renormalized at d) against the page's bits read as +-1."""
    signs = np.unpackbits(pb[:, :d // 8], axis=1).astype(np.float32) * 2 - 1
    return renorm(qf[:, :d]) @ signs.T


def int8_dot(qi: np.ndarray, pi: np.ndarray) -> np.ndarray:
    return (qi.astype(np.int32) @ pi.astype(np.int32).T).astype(np.float64)


def scores(precision: str, q: dict, p: dict, d: int) -> np.ndarray:
    """One system's score matrix. q, p: cached() vectors of the questions and pages (a page's int8 at its size)."""
    if precision == "float":
        return cosine(q["float"], p["float"], d)
    if precision == "binary":
        return hamming(q["ubinary"], p["ubinary"], d)
    if precision == "binary_asym":
        return binary_asym(q["float"], p["ubinary"], d)
    if precision == "int8":
        assert p["int8"].shape[1] == d, "a page's int8 is only read at the size it was requested"
        return renorm(q["float"][:, :d]) @ p["int8"].astype(np.float32).T
    raise ValueError(precision)


def two_stage(first: np.ndarray, rescore: np.ndarray, n: int) -> np.ndarray:
    """A first pass keeps each question's top n pages, the rescore orders them; every kept page ranks above every
    dropped one, which keeps its first-pass score. Ties at the cut are broken by page index (corpus order, which
    carries no relevance), so exactly n pages are rescored."""
    keep = np.argsort(-first, axis=1, kind="stable")[:, :n]
    rows = np.arange(len(first))[:, None]
    first, rescore = first.astype(np.float64), rescore.astype(np.float64)
    out = first - first.max() - 1.0                       # dropped pages: <= -1, in first-pass order
    kept = rescore[rows, keep]
    out[rows, keep] = kept - kept.min() + 1.0             # kept pages: >= 1, in rescore order (shifts keep ties)
    return out
