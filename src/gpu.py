"""GPU search over 32-byte Embed 5 codes on Apple silicon: Metal kernels (kernels/search.metal) compiled with
torch.mps.compile_shader, the way the metal-inference-engine compiles its kernels.

  Index(codes).topk(q, k)        exact top-k by Hamming distance (counting select), ties by page index
  Index(codes).asym_topk(qf, k)  top-k by the float question against the page's bits (+-1), via a lookup table
"""
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEV = torch.device("mps")
LIB = torch.mps.compile_shader((ROOT / "kernels" / "search.metal").read_text())
TG = 256                         # threads per threadgroup
NBINS, CAND0 = 257, 260          # the state buffer layout of kernels/search.metal
sync = torch.mps.synchronize


def lut(qf: np.ndarray) -> np.ndarray:
    """The asym kernel's table, float32 [32 * 256]: entry 256 j + v is the score of byte value v at byte position
    j, i.e. sum over its 8 bits of q[8 j + b] * (+1 if the bit is set else -1). Bits are MSB first, as in ubinary.
    qf: the question's first 256 float values (renormalized here, as a 256-dim vector)."""
    q = qf[:256].astype(np.float64)
    q = (q / np.linalg.norm(q)).reshape(32, 8)
    bits = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.float64) * 2 - 1   # [256, 8]
    return (q @ bits.T).astype(np.float32).reshape(-1)                                                # [32, 256]


class Index:
    def __init__(self, codes: np.ndarray, cap: int = 4096):
        """codes: uint8 [N, 32], the pages' 256-bit ubinary codes."""
        assert codes.dtype == np.uint8 and codes.ndim == 2 and codes.shape[1] == 32, codes.shape
        self.n, self.cap = len(codes), cap
        self.codes = torch.from_numpy(np.ascontiguousarray(codes).view(np.int32)).to(DEV)     # [N, 8]
        self.dist = torch.empty(self.n, dtype=torch.int16, device=DEV)
        self.scores = torch.empty(self.n, dtype=torch.float32, device=DEV)
        self.state = torch.zeros(CAND0 + 2 * cap, dtype=torch.int32, device=DEV)

    def _q(self, q: np.ndarray) -> torch.Tensor:
        assert q.dtype == np.uint8 and q.shape == (32,), q.shape
        return torch.from_numpy(np.ascontiguousarray(q).view(np.int32).copy()).to(DEV)

    def hamming(self, q: np.ndarray) -> np.ndarray:
        """Distances from one 32-byte question code to every page, int [N] (for checks; topk does not copy these)."""
        LIB.hamming(self.dist, self.codes, self._q(q), self.n, threads=self.n, group_size=TG)
        return self.dist.cpu().numpy().astype(np.int32)

    def topk(self, q: np.ndarray, k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-k (smallest distances), ordered by (distance, page index). Three kernels and one readback of
        the small candidate list; if more than `cap` pages tie at the cut, the distances are read back instead."""
        k = min(k, self.n)
        self.state[:NBINS + 2].zero_()
        LIB.hamming_hist(self.dist, self.codes, self._q(q), self.state, self.n, threads=self.n, group_size=TG)
        LIB.kth(self.state, k, threads=1, group_size=1)
        LIB.compact(self.dist, self.state, self.n, self.cap, threads=self.n, group_size=TG)
        s = self.state.cpu().numpy()
        count = int(s[NBINS + 1])
        if count > self.cap:                                    # a huge tie at the cut: do it on the CPU, exactly
            d = self.dist.cpu().numpy().astype(np.int32)
            cand = np.flatnonzero(d <= s[NBINS])
            pairs = np.stack([cand, d[cand]], 1)
        else:
            pairs = s[CAND0:CAND0 + 2 * count].reshape(count, 2).astype(np.int64)
        order = np.lexsort((pairs[:, 0], pairs[:, 1]))[:k]
        return pairs[order, 0], pairs[order, 1]

    def asym_scores(self, qf: np.ndarray) -> np.ndarray:
        table = torch.from_numpy(lut(qf)).to(DEV)
        LIB.asym(self.scores, self.codes, table, self.n, threads=self.n, group_size=TG)
        return self.scores.cpu().numpy()

    def asym_topk(self, qf: np.ndarray, k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Top-k by the float question against the bits (highest scores), on the GPU (torch.topk)."""
        table = torch.from_numpy(lut(qf)).to(DEV)
        LIB.asym(self.scores, self.codes, table, self.n, threads=self.n, group_size=TG)
        v, i = torch.topk(self.scores, min(k, self.n))
        return i.cpu().numpy(), v.cpu().numpy()
