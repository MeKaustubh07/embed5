"""src/systems.py on synthetic vectors: the Hamming, asymmetric and Matryoshka scores against brute force, and the
two-stage ordering. No network. Run: .venv/bin/python tests/test_systems.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data  # noqa: E402
import systems  # noqa: E402

passed = 0


def check(cond, what):
    global passed
    assert cond, what
    passed += 1
    print(f"  ok  {what}")


rng = np.random.default_rng(0)
qf, pf = rng.standard_normal((70, 2048)), rng.standard_normal((300, 2048))
qb, pb = np.packbits(qf >= 0, axis=1), np.packbits(pf >= 0, axis=1)
for d in (256, 768, 2048):
    brute = -(np.unpackbits(qb[:, :d // 8], axis=1)[:, None, :] != np.unpackbits(pb[:, :d // 8], axis=1)[None]).sum(-1)
    check(np.array_equal(systems.hamming(qb, pb, d), brute), f"hamming at {d} bits == brute force")
rn = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
check(np.allclose(systems.cosine(qf, pf, 512), rn(qf[:, :512]) @ rn(pf[:, :512]).T, atol=1e-5),
      "cosine at 512 == renormalized prefix")
check(np.allclose(systems.binary_asym(qf, pb, 256), rn(qf[:, :256]) @ np.where(pf[:, :256] >= 0, 1, -1).T, atol=1e-5),
      "asymmetric binary == float question . sign(page)")
qi, pi = rng.integers(-128, 128, (5, 64)), rng.integers(-128, 128, (9, 64))
check(np.array_equal(systems.int8_dot(qi.astype(np.int8), pi.astype(np.int8)), qi @ pi.T), "int8 dot accumulates exactly")

first = systems.hamming(qb, pb, 256)
for name, rescore in (("cosine", systems.cosine(qf, pf, 2048)),
                      ("int8 dots, mostly negative", -1e6 + 1e3 * systems.cosine(qf, pf, 2048))):
    S = systems.two_stage(first, rescore, 40, seed=7)
    ok = True
    for i in range(len(S)):
        kept, dropped = np.flatnonzero(S[i] >= 1), np.flatnonzero(S[i] <= -1)
        ok &= len(kept) == 40 and len(dropped) == 260
        ok &= first[i, kept].min() >= first[i, dropped].max()               # the top 40 by the first pass
        ok &= np.array_equal(np.argsort(-S[i, kept], kind="stable"), np.argsort(-rescore[i, kept], kind="stable"))
        ok &= np.array_equal(np.argsort(-S[i, dropped], kind="stable"), np.argsort(-first[i, dropped], kind="stable"))
    check(ok, f"two-stage ({name} rescore): the top 40 kept above the rest, rescore order inside, first-pass below")
tie = np.zeros((1, 10))
kept = np.array([systems.two_stage(tie, np.arange(10.0)[None], 3, seed=s)[0] >= 1 for s in range(3000)])
freq = kept.mean(0)
check((kept.sum(1) == 3).all() and np.abs(freq - 0.3).max() < 0.04,
      f"a 10-way tie at the cut keeps exactly 3, each page equally often (no index bias: {freq.min():.2f}-{freq.max():.2f})")
from fractions import Fraction  # noqa: E402

half = np.array([[127, 63.5, -63.5, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5] + [0.0] * 246], np.float32)
want = [127, 64, -64, 0, 2, 2, 0, -2, -2, 126]                       # halves to even
check(systems.quantize(half)[0, :10].tolist() == want, "the quantizer rounds exact halves to even")
real = np.concatenate([systems.cached(data.load(s).queries.text.tolist(), "pro", "search_query",
                                      types=("float",))["float"][:, :256] for s in data.DOMAINS])
qi = systems.quantize(real)
ok, halves = True, 0
for v, row in zip(real, qi):
    top = max(abs(Fraction(float(x))) for x in v)
    for x, got in zip(v, row):
        exact = Fraction(float(x)) * 127 / top
        halves += exact.denominator == 2
        ok &= round(exact) == got                                       # Python rounds halves to even
check(ok, f"quantize == exact rational rounding on all {len(real):,} real questions ({halves} exact halves)")
print(f"{passed} checks passed")
