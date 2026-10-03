"""src/systems.py on synthetic vectors: the Hamming, asymmetric and Matryoshka scores against brute force, and the
two-stage ordering. No network. Run: .venv/bin/python tests/test_systems.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
    S = systems.two_stage(first, rescore, 40)
    ok = True
    for i in range(len(S)):
        kept = np.argsort(-first[i], kind="stable")[:40]
        dropped = np.setdiff1d(np.arange(300), kept)
        ok &= S[i, kept].min() > S[i, dropped].max()                          # every kept page above every dropped
        ok &= np.array_equal(np.argsort(-S[i, kept], kind="stable"), np.argsort(-rescore[i, kept], kind="stable"))
        ok &= np.array_equal(np.argsort(-S[i, dropped], kind="stable"), np.argsort(-first[i, dropped], kind="stable"))
    check(ok, f"two-stage ({name} rescore): kept above dropped, rescore order inside, first-pass order below")
tie = np.zeros((1, 10))
check(set(np.flatnonzero(systems.two_stage(tie, np.arange(10.0)[None], 3)[0] >= 1)) == {0, 1, 2},
      "a tie at the cut keeps the first pages by index (no relevance leak)")
print(f"{passed} checks passed")
