"""src/gpu.py on the Metal GPU: the Hamming scan bit-exact, the counting top-k exact with its tie-break (and its
overflow path), the lookup-table asym scores against numpy, and both against the CPU scoring of src/systems.py on
real Embed 5 codes (hr, English, Pro). No network. Run: .venv/bin/python tests/test_gpu.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gpu  # noqa: E402
import systems as sy  # noqa: E402

passed = 0


def check(cond, what):
    global passed
    assert cond, what
    passed += 1
    print(f"  ok  {what}")


rng = np.random.default_rng(0)
print("random codes")
N = 100_003                                                     # not a multiple of the threadgroup size
codes = rng.integers(0, 256, (N, 32), dtype=np.uint8)
codes[rng.choice(N, 300, replace=False)] = codes[7]             # duplicates: ties at small distances
index = gpu.Index(codes)
q = codes[7] ^ np.where(rng.random(32) < 0.05, 1, 0).astype(np.uint8)
ref = np.bitwise_count(codes ^ q).sum(1, dtype=np.int32)
check(np.array_equal(index.hamming(q), ref), "Hamming distances bit-exact against numpy")
for k in (1, 100, 1000):
    idx, d = index.topk(q, k)
    want = np.lexsort((np.arange(N), ref))[:k]
    check(np.array_equal(idx, want) and np.array_equal(d, ref[want]), f"top-{k} exact, ties broken by page index")
small = gpu.Index(codes, cap=64)                                # 301 pages tie at the cut: more than cap
idx, d = small.topk(q, 100)
check(np.array_equal(idx, np.lexsort((np.arange(N), ref))[:100]), "candidate overflow falls back, still exact")
qf = rng.standard_normal(2048).astype(np.float32)
signs = np.unpackbits(codes, axis=1).astype(np.float64) * 2 - 1
want = signs @ (qf[:256] / np.linalg.norm(qf[:256]))
got = index.asym_scores(qf)
check(np.abs(got - want).max() < 1e-4, f"asym scores = numpy (max diff {np.abs(got - want).max():.1e})")
idx, v = index.asym_topk(qf, 100)
check(set(idx) == set(np.argsort(-want)[:100]), "asym top-100 = numpy's")

print("the bit-serial asym kernel (8-bit question, radix select)")
ref8 = sy.binary_asym8(qf[None], codes, 256)[0].astype(np.int64)
check(np.array_equal(index.asym_bits_scores(qf), ref8), "bit-serial scores == sum(qi * bit), exactly")
for k in (1, 100, 1000):
    idx, sc = index.asym_bits_topk(qf, k)
    want = np.lexsort((np.arange(N), -ref8))[:k]
    check(np.array_equal(idx, want) and np.array_equal(sc, ref8[want]), f"asym top-{k} exact, ties by page index")
tied = codes.copy()
tied[rng.choice(N, 5000, replace=False)] = codes[np.argmax(ref8)]     # 5,001 pages share the top score
tix = gpu.Index(tied, cap=1024)
r8 = sy.binary_asym8(qf[None], tied, 256)[0].astype(np.int64)
idx, sc = tix.asym_bits_topk(qf, 100)
check(np.array_equal(idx, np.lexsort((np.arange(N), -r8))[:100]), "a tie over the candidate cap falls back, still exact")

print("real Embed 5 codes (hr, English questions, Pro)")
import data  # noqa: E402

d = data.load("hr")
P = sy.cached(d.pages().text.tolist(), "pro", "search_document")
Q = sy.cached(d.queries[d.queries.language == "english"].text.tolist()[:40], "pro", "search_query")
idx_real = gpu.Index(np.ascontiguousarray(P["ubinary"][:, :32]))
cpu_h = -sy.hamming(Q["ubinary"], P["ubinary"], 256)
cpu_a = sy.binary_asym(Q["float"], P["ubinary"], 256)
ok_h = all(np.array_equal(idx_real.hamming(Q["ubinary"][i, :32]), cpu_h[i]) for i in range(len(cpu_h)))
check(ok_h, "GPU Hamming == the CPU scores used in the evaluation, 40 questions")
ok_a = all(np.abs(idx_real.asym_scores(Q["float"][i]) - cpu_a[i]).max() < 1e-4 for i in range(len(cpu_a)))
check(ok_a, "GPU asym == the CPU binary_asym scores, 40 questions")
cpu_a8 = sy.binary_asym8(Q["float"], P["ubinary"], 256)
ok_8 = all(np.array_equal(idx_real.asym_bits_scores(Q["float"][i]), cpu_a8[i].astype(np.int64)) for i in range(40))
check(ok_8, "GPU bit-serial == the CPU binary_asym8 scores, 40 questions")
top = lambda S, i: set(np.argsort(-S[i], kind="stable")[:100])
overlap = np.mean([len(top(cpu_a8, i) & top(cpu_a, i)) / 100 for i in range(40)])
check(overlap > 0.95, f"the 8-bit question keeps {overlap:.1%} of the float question's top-100")
print(f"{passed} checks passed")
