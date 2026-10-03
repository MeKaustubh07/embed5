"""Search speed over 32-byte codes on this MacBook Air (M2, 8-core GPU, 8 GB; fanless): src/gpu.py's Metal kernels
against FAISS, the strongest CPU baselines found (scripts/bench_faiss.py), and numpy as unoptimized context.

The index: every embedded page's real Embed 5 code (Pro), padded with SYNTHETIC codes to N pages (bits drawn with
the real codes' per-bit frequencies). The padding stands in for speed only: such codes crowd a real question's
top-100, so nothing here says anything about search quality. Every method uses the same 5 real questions, and
every number is a median of synchronized runs after warm-up, labelled with the questions per call ("batch").
  GPU, batch 1      exact top-100 by Hamming (counting select); by the 8-bit float question (bit-serial, radix
                    select), timed interleaved with Hamming (ABBA blocks) so drift does not bias their ratio
  GPU, batch 8/32   Hamming, 8 questions per pass over the codes
  sustained         400 Hamming calls back to back: the first 30 vs the last 100 (a fanless machine)
  bandwidth         the Hamming scan alone, and a read-only probe kernel: the ceiling the scan runs against
Writes results/bench_gpu.csv (speeds only: no Embed 5 quality numbers). Usage: python scripts/bench_gpu.py [N ...]
"""
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import gpu  # noqa: E402
import systems as sy  # noqa: E402

BANDWIDTH = 100e9                                   # M2 unified memory, Apple's spec, bytes/s
K, RUNS, WARMUP, QUESTIONS = 100, 30, 5, 5


def real_codes():
    pages, qb, qf = [], [], []
    for s in data.DOMAINS:
        d = data.load(s)
        pages.append(sy.cached(d.pages().text.tolist(), "pro", "search_document")["ubinary"][:, :32])
        q = sy.cached(d.queries[d.queries.language == "english"].text.tolist()[:QUESTIONS], "pro", "search_query")
        qb.append(q["ubinary"][:, :32])
        qf.append(q["float"])
    return np.concatenate(pages), np.concatenate(qb), np.concatenate(qf)


def padded(real: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    p = np.unpackbits(real, axis=1).mean(0).astype(np.float32)
    rng = np.random.default_rng(seed)
    out = np.empty((n, 32), np.uint8)
    out[:len(real)] = real
    for s in range(len(real), n, 1 << 16):         # 65k rows x 256 float32 = 67 MB at a time (8 GB machine)
        e = min(n, s + (1 << 16))
        out[s:e] = np.packbits(rng.random((e - s, 256), dtype=np.float32) < p, axis=1)
    return out


def calls(fn, runs: int = RUNS, warmup: int = WARMUP) -> list[float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        gpu.sync()
        t0 = time.perf_counter()
        fn()
        gpu.sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def row(method, n, batch, ms, **extra):
    return {"method": method, "pages": n, "batch": batch, "ms_per_question": ms} | extra


def run(n: int, real, qb, qf) -> list[dict]:
    codes = padded(real, n)
    ix = gpu.Index(codes)
    rows = []
    # batch 1, the latency case: median over the 5 questions of each question's median
    ham = statistics.median(statistics.median(calls(lambda q=q: ix.topk(q, K))) for q in qb[:QUESTIONS])
    rows.append(row("GPU Hamming top-100", n, 1, ham))
    # Hamming vs the 8-bit question, interleaved A B B A so slow drift hits both alike
    a, b = [], []
    for rep in range(8):
        order = ("A", "B", "B", "A") if rep % 2 == 0 else ("B", "A", "A", "B")
        for which in order:
            q = rep % QUESTIONS
            if which == "A":
                a += calls(lambda: ix.topk(qb[q], K), runs=10, warmup=2)
            else:
                b += calls(lambda: ix.asym_bits_topk(qf[q], K), runs=10, warmup=2)
    rows.append(row("GPU 8-bit question top-100 (bit-serial)", n, 1, statistics.median(b),
                    ratio_to_hamming=statistics.median(b) / statistics.median(a),
                    ratio_p10=float(np.percentile(np.array(b) / np.array(a), 10)),
                    ratio_p90=float(np.percentile(np.array(b) / np.array(a), 90))))
    batch = np.concatenate([qb] * 8)[:32]
    for nb in (8, 32):
        t = statistics.median(calls(lambda: ix.topk_batch(batch[:nb], K), runs=10))
        rows.append(row("GPU Hamming top-100, 8 per pass", n, nb, t / nb))
    sus = calls(lambda: ix.topk(qb[0], K), runs=400, warmup=5)
    rows.append(row("GPU Hamming top-100, sustained", n, 1, statistics.median(sus[-100:]),
                    first_30_ms=statistics.median(sus[:30]), last_100_p90_ms=float(np.percentile(sus[-100:], 90))))
    scan = statistics.median(calls(lambda: gpu.LIB.hamming(ix.dist, ix.codes, ix._q(qb[0]), n, threads=n,
                                                            group_size=gpu.TG)))
    probe_out = gpu.torch.empty(n // 32 + 1, dtype=gpu.torch.int32, device=gpu.DEV)
    probe = statistics.median(calls(lambda: gpu.LIB.read_probe(probe_out, ix.codes, n, threads=n,
                                                                group_size=gpu.TG)))
    gb = codes.nbytes / 1e9
    rows.append(row("GPU Hamming scan only (no top-k)", n, 1, scan, index_GBps=gb / (scan / 1e3),
                    pct_of_spec=100 * gb / (scan / 1e3) * 1e9 / BANDWIDTH,
                    probe_GBps=gb / (probe / 1e3), pct_of_probe=100 * probe / scan))
    assert ix.fallbacks == 0, f"{ix.fallbacks} CPU fallbacks during timing"
    del ix
    c64, q64 = codes.view(np.uint64), qb[0].view(np.uint64)
    t = statistics.median(calls(lambda: np.argpartition(np.bitwise_count(c64 ^ q64).sum(1, dtype=np.int32), K),
                                runs=10, warmup=2))
    rows.append(row("numpy, 1 thread (unoptimized, context only)", n, 1, t))
    tmp = ROOT / "cache" / "bench"                  # FAISS in its own process (OpenMP clash with PyTorch)
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "codes.npy", codes)
    np.save(tmp / "questions.npy", batch)
    del codes
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "bench_faiss.py"), str(tmp / "codes.npy"),
                          str(tmp / "questions.npy"), str(K), str(RUNS), str(WARMUP)],
                         capture_output=True, text=True, check=True)
    (tmp / "codes.npy").unlink()
    rows += [r | {"pages": n} for r in json.loads(out.stdout)]
    return rows


def main(sizes: list[int]) -> None:
    real, qb, qf = real_codes()
    rows = []
    for n in sizes:
        r = run(n, real, qb, qf)
        rows += r
        for x in r:
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in x.items()}, flush=True)
    out = ROOT / "results" / "bench_gpu.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1_000_000, 10_000_000])
