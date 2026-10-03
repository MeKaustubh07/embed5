"""Search speed over 32-byte codes on this MacBook Air (M2, 8 GB): the Metal kernels of src/gpu.py against the CPU.

The index: every embedded page's real Embed 5 code (Pro), padded with SYNTHETIC codes to N pages. The padding
draws each bit with the real codes' per-bit frequency (independent bits), so it is labelled synthetic everywhere.
Check: the real questions' top-100 over the padded index must equal their top-100 over the real pages alone.
Timings per question (median of 30, synchronized; 5 warm-up runs): exact top-100 by Hamming (counting select), by
the 8-bit float question against the bits (bit-serial, radix select) and by the first lookup-table version, against
numpy (1 and 4 threads) and FAISS
IndexBinaryFlat (all cores, in its own process). Writes results/bench_gpu.csv (no Embed 5 quality numbers: publishable).
Usage: python scripts/bench_gpu.py [N ...]   (default 1000000 10000000)
"""
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import gpu  # noqa: E402
import systems as sy  # noqa: E402

BANDWIDTH = 100e9                                   # M2 unified memory, bytes/s (Apple's spec)
K, RUNS, WARMUP, QUESTIONS = 100, 30, 5, 20


def real_codes():
    pages, questions = [], []
    for s in data.DOMAINS:
        try:
            d = data.load(s)
            pages.append(sy.cached(d.pages().text.tolist(), "pro", "search_document")["ubinary"][:, :32])
            q = d.queries[d.queries.language == "english"].text.tolist()
            questions.append(sy.cached(q, "pro", "search_query"))
        except KeyError:
            print(f"{s}: not fully embedded, left out")
    return (np.concatenate(pages), np.concatenate([q["ubinary"][:, :32] for q in questions]),
            np.concatenate([q["float"] for q in questions]))


def padded(real: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """real codes first, then synthetic ones with the real per-bit frequencies, built in chunks."""
    p = np.unpackbits(real, axis=1).mean(0).astype(np.float32)
    rng = np.random.default_rng(seed)
    out = np.empty((n, 32), np.uint8)
    out[:len(real)] = real
    for s in range(len(real), n, 1 << 16):         # 65k rows x 256 float32 = 67 MB at a time (8 GB machine)
        e = min(n, s + (1 << 16))
        out[s:e] = np.packbits(rng.random((e - s, 256), dtype=np.float32) < p, axis=1)
    return out


def timed(fn, sync=None) -> float:
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(RUNS):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def numpy_topk(codes64: np.ndarray, q64: np.ndarray, threads: int) -> np.ndarray:
    def part(lo, hi):
        return np.bitwise_count(codes64[lo:hi] ^ q64).sum(1, dtype=np.int32)
    step = -(-len(codes64) // threads)
    with ThreadPoolExecutor(threads) as ex:
        d = np.concatenate(list(ex.map(lambda i: part(i, i + step), range(0, len(codes64), step))))
    cut = np.argpartition(d, K)[:K + 1]
    return cut[np.lexsort((cut, d[cut]))][:K]


def main(sizes: list[int]) -> None:
    real, qb, qf = real_codes()
    qb, qf = qb[:QUESTIONS], qf[:QUESTIONS]
    base = gpu.Index(real)
    want = [base.topk(q, K)[0] for q in qb]                       # top-100 over the real pages alone
    rows = []
    for n in sizes:
        codes = padded(real, n)
        index = gpu.Index(codes)
        same = np.mean([np.array_equal(index.topk(q, K)[0], w) for q, w in zip(qb, want)])
        intruders = np.mean([(index.topk(q, K)[0] >= len(real)).sum() for q in qb])
        mb = codes.nbytes / 1e6
        res = {"pages": n, "real_pages": len(real), "index_MB": mb, "real_top100_unchanged": same,
               "synthetic_in_top100": intruders}
        res["gpu_hamming_top100_ms"] = statistics.median(
            timed(lambda q=q: index.topk(q, K), gpu.sync) for q in qb[:5])
        res["gpu_asym_bits_top100_ms"] = statistics.median(
            timed(lambda q=q: index.asym_bits_topk(q, K), gpu.sync) for q in qf[:5])
        res["gpu_asym_lut_v1_top100_ms"] = statistics.median(
            timed(lambda q=q: index.asym_topk(q, K), gpu.sync) for q in qf[:2])
        res["gpu_scan_only_ms"] = timed(lambda: gpu.LIB.hamming(index.dist, index.codes, index._q(qb[0]), n,
                                                                threads=n, group_size=gpu.TG), gpu.sync)
        res["gpu_scan_GBps"] = mb / 1e3 / (res["gpu_scan_only_ms"] / 1e3)
        res["gpu_scan_pct_of_bandwidth"] = 100 * res["gpu_scan_GBps"] * 1e9 / BANDWIDTH
        del index
        c64, q64 = codes.view(np.uint64), qb[0].view(np.uint64)
        res["numpy_1thread_top100_ms"] = timed(lambda: numpy_topk(c64, q64, 1))
        res["numpy_4threads_top100_ms"] = timed(lambda: numpy_topk(c64, q64, 4))
        tmp = ROOT / "cache" / "bench"                          # FAISS runs in its own process (OpenMP clash)
        tmp.mkdir(parents=True, exist_ok=True)
        np.save(tmp / "codes.npy", codes)
        np.save(tmp / "questions.npy", qb)
        del codes
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "bench_faiss.py"), str(tmp / "codes.npy"),
                              str(tmp / "questions.npy"), str(K), str(RUNS), str(WARMUP)],
                             capture_output=True, text=True, check=True)
        res |= json.loads(out.stdout)
        (tmp / "codes.npy").unlink()
        res["speedup_vs_faiss"] = res["faiss_top100_ms"] / res["gpu_hamming_top100_ms"]
        rows.append(res)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}, flush=True)
    out = ROOT / "results" / "bench_gpu.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1_000_000, 10_000_000])
