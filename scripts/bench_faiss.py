"""The FAISS baseline of scripts/bench_gpu.py, in its own process: FAISS and PyTorch each bundle an OpenMP runtime,
and two in one process is unsafe. Reads the codes and the question codes from .npy files, prints JSON.
Usage: python scripts/bench_faiss.py codes.npy questions.npy k runs warmup"""
import json
import statistics
import sys
import time

import faiss
import numpy as np

codes, questions = np.load(sys.argv[1], mmap_mode="r"), np.load(sys.argv[2])
k, runs, warmup = (int(a) for a in sys.argv[3:6])
index = faiss.IndexBinaryFlat(256)
index.add(np.ascontiguousarray(codes))
q = np.ascontiguousarray(questions[:1])
for _ in range(warmup):
    index.search(q, k)
ts = []
for _ in range(runs):
    t0 = time.perf_counter()
    index.search(q, k)
    ts.append(time.perf_counter() - t0)
print(json.dumps({"faiss_top100_ms": statistics.median(ts) * 1e3, "faiss_threads": faiss.omp_get_max_threads()}))
