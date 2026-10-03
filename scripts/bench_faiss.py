"""The FAISS baselines of scripts/bench_gpu.py, in their own process (FAISS and PyTorch each bundle OpenMP; two in
one process is unsafe). Prints a JSON list of rows: method, questions per call, ms per question, threads.

  IndexBinaryFlat, 1 thread             FAISS splits work across questions, so one question runs on one core
  IndexBinaryShards, 4 / 8 shards       the index split into contiguous shards searched on one thread each: the
                                        strongest single-question CPU setup (4 = the M2's performance cores)
  IndexBinaryFlat, 8 / 32 per call      with several questions per call FAISS uses every thread
Usage: python scripts/bench_faiss.py codes.npy questions.npy k runs warmup"""
import json
import statistics
import sys
import time

import faiss
import numpy as np

codes, questions = np.load(sys.argv[1], mmap_mode="r"), np.load(sys.argv[2])
k, runs, warmup = (int(a) for a in sys.argv[3:6])


def ms(index, batch):
    """median over `runs` calls, each over the same questions, per question"""
    for _ in range(warmup):
        index.search(batch, k)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        index.search(batch, k)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3 / len(batch)


rows = []
flat = faiss.IndexBinaryFlat(256)
flat.add(np.ascontiguousarray(codes))
faiss.omp_set_num_threads(1)
one = [ms(flat, np.ascontiguousarray(questions[i:i + 1])) for i in range(min(5, len(questions)))]
rows.append({"method": "FAISS IndexBinaryFlat", "batch": 1, "threads": 1, "ms_per_question": statistics.median(one)})
for nshard in (4, 8):
    faiss.omp_set_num_threads(1)                         # one thread inside each shard: no nested parallelism
    shards = faiss.IndexBinaryShards(256, True, True)    # threaded, successive ids
    parts = []
    for part in np.array_split(np.arange(len(codes)), nshard):
        sub = faiss.IndexBinaryFlat(256)
        sub.add(np.ascontiguousarray(codes[part[0]:part[-1] + 1]))
        parts.append(sub)                                # keep the Python objects alive
        shards.add_shard(sub)
    one = [ms(shards, np.ascontiguousarray(questions[i:i + 1])) for i in range(min(5, len(questions)))]
    rows.append({"method": f"FAISS IndexBinaryShards ({nshard} threaded shards)", "batch": 1, "threads": nshard,
                 "ms_per_question": statistics.median(one)})
    del shards, parts
for threads in (4, 8):
    faiss.omp_set_num_threads(threads)
    for batch in (8, 32):
        if batch <= len(questions):
            rows.append({"method": "FAISS IndexBinaryFlat", "batch": batch, "threads": threads,
                         "ms_per_question": ms(flat, np.ascontiguousarray(questions[:batch]))})
print(json.dumps(rows))
