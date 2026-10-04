# 32 Bytes: Embed 5's smallest vectors, without a shortlist

Cohere's [Embed 5](https://cohere.com/blog/embed-5) (released 2026-09-30) can store a page as 32 bytes: the sign
bits of its first 256 dimensions. The launch post's storage chart is scored with RCP-nDCG@10, which ranks a fixed
set of candidate pages per question; the post's first footnote notes that such scores reflect re-ranking quality,
and that first-stage retrieval is evaluated separately with nDCG and recall. A first-stage index has no candidate
set: every page of the collection competes. This project asks what the small sizes do then, on public data:

- every page ranked, versus only each question's 150-page judged pool, at every size from 32 bytes to 8 KB;
- questions in English and five machine-translated languages, over English pages;
- a 32-byte first pass followed by a re-check of its top pages with int8 or float vectors;
- an index built with Embed 5 Pro and queried with Embed 5 Fast, and the reverse;
- how deep the re-check must go as the collection grows;
- how fast a MacBook Air's GPU can search 32-byte codes (Metal kernels, against FAISS on the same machine).

> **No Embed 5 quality results are published here.** They stay private unless Cohere agrees to their publication.
> The repository holds the code and the GPU search speeds, which time the kernels over 32-byte codes, not the model.

What is here now: the code for the whole study; a Metal search that returns the exact top 100 of 10 million
32-byte codes in 4.45 ms on a fanless M2 MacBook Air (1.04 ms per question at 8 questions per pass, 4.3x FAISS on
the same machine at the same batch size); and a client that ran the whole study, probe included, in 358 logged
calls of a trial key's 1,000 a month. The metrics match Cohere's rcp-ndcg query by
query, and the GPU's integer scores and top-k match numpy exactly, ties included.

## The data

Three sets of ViDoRe v3, as packaged in Cohere's RCP-nDCG release
([`fabianschmidt-cohere/rcp-ndcg-vidore-v3`](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3),
pinned to revision `4e5ca46e`):

| set | pages (non-empty) | questions per language | documents |
|---|---:|---:|---|
| `finance_en` | 2,923 | 309 | US 10-K filings |
| `computer_science` | 1,358 | 215 | CS textbooks |
| `hr` | 1,098 | 318 | EU HR reports |

Each question exists in English, French, German, Italian, Portuguese and Spanish (5,052 questions in all), with
ViDoRe's human relevance grades and, per language, a judged pool of 150 pages, each with an RCP gain: a relevance
score from calibrated LLM judgements, which [RCP-nDCG](https://github.com/cohere-ai/rcp-ndcg) uses in place of
sparse human labels. Pages outside the pool have no RCP gain.

**Pages are embedded as their parsed text, not as images.** The page images (99.6% of the corpus files, 2.2 GB)
are never fetched.

## What is measured

- **Storage.** Float at 256, 512, 768, 1024, 1536 and 2048 dimensions; Embed 5's own int8 at 2048 and, for the Pro
  index only, 1024; binary sign bits at every dimension (32 bytes = 256 bits). Every float size and binary code is
  derived from the 2048-dimensional vectors; [`scripts/probe.py`](scripts/probe.py) checks that assumption with 9
  calls before the main run.
- **Scoring a binary index** three ways: Hamming distance between the question's and the page's bits; the float
  question against the page's bits read as ±1; and the same with the question quantized to 8 bits, which is what
  the GPU kernel computes.
- **Two protocols.** *Shortlist*: rcp-ndcg's `vidore` protocol, only the question's 150-page pool is ranked.
  *Every page*: rcp-ndcg's `plain` protocol, the whole set is ranked and a page outside the pool has gain 0.
- **Metrics.** The main metric is nDCG@10 on ViDoRe's human grades: every page graded relevant is in the pool, so
  it is exact over every page too (relative to ViDoRe's labels, which are not exhaustive). Also RCP-nDCG@10,
  recall@10 and @100, and the number of never-judged pages in the top 10.
  [`src/metrics.py`](src/metrics.py) matches rcp-ndcg's `score_query` query by query and reproduces the scores
  rcp-ndcg's CLI reports for Cohere's stored runs ([`tests/test_metrics.py`](tests/test_metrics.py)).
- **Two-stage search.** A first pass over every page (32-byte Hamming, 32 bytes with the float or the 8-bit
  question, 128-byte Hamming) keeps its top n pages (n = 10, 20, 50, 100 and 200, and 1, 2, 5 and 10% of the
  set); float 2048, int8 2048 or int8 1024 re-scores them. Ties at the cut are broken in random order, averaged
  over orders.
- **Pro and Fast.** All four index model × question model pairs at every storage setting, except int8 at 1024 for
  a Fast index, which was not embedded (`scripts/embed_all.py --with-fast-int8` adds it).
- **Collection size.** For each English question (Pro index, Pro questions), collections from 100 pages up to all
  5,379 that always hold the question's relevant pages: first more pages of its own set, then the other sets'
  pages as distractors. Five random draws per size, except the whole own set and all 5,379 pages, which are run
  once. Reported: the depth at which the first pass reaches 95% recall, with bootstrap intervals over questions.
- **Uncertainty.** Paired bootstrap confidence intervals (10,000 resamples) in which a question and its five
  translations form one cluster, since a question's six versions are not independent, and Holm's correction across
  the six-language comparisons.
- **Scope.** Embed 5 is compared only with itself (sizes, precisions, Pro and Fast); no other embedding model or
  provider is run. FAISS appears only as a CPU baseline for search speed.

## GPU search over 32-byte codes

[`kernels/search.metal`](kernels/search.metal), compiled at run time with `torch.mps.compile_shader` (the same way
[metal-inference-engine](https://github.com/MeKaustubh07/metal-inference-engine) compiles its kernels), driven from
[`src/gpu.py`](src/gpu.py):

- **Hamming scan:** one thread per page, `popcount(code XOR question)` over two `uint4` loads; 10M codes are
  320 MB, read once per query.
- **Exact top-k:** the scan also builds a 257-bin histogram of the distances; one thread finds the k-th distance;
  pages at or under it are compacted with an atomic counter, and the host orders the few candidates by (distance,
  page). If more than 4,096 pages tie at the cut, the distances are read back and the CPU finishes the selection,
  still exact.
- **Batched:** 8 questions per pass over the codes, so each code is read once for 8 questions.
- **8-bit question:** the float question's first 256 values scaled so the largest magnitude is 127, rounded to
  integers in [-127, 127], offset by 128 and split into 8 bit-planes. A page's score is the 8 plane AND+popcounts
  weighted 1, 2, 4, ..., 128, minus 128 times the page's own popcount: 9 popcounts and no lookups. An exact
  two-level radix select (high byte during the scan, low byte inside the threshold bucket) finds the top k.

Median milliseconds per question, exact top-100, MacBook Air M2 (8-core GPU, 8 GB, fanless), macOS 26.4, torch
2.14.1, faiss-cpu 1.15.1:

| setup | questions per call | 10M codes | 1M codes |
|---|---:|---:|---:|
| GPU, Hamming | 1 | **4.45** | 0.91 |
| GPU, 8-bit question | 1 | 4.80 | 0.97 |
| GPU, Hamming scan alone (no top-k) | 1 | 4.08 (78.4 GB/s) | 0.72 |
| GPU, Hamming | 8 / 32 | **1.04** / 1.07 | 0.17 / 0.18 |
| FAISS `IndexBinaryShards`, 4 shards | 1 | 5.38 | 0.85 |
| FAISS `IndexBinaryFlat`, 4 threads | 8 / 32 | 4.43 / 4.35 | 0.62 / 0.40 |
| FAISS `IndexBinaryFlat`, 8 threads | 8 / 32 | 4.51 / 3.62 | 0.44 / 0.34 |

- FAISS gives one question one core, so for one question per call the index is split into 4 shards (the M2's
  performance cores) searched on one thread each: the fastest single-question CPU setup found
  ([`scripts/bench_faiss.py`](scripts/bench_faiss.py)).
- One question per call: the GPU is 1.2x faster at 10M codes; at 1M, FAISS is 1.07x faster (0.85 vs 0.91 ms).
- At equal batch sizes and 10M codes the GPU is 4.3x faster at 8 questions per call and 3.4x at 32 (1.04 vs
  4.43 ms; 1.07 vs 3.62 ms, each against the faster FAISS row); at 1M, 2.6x and 1.9x.
- At 10M codes the scan alone runs at 87% of the 89.9 GB/s that a read-only probe kernel reaches on this machine,
  and a whole query, top-k included, at about 80%.
- Sustained (400 calls back to back, the last 100): median 4.39 ms, p90 5.51 ms at 10M.
- The 8-bit question costs 1.05x a Hamming query at 10M (1.04x at 1M) when the two are timed interleaved; the
  table's two medians, which come from different runs, give 1.08x.
- **Both indexes are speed only:** the 5,379 real codes padded to 1M and to 10M with synthetic codes (bits drawn
  at the real codes' per-bit frequencies), so these runs say nothing about search quality. 100M was not run.
- FAISS runs in its own process (FAISS and PyTorch each bundle OpenMP). GPU-vs-FAISS rows are single medians,
  without run-to-run spread, and whether FAISS is bandwidth-bound here was not checked.

## Running on a trial key

A trial key allows 1,000 calls a month, so [`src/client.py`](src/client.py) treats calls as a budget:

- every HTTP attempt is written to a SQLite ledger *before* it is sent, and a hard cap (450 a month) stops a run;
  the SDK's own retries are off, because each silent retry may count as a call;
- every vector is cached per (model, input type, dimension, embedding type, text), so a rerun never pays twice;
- batches are sized from a per-character token estimate fitted on real calls and corrected from the token counts
  each response reports, and calls are paced to the key's rate limit;
- [`scripts/embed_all.py`](scripts/embed_all.py) splits the run into nine chunks (one set × one model × one job)
  that can be stopped at any time and resumed without re-sending a finished call (a call cut off mid-flight is
  sent again).

By the ledger's count, the whole study, probe included, made 358 HTTP requests, every attempt counted.

## Reproducing

An Apple silicon Mac on macOS 14 or newer (the pinned torch and faiss-cpu wheels need it), Python 3.12 or newer
(rcp-ndcg needs it), and git. rcp-ndcg is not on PyPI yet, so `requirements.txt` pins it to a commit of
[cohere-ai/rcp-ndcg](https://github.com/cohere-ai/rcp-ndcg).

```bash
# with uv (https://docs.astral.sh/uv/), which also fetches Python 3.12 if needed
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
# or without uv
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Put your key in `.env` as `CO_API_KEY=...` (gitignored; the scripts read it with python-dotenv and never print it).
The steps that call the API (`probe.py --go`, `embed_all.py --go`) run on your own key, under Cohere's terms;
everything from `evaluate.py` on makes no API calls.

```bash
.venv/bin/python scripts/fetch_data.py         # the three sets, ~28 MB, ~4 min (no images; no Hugging Face token)
.venv/bin/python scripts/probe.py --go         # 9 calls that check the main run's assumptions
.venv/bin/python scripts/embed_all.py          # status: calls and minutes left per chunk
.venv/bin/python scripts/embed_all.py --next --go   # run the next chunk; repeat until done
.venv/bin/python scripts/evaluate.py           # score every setting, ~30 min on the M2
.venv/bin/python scripts/stats.py              # clustered bootstrap intervals
.venv/bin/python scripts/sweep.py              # collection size vs re-check depth
.venv/bin/python scripts/sheet_numbers.py      # summary contrasts and Holm-corrected p-values
.venv/bin/python scripts/bench_gpu.py          # GPU vs FAISS speeds at 1M and 10M codes
```

The GPU speeds need only `fetch_data.py` and the three Pro `vectors` chunks
(`scripts/embed_all.py hr pro vectors --go`, then `computer_science` and `finance_en`): `bench_gpu.py` reads the
Pro page codes and five English Pro questions per set. `stats.py` needs `evaluate.py`; `sheet_numbers.py` needs
`evaluate.py`, `sweep.py` and every default chunk, Fast included.

Results go to `results/`; the vector cache, call ledger and per-question scores to `cache/`; the probe's report
and raw responses to `probe_run/`. All three are gitignored.

## Tests

Each test is a plain script (no pytest): `.venv/bin/python tests/test_client.py`, or all of them with
`for t in tests/test_*.py; do .venv/bin/python "$t" || break; done`.

| test | checks | needs |
|---|---:|---|
| `tests/test_client.py` | 22 | nothing: a fake Cohere server, a fake key |
| `tests/test_probe.py` | 12 | `data/`; a fake Embed 5 |
| `tests/test_embed_all.py` | 10 | `data/`; a fake Embed 5 on a fake clock |
| `tests/test_metrics.py` | 13 | `data/`, and Cohere's stored runs from Hugging Face (downloaded once) |
| `tests/test_systems.py` | 11 | checks 1-10: nothing; check 11: `data/` and the cached Pro question vectors |
| `tests/test_gpu.py` | 22 | an Apple silicon Mac; checks 19-22 also `data/` and the cached Pro vectors |

Without your own Embed 5 run, `test_systems.py` and `test_gpu.py` stop at their first check that needs the vectors.
With the full cache, all 90 checks pass on the M2. What they pin down: `src/metrics.py` matches rcp-ndcg's
`score_query` on random cases, including the heavy ties of Hamming scores, and reproduces rcp-ndcg's CLI on
Cohere's stored runs; the GPU's Hamming and 8-bit scores and top-k match numpy exactly (ties by page index, the CPU
fallback on a huge tie, 8 questions per pass equal to one at a time) and equal the CPU scores the evaluation uses;
the quantizer equals exact rational rounding on all 5,052 questions; the client sends only uncached texts, stops at
the hard cap, and keeps the key out of the ledger and the saved responses.

## Layout

```
src/client.py         budgeted, cached Embed 5 client (ledger, cache, token-aware batching)
src/data.py           the three sets as local parquet files
src/metrics.py        nDCG with rcp-ndcg's tie handling, both protocols, recall
src/systems.py        score matrices for every storage setting, the two-stage combiner
src/gpu.py            the Metal index: Hamming, batched and 8-bit-question top-k
kernels/search.metal  the kernels
scripts/              fetch, probe, embed, evaluate, stats, sweep, sheet_numbers, bench_gpu (in that order;
                      bench_gpu runs bench_faiss.py in its own process)
tests/                the checks above
```

## Prior work

Binary first pass plus re-scoring is an established recipe, not something this project introduces: Cohere's
[int8 and binary embeddings](https://cohere.com/blog/int8-binary-embeddings) post and
[BinaryVectorDB](https://github.com/cohere-ai/BinaryVectorDB), Hugging Face and mixedbread's
[embedding quantization](https://huggingface.co/blog/embedding-quantization) post, Qdrant's
[quantization with oversampling](https://qdrant.tech/documentation/manage-data/quantization/) and Elastic's
[BBQ](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch). Built on ViDoRe v3,
[rcp-ndcg](https://github.com/cohere-ai/rcp-ndcg) and [FAISS](https://github.com/facebookresearch/faiss). While
building this I reported an [install issue](https://github.com/cohere-ai/rcp-ndcg/issues/4) to rcp-ndcg.

## License

[Apache-2.0](LICENSE).
