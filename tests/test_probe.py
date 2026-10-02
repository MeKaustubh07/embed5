"""The probe end to end against a fake Embed 5 that has the properties we hope the real one has (Matryoshka:
a d-dim vector is the renormalized first d numbers of the 2048 one; ubinary = big-endian sign bits; binary =
ubinary - 128; int8 = a fixed scale; Pro and Fast close; deterministic). The analysis must report each property.
No network, no key. Needs data/ (scripts/fetch_data.py). Run: .venv/bin/python tests/test_probe.py"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ.pop("CO_API_KEY", None)
import cohere  # noqa: E402

import client as ec  # noqa: E402
import probe  # noqa: E402

NOW = [0.0]
ec._clock = lambda: NOW[0]
ec._sleep = lambda s: NOW.__setitem__(0, NOW[0] + s)
CALLS = []
passed = 0


def check(cond, what):
    global passed
    assert cond, what
    passed += 1
    print(f"  ok  {what}")


def fake_2048(text, model):
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).standard_normal(2048)
    if model.endswith("fast"):                       # a nearby point in the same space
        v = v + 0.3 * np.random.default_rng(seed + 1).standard_normal(2048)
    return v


def handler(req):
    body = json.loads(req.content)
    CALLS.append(body)
    d = body["output_dimension"]
    f = np.stack([fake_2048(t, body["model"])[:d] for t in body["texts"]])
    f = (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float16).astype(np.float32)
    ub = np.packbits(f > 0, axis=-1)
    q = np.clip(np.round(f * 400), -128, 127).astype(int)
    emb = {"float": f.tolist(), "int8": q.tolist(), "uint8": (q + 128).tolist(), "ubinary": ub.tolist(),
           "binary": (ub.astype(int) - 128).tolist()}
    return httpx.Response(200, json={"id": "r", "texts": body["texts"],
                                     "embeddings": {k: emb[k] for k in body["embedding_types"]},
                                     "meta": {"billed_units": {"input_tokens": sum(len(t) // 4 + 9 for t in body["texts"])}}})


tmp = Path(tempfile.mkdtemp())
co = cohere.ClientV2(api_key="offline-fake", base_url="http://mock.invalid", max_retries=0,
                     httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))
emb = ec.Embedder(co, ec.Ledger(tmp / "ledger.sqlite", run_cap=11), ec.Cache(tmp / "vectors.sqlite"),
                  min_interval_s=0)

print("probe against a fake Embed 5")
pages, _, questions, _, _ = probe.pick_texts()
check(emb.embed(pages, model=ec.MODELS[0], input_type="search_document", dry_run=True)["calls"] == 1
      and emb.embed(questions, model=ec.MODELS[0], input_type="search_query", dry_run=True)["calls"] == 1,
      "the 48 probe pages fit in one call, and so do the 96 questions")
A = probe.main(True, emb, tmp)
check(len(CALLS) == 9 and sorted(len(c["texts"]) for c in CALLS) == [48] * 6 + [96] * 3,
      "exactly 9 calls: 6 of 48 pages, 3 of 96 questions")
check(emb.ledger.used_this_month() == 9, "9 ledger rows")
check(A["types returned"] == sorted(probe.TYPES), "all five types read back")
check(all(A[f"{p} matryoshka"]["min_cos"] > 0.9999 for p in ("P2", "P3", "P8", "P9")),
      "Matryoshka detected for Pro and Fast, pages and questions")
check(all(A[f"{p} ubinary == first {d // 8} bytes of 2048"] == 1.0
          for p, d in (("P2", 256), ("P3", 768), ("P8", 256), ("P9", 256))),
      "binary of a short vector == the first bytes of the 2048 binary")
check(A["P1 bits"]["ubinary==signbits(big-endian)"] == 1.0 and A["P1 bits"]["binary==ubinary-128"] == 1.0,
      "bit order and the signed offset identified")
check(A["P1 int8 rule"]["uint8==int8+128"] == 1.0
      and max(v[0] for k, v in A["P1 int8 rule"].items() if k.startswith("global")) > 0.99,
      "the int8 rule (a fixed scale) identified")
check(A["P7 == P1 bit for bit (reversed batch)"], "determinism detected")
check(0.9 < A["cos pro vs fast, same page (p5, p50, p95)"][1] < 0.99, "Pro vs Fast cosine reported")
r = A["mean rank of relevant pages among the 48 (index/question)"]
check(set(r) == {"pro/pro", "pro/fast", "fast/pro", "fast/fast"} and all(1 <= v <= 48 for v in r.values()),
      "retrieval ranks for the 4 index/question pairings")
again = probe.main(True, emb, tmp)
check(len(CALLS) == 9 and again["P7 == P1 bit for bit (reversed batch)"], "a rerun is served from the cache: 0 calls")
print(f"{passed} checks passed")
