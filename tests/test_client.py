"""Offline checks of src/client.py against a fake Cohere server (httpx.MockTransport): no network, a fake key,
.env is never read. Run: .venv/bin/python tests/test_client.py"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.pop("CO_API_KEY", None)
import cohere  # noqa: E402

import client as ec  # noqa: E402

SLEPT, HTTP, SCRIPT = [], [], []          # sleeps taken, request bodies seen, scripted (status, headers) replies
NOW = [1000.0]                            # the fake clock: only sleeping moves it
WINDOW = []                               # the fake server's (time, tokens) of accepted calls
LIMIT = {"tpm": None}                     # tokens per minute the fake server enforces (None: no limit)
FAKE_KEY = "offline-fake-key-0123456789"
passed = 0


def fake_sleep(s):
    SLEPT.append(s)
    NOW[0] += s


ec._clock, ec._sleep = (lambda: NOW[0]), fake_sleep


def check(cond, what):
    global passed
    assert cond, what
    passed += 1
    print(f"  ok  {what}")


def fake_vec(text, dim):
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).standard_normal(2048).astype(np.float32)[:dim]
    return (v / np.linalg.norm(v)).astype(np.float16).astype(np.float32)


def handler(req):
    body = json.loads(req.content)
    HTTP.append(body)
    if SCRIPT:
        status, headers, *msg = SCRIPT.pop(0)
        return httpx.Response(status, headers=headers, json={"message": msg[0] if msg else "scripted"})
    tokens = sum(len(t) // 4 + 9 for t in body["texts"])      # the fake tokenizer: 4 chars a token
    if LIMIT["tpm"] is not None:                              # Cohere's trial rule: tokens in any 60 s window
        WINDOW[:] = [(t, n) for t, n in WINDOW if NOW[0] - t < 60.0]
        if sum(n for _, n in WINDOW) + tokens > LIMIT["tpm"]:
            return httpx.Response(429, json={"message": "trial token rate limit exceeded, limit is "
                                                        f"{LIMIT['tpm']} tokens per minute"})
        WINDOW.append((NOW[0], tokens))
    f = np.stack([fake_vec(t, body["output_dimension"]) for t in body["texts"]])
    ub = np.packbits(f > 0, axis=-1)
    emb = {"float": f.tolist(), "int8": np.clip(np.round(f * 300), -128, 127).astype(int).tolist(),
           "ubinary": ub.tolist(), "binary": (ub.astype(int) - 128).tolist()}
    return httpx.Response(200, headers={"x-ratelimit-remaining": "39", "x-unrelated": "no"}, json={
        "id": "r", "texts": body["texts"], "embeddings": {k: emb[k] for k in body["embedding_types"]},
        "meta": {"api_version": {"version": "2"}, "billed_units": {"input_tokens": tokens - 2 * len(body["texts"])},
                 "tokens": {"input_tokens": tokens}, "cached_tokens": 0}})


tmp = Path(tempfile.mkdtemp())
co = cohere.ClientV2(api_key=FAKE_KEY, base_url="http://mock.invalid", max_retries=0,
                     httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))
emb = ec.Embedder(co, ec.Ledger(tmp / "ledger.sqlite", monthly_cap=12), ec.Cache(tmp / "vectors.sqlite"),
                  min_interval_s=0, raw_dir=tmp / "raw")
PRO = dict(model="embed-v5.0-pro", input_type="search_document")
texts = [f"page {i}" for i in range(200)]

print("batching and cache")
dry = emb.embed(texts, **PRO, dry_run=True)
check(dry["calls"] == 3 and dry["texts"] == 200, "dry run: 200 short texts -> 3 calls")
out = emb.embed(texts, **PRO, label="t1")
check([len(b["texts"]) for b in HTTP] == [96, 96, 8], "sent as 96 + 96 + 8")
check(all(b["output_dimension"] == 2048 and b["embedding_types"] == ["float", "int8", "ubinary"] for b in HTTP),
      "every request names its dimension and types explicitly")
check(all(b["truncate"] == "NONE" for b in HTTP), 'every request sends truncate="NONE" (no silent shortening)')
check(out["float"].shape == (200, 2048) and out["ubinary"].shape == (200, 256) and out["int8"].dtype == np.int8,
      "shapes and dtypes")
check(np.array_equal(out["float"][5], fake_vec("page 5", 2048)), "vectors come back in input order")
n = len(HTTP)
again = emb.embed(texts[::-1], **PRO)
check(len(HTTP) == n and np.array_equal(again["float"][::-1], out["float"]), "a rerun in any order makes 0 calls")
emb.embed(texts[:10] + ["new text"], **PRO)
check(len(HTTP) == n + 1 and HTTP[-1]["texts"] == ["new text"], "only the uncached text is sent")

print("errors and retries")
SCRIPT[:] = [(429, {"retry-after": "3"})]
emb.embed(["q1"], model="embed-v5.0-fast", input_type="search_query", label="t429")
rows = emb.ledger.db.execute("SELECT attempt, status FROM calls WHERE label='t429'").fetchall()
check(rows == [(1, "429"), (2, "ok")] and 3.0 in SLEPT, "429: waits Retry-After, retries, both attempts logged")
SCRIPT[:] = [(502, {})]
emb.embed(["q2"], model="embed-v5.0-fast", input_type="search_query", label="t502")
rows = emb.ledger.db.execute("SELECT attempt, status, http_status FROM calls WHERE label='t502'").fetchall()
check(rows == [(1, "5xx", 502), (2, "ok", 200)], "a bare 502 is retried, not fatal")
SCRIPT[:] = [(400, {})]
try:
    emb.embed(["bad"], model="embed-v5.0-fast", input_type="search_query", label="t400")
    raised = False
except cohere.errors.BadRequestError:
    raised = True
rows = emb.ledger.db.execute("SELECT status FROM calls WHERE label='t400'").fetchall()
check(raised and rows == [("error",)], "400 raises without a retry and is logged")

SCRIPT[:] = [(429, {}, "monthly call limit exceeded for trial key")]
try:
    emb.embed(["q3"], model="embed-v5.0-fast", input_type="search_query", label="tquota")
    stopped = False
except ec.BudgetExhausted:
    stopped = True
rows = emb.ledger.db.execute("SELECT status FROM calls WHERE label='tquota'").fetchall()
check(stopped and rows == [("429",)], "a quota 429 stops the run at once, no retry")

print("budget and secrets")
try:
    for i in range(20):
        emb.embed([f"x{i}"], model="embed-v5.0-fast", input_type="search_query")
    stopped = False
except ec.BudgetExhausted:
    stopped = True
check(stopped and emb.ledger.used_this_month() == 12, "the hard cap stops the run at exactly 12 attempts")
dump = "\n".join(emb.ledger.db.iterdump()) + "".join(p.read_text() for p in (tmp / "raw").iterdir())
check(FAKE_KEY not in dump, "the key appears in neither the ledger nor the saved responses")
hdr = json.loads(emb.ledger.db.execute("SELECT resp_headers FROM calls WHERE status='ok' LIMIT 1").fetchone()[0])
check(hdr == {"x-ratelimit-remaining": "39"}, "rate-limit headers are logged, others dropped")
tok = emb.ledger.db.execute("SELECT billed_input_tokens FROM calls WHERE label='t1' LIMIT 1").fetchone()[0]
check(tok == sum(len(t) // 4 + 7 for t in HTTP[0]["texts"]), "billed tokens are logged per call")

print("the trial's token limit (100k tokens a minute)")
LIMIT["tpm"] = 100_000
pages = [f"page {i} " + "x" * 8000 for i in range(120)]          # ~2,000 tokens each: 240k in all
emb2 = ec.Embedder(co, ec.Ledger(tmp / "ledger2.sqlite", monthly_cap=60), ec.Cache(tmp / "vectors2.sqlite"),
                   min_interval_s=0, call_token_cap=60_000)
plan = emb2.embed(pages, **PRO, dry_run=True)
check(plan["calls"] == 6 and plan["est_tokens"] <= 6 * 60_000,
      "long texts are batched by estimated tokens (60k per call), not 96 at a time")
n, t0, nslept = len(HTTP), NOW[0], len(SLEPT)
got = emb2.embed(pages, **PRO, label="long")
rows = emb2.ledger.db.execute("SELECT status FROM calls WHERE label='long'").fetchall()
check(got["float"].shape == (120, 2048) and all(r == ("ok",) for r in rows),
      "240k tokens embedded with no 429: the client paces itself under the limit")
check(NOW[0] - t0 >= 60.0 and any(s > 30 for s in SLEPT[nslept:]), "it waits for the minute window to clear")
check(abs(emb2.chars_per_token - 8008 / 2011) < 0.05 and len(HTTP) - n < 6,
      "it learns the real chars per token from the response, then sends fewer, fuller calls")
emb3 = ec.Embedder(co, ec.Ledger(tmp / "ledger3.sqlite", monthly_cap=60), ec.Cache(tmp / "vectors3.sqlite"),
                   min_interval_s=0, call_token_cap=500_000)                    # a cap far above the limit
got = emb3.embed(pages[:96], **PRO, label="toobig")
rows = [r[0] for r in emb3.ledger.db.execute("SELECT status FROM calls WHERE label='toobig' ORDER BY id")]
check(got["float"].shape == (96, 2048) and rows[0] == "429" and rows[-1] == "ok" and emb3.call_token_cap < 500_000,
      "a call refused for tokens is split more finely and completes")
check(rows.count("429") <= 3, f"and it converges quickly ({rows.count('429')} refusals)")
print(f"{passed} checks passed")
