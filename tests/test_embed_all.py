"""scripts/embed_all.py against a fake Embed 5 on a fake clock: chunk order, status, an interrupted chunk resumed
without re-sending anything. No network, no key; uses data/. Run: .venv/bin/python tests/test_embed_all.py"""
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
import embed_all  # noqa: E402

NOW = [0.0]
ec._clock = lambda: NOW[0]
ec._sleep = lambda s: NOW.__setitem__(0, NOW[0] + s)
CALLS, FAIL_AT = [], [None]           # FAIL_AT: the call number that gets a 400 (an interruption)
passed = 0


def check(cond, what):
    global passed
    assert cond, what
    passed += 1
    print(f"  ok  {what}")


def handler(req):
    body = json.loads(req.content)
    CALLS.append(body)
    if FAIL_AT[0] is not None and len(CALLS) == FAIL_AT[0]:
        return httpx.Response(400, json={"message": "simulated interruption"})
    n, d = len(body["texts"]), body["output_dimension"]
    f = np.stack([np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).standard_normal(d)
                  for t in body["texts"]]).astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    emb = {"float": f.tolist(), "int8": np.clip(np.round(f * 400), -128, 127).astype(int).tolist(),
           "ubinary": np.packbits(f >= 0, axis=-1).tolist()}
    return httpx.Response(200, json={"id": "r", "texts": body["texts"],
                                     "embeddings": {k: emb[k] for k in body["embedding_types"]},
                                     "meta": {"billed_units": {"input_tokens": sum(len(t) // 4 + 9
                                                                                   for t in body["texts"])}}})


tmp = Path(tempfile.mkdtemp())
co = cohere.ClientV2(api_key="offline-fake", base_url="http://mock.invalid", max_retries=0,
                     httpx_client=httpx.Client(transport=httpx.MockTransport(handler)))
ledger, cache = ec.Ledger(tmp / "ledger.sqlite"), ec.Cache(tmp / "vectors.sqlite")
run = lambda *a: embed_all.main(list(a), client=co, ledger=ledger, cache=cache)
first = embed_all.chunks()[0]
want = embed_all.plan(embed_all.embedder("pro", co, ledger, cache), first)

print("chunks")
run()
check(not CALLS, "status alone makes no calls")
run("--next")
check(not CALLS, "--next without --go is a dry run")
FAIL_AT[0] = 10
try:
    run("--next", "--go")
    interrupted = False
except cohere.errors.BadRequestError:
    interrupted = True
check(interrupted and len(CALLS) == 10, "a chunk interrupted at call 10")
left = embed_all.plan(embed_all.embedder("pro", co, ledger, cache), first)
check(0 < left["calls"] < want["calls"], f"its finished calls are kept ({want['calls']} -> {left['calls']} left)")
FAIL_AT[0] = None
sent_before = {t for c in CALLS[:9] for t in c["texts"]}
n = len(CALLS)
run("--next", "--go")
resent = sent_before & {t for c in CALLS[n:] for t in c["texts"]}
check(not resent, "resuming re-sends nothing that was already embedded")
check(embed_all.plan(embed_all.embedder("pro", co, ledger, cache), first)["calls"] == 0, f"{'/'.join(first)} done")
check(all(c["model"] == "embed-v5.0-pro" and c["truncate"] == "NONE" for c in CALLS), "Pro, truncate NONE")
check({c["input_type"] for c in CALLS} == {"search_document", "search_query"}, "pages and questions both sent")
n = len(CALLS)
run("--next", "--go")
nxt = embed_all.chunks()[1]
check(embed_all.plan(embed_all.embedder("pro", co, ledger, cache), nxt)["calls"] == 0 and len(CALLS) > n,
      f"the next --next runs {'/'.join(nxt)}")
check(ledger.used_this_month() == len(CALLS), "every HTTP attempt is in the ledger")
print(f"{passed} checks passed")
