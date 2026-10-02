"""A budgeted, cached client for Cohere Embed 5 (cohere SDK 7.2.0, ClientV2), built for a trial key's 1,000 calls
a month.

- Ledger: every HTTP attempt is written to SQLite BEFORE it is sent, and a hard cap stops the run. The SDK's own
  retries are off (it retries 429/5xx silently, and each retry may count as a call), so no attempt is invisible.
- Cache: one row per (model, input type, dim, embedding type, text). A rerun never pays twice for a vector, and a
  later request that adds a type fetches only what is missing.
- The key is read from .env with python-dotenv and passed to the client; it is never printed, logged or exported.
Both databases live in cache/ (gitignored: the repo is public and raw Cohere outputs stay local).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
MODELS = ("embed-v5.0-pro", "embed-v5.0-fast")
INPUT_TYPES = ("search_document", "search_query")
DIMS = (256, 512, 768, 1024, 1536, 2048)
MAX_TEXTS_PER_CALL = 96
DTYPES = {"float": np.float32, "int8": np.int8, "uint8": np.uint8, "binary": np.int8, "ubinary": np.uint8}
MONTHLY_CAP = 450                       # our hard stop: under half the trial's 1,000, leaving room for reruns
TOKENS_PER_MINUTE = 100_000             # trial limit, undocumented: from the 429 message, 2026-10-02
CALL_TOKEN_CAP = 80_000                 # tokens per call: one call must fit inside 90% of a minute's limit
CACHE_VERSION = 1
_clock, _sleep = time.monotonic, time.sleep     # module-level, so tests can run the pacing on a fake clock


class BudgetExhausted(RuntimeError):
    pass


class _Rebatch(Exception):
    """A call was refused for the token limit: halve the tokens per call and split what is still missing."""


def load_api_key(env_path: Path = ROOT / ".env") -> str:
    from dotenv import dotenv_values

    key = (dotenv_values(env_path) or {}).get("CO_API_KEY")
    if not key:
        raise RuntimeError(f"CO_API_KEY missing in {env_path}")
    return key.strip()


def make_client(api_key: str, timeout_s: float = 300.0):
    """timeout: a 96-page batch can be ~100k tokens; a timed-out call is not retried by the SDK and may still be
    billed, so keep the SDK's default of 300 s until real latencies are known."""
    import cohere

    return cohere.ClientV2(api_key=api_key, client_name="32-bytes", timeout=timeout_s, max_retries=0)


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(model: str, input_type: str, dim: int, etype: str, text: str, tag: str = "") -> str:
    """tag "" for the real vectors; a probe that must send cached texts again (the determinism check) uses a tag.
    Every vector is requested with truncate="NONE", so a cached vector always saw its whole text."""
    spec = {"v": CACHE_VERSION, "model": model, "input_type": input_type, "dim": int(dim), "etype": etype,
            "truncate": "NONE", "tag": tag, "text": text_sha(text)}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def keep_headers(headers) -> str:
    """Response headers worth logging (rate limits, trial counters, request ids). Request auth is never in here."""
    pat = ("limit", "remaining", "reset", "retry", "warning", "trace", "request-id", "x-api", "num_")
    return json.dumps({k: v for k, v in dict(headers or {}).items() if any(p in k.lower() for p in pat)})


class Ledger:
    """One row per HTTP attempt, inserted before sending, so a crash mid-call still counts."""

    def __init__(self, path: Path = CACHE / "ledger.sqlite", monthly_cap: int = MONTHLY_CAP,
                 run_cap: int | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS calls (
            id INTEGER PRIMARY KEY, ts TEXT, month TEXT, label TEXT, endpoint TEXT, model TEXT,
            input_type TEXT, dim INTEGER, types TEXT, n_texts INTEGER, attempt INTEGER,
            status TEXT, http_status INTEGER, billed_input_tokens REAL, tokens_input REAL,
            cached_tokens REAL, warnings TEXT, api_version TEXT, response_id TEXT,
            latency_s REAL, error TEXT, resp_headers TEXT)""")
        self.monthly_cap, self.run_cap, self.run_count = monthly_cap, run_cap, 0

    def used_this_month(self) -> int:
        month = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m")
        return self.db.execute("SELECT COUNT(*) FROM calls WHERE month=?", (month,)).fetchone()[0]

    def reserve(self, **row) -> int:
        used = self.used_this_month()
        if used >= self.monthly_cap:
            raise BudgetExhausted(f"ledger: {used} attempts this month, hard cap {self.monthly_cap}")
        if self.run_cap is not None and self.run_count >= self.run_cap:
            raise BudgetExhausted(f"ledger: run cap {self.run_cap} reached")
        now = dt.datetime.now(dt.timezone.utc)
        row = {**row, "ts": now.isoformat(timespec="seconds"), "month": now.strftime("%Y-%m"), "status": "sent"}
        cur = self.db.execute(f"INSERT INTO calls ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                              tuple(row.values()))
        self.run_count += 1
        return cur.lastrowid

    def finish(self, call_id: int, **fields) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE calls SET {sets} WHERE id=?", (*fields.values(), call_id))


class Cache:
    def __init__(self, path: Path = CACHE / "vectors.sqlite"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS vecs (
            key TEXT PRIMARY KEY, model TEXT, input_type TEXT, dim INTEGER, etype TEXT,
            text_sha TEXT, n INTEGER, blob BLOB, call_id INTEGER, created TEXT)""")

    def get(self, key: str, etype: str) -> np.ndarray | None:
        r = self.db.execute("SELECT blob FROM vecs WHERE key=?", (key,)).fetchone()
        return None if r is None else np.frombuffer(r[0], dtype=DTYPES[etype]).copy()

    def put_many(self, rows: list[tuple]) -> None:
        self.db.execute("BEGIN")
        self.db.executemany("INSERT OR IGNORE INTO vecs VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        self.db.execute("COMMIT")


@dataclass
class Embedder:
    client: object
    ledger: Ledger
    cache: Cache
    min_interval_s: float = 12.5          # the FAQ's stricter trial figure, 5 Embed calls a minute
    max_attempts: int = 3                 # per batch; each attempt is a ledger row
    raw_dir: Path | None = None           # keep each response's raw JSON (the probe)
    tokens_per_minute: int = TOKENS_PER_MINUTE
    call_token_cap: int = CALL_TOKEN_CAP
    chars_per_token: float = 3.0          # cautious until a response reports real counts (then the lowest seen)
    _last: float = 0.0
    _sent: list = None                    # (time, tokens) of the calls in the last minute
    _observed: float | None = None        # chars per token, lowest measured so far

    def est_tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token) + 8          # + the model's own special tokens

    def _batches(self, texts: list[str]) -> list[list[str]]:
        """Greedy: up to 96 texts and call_token_cap estimated tokens per call."""
        out, cur, tok = [], [], 0
        for t in texts:
            n = self.est_tokens(t)
            if cur and (len(cur) == MAX_TEXTS_PER_CALL or tok + n > self.call_token_cap):
                out.append(cur)
                cur, tok = [], 0
            cur.append(t)
            tok += n
        return out + ([cur] if cur else [])

    def _pace(self, tokens: int) -> None:
        """At least min_interval_s between calls, and at most 90% of the per-minute token limit in any minute."""
        wait = self.min_interval_s - (_clock() - self._last)
        if wait > 0:
            _sleep(wait)
        self._sent = [(t, n) for t, n in (self._sent or []) if _clock() - t < 60.0]
        while self._sent and sum(n for _, n in self._sent) + tokens > 0.9 * self.tokens_per_minute:
            _sleep(max(0.0, 60.0 - (_clock() - self._sent[0][0])) + 0.5)
            self._sent = [(t, n) for t, n in self._sent if _clock() - t < 60.0]
        self._last = _clock()
        self._sent.append((self._last, tokens))

    def embed(self, texts: Sequence[str], *, model: str, input_type: str, dim: int = 2048,
              types: Sequence[str] = ("float", "int8", "ubinary"), label: str = "", tag: str = "",
              dry_run: bool = False) -> dict[str, np.ndarray] | dict[str, int]:
        """Vectors for texts, in input order: {type: [len(texts), n]}. Only texts missing from the cache are sent,
        in calls of up to 96 texts and call_token_cap estimated tokens. dry_run: {"calls", "texts", "est_tokens"}
        that would be sent."""
        assert model in MODELS and input_type in INPUT_TYPES and dim in DIMS and set(types) <= set(DTYPES)
        keys = {(t, e): cache_key(model, input_type, dim, e, t, tag) for t in texts for e in types}
        missing = list(dict.fromkeys(t for (t, e), k in keys.items() if self.cache.get(k, e) is None))
        if dry_run:
            return {"calls": len(self._batches(missing)), "texts": len(missing),
                    "est_tokens": sum(self.est_tokens(t) for t in missing)}
        while missing:                    # plan one call at a time: each uses the latest token estimate
            batch = self._batches(missing)[0]
            try:
                self._call(batch, model, input_type, dim, list(types), keys, label)
            except _Rebatch:
                continue                  # the cap was halved: the next batch is smaller
            sent = set(batch)
            missing = [t for t in missing if t not in sent]
        return {e: np.stack([self.cache.get(keys[(t, e)], e) for t in texts]) for e in types}

    def _call(self, batch, model, input_type, dim, types, keys, label) -> None:
        import cohere
        import httpx

        # truncate="NONE": a text over the model's limit is an error, never a silently shortened vector
        kwargs = dict(model=model, input_type=input_type, texts=batch, output_dimension=dim, embedding_types=types,
                      truncate="NONE")
        est = sum(self.est_tokens(t) for t in batch)
        for attempt in range(1, self.max_attempts + 1):
            cid = self.ledger.reserve(label=label, endpoint="v2/embed", model=model, input_type=input_type, dim=dim,
                                      types=",".join(types), n_texts=len(batch), attempt=attempt)
            self._pace(est)
            t0 = _clock()
            try:
                raw = self.client.with_raw_response.embed(**kwargs, request_options={"max_retries": 0})
            except cohere.errors.TooManyRequestsError as ex:
                ra = (ex.headers or {}).get("retry-after")
                self.ledger.finish(cid, status="429", http_status=429, error=str(ex.body)[:500],
                                   resp_headers=keep_headers(ex.headers))
                if any(w in str(ex.body).lower() for w in ("month", "quota")):   # waiting will not help
                    raise BudgetExhausted(f"Cohere refused the call for a quota: {str(ex.body)[:300]}") from ex
                _sleep(min(float(ra), 300.0) if ra and ra.strip().isdigit() else 61.0)
                if "token" in str(ex.body).lower() and len(batch) > 1:   # this batch alone may be over the limit
                    self.call_token_cap = max(1_000, self.call_token_cap // 2)
                    raise _Rebatch from ex
                continue
            except (httpx.TimeoutException, httpx.TransportError) as ex:
                self.ledger.finish(cid, status="transport", error=type(ex).__name__)   # may still be billed
                _sleep(10.0 * attempt)
                continue
            except cohere.core.api_error.ApiError as ex:
                if ex.status_code is not None and ex.status_code >= 500:            # any 5xx, incl. a bare 502
                    self.ledger.finish(cid, status="5xx", http_status=ex.status_code, error=str(ex.body)[:500])
                    _sleep(5.0 * 2 ** attempt)
                    continue
                self.ledger.finish(cid, status="error", http_status=ex.status_code, error=str(ex.body)[:500])
                raise                                                               # 4xx: retrying will not help
            except Exception as ex:                                                 # e.g. an unparsable response
                self.ledger.finish(cid, status="error", error=f"{type(ex).__name__}: {ex}"[:500])
                raise
            self.ledger.finish(cid, resp_headers=keep_headers(raw.headers))
            self._store(raw.data, cid, batch, model, input_type, dim, types, keys, t0, label)
            return
        raise RuntimeError(f"batch failed after {self.max_attempts} attempts (see the ledger)")

    def _store(self, r, cid, batch, model, input_type, dim, types, keys, t0, label) -> None:
        m, e = r.meta, r.embeddings
        bu = getattr(m, "billed_units", None)
        av = getattr(m, "api_version", None)
        counted = max(getattr(bu, "input_tokens", None) or 0,
                      getattr(getattr(m, "tokens", None), "input_tokens", None) or 0)
        chars = sum(len(t) for t in batch)
        if counted and self._sent:                   # the minute window counts what the server counted
            self._sent[-1] = (self._sent[-1][0], max(self._sent[-1][1], int(counted)))
        if counted and chars >= 500 * len(batch):    # learn the real rate from page-length texts only: on short
            seen = chars / counted                   # questions each text's fixed token overhead dominates
            # keep the densest rate seen (most tokens per char), so estimates stay on the cautious side
            self.chars_per_token = seen if self._observed is None else min(self._observed, seen)
            self._observed = self.chars_per_token
        self.ledger.finish(
            cid, status="ok", http_status=200, latency_s=round(_clock() - t0, 3), response_id=r.id,
            billed_input_tokens=getattr(bu, "input_tokens", None),
            tokens_input=getattr(getattr(m, "tokens", None), "input_tokens", None),
            cached_tokens=getattr(m, "cached_tokens", None), warnings=json.dumps(getattr(m, "warnings", None)),
            api_version=json.dumps(av.dict() if av is not None else None))
        if self.raw_dir is not None:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            (self.raw_dir / f"call{cid:05d}_{label or 'x'}.json").write_text(json.dumps(
                {"request": {"model": model, "input_type": input_type, "dim": dim, "types": types,
                             "text_sha256": [text_sha(t) for t in batch]},
                 "response": r.dict(by_alias=True)}))
        rows, now = [], dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        for etype in types:
            arrs = getattr(e, "float_" if etype == "float" else etype)
            if arrs is None or len(arrs) != len(batch):
                raise RuntimeError(f"response lacks {etype!r} for all {len(batch)} texts (call {cid})")
            for text, vec in zip(batch, arrs):
                a = np.asarray(vec, dtype=DTYPES[etype])
                rows.append((keys[(text, etype)], model, input_type, dim, etype, text_sha(text), a.size,
                             a.tobytes(), cid, now))
        self.cache.put_many(rows)
