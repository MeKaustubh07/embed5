"""The main embedding run, in chunks of 5-30 minutes. Run one, stop any time (Ctrl-C is safe: every finished call
is cached), and rerun: what is cached is never sent again.

A chunk is one set x one model x one job:
  vectors    the set's pages and its questions in 6 languages, at 2048: float, int8, ubinary. Every shorter size
             and every binary code derives from these (the probe: Matryoshka prefix, binary = sign bits)
  int8-1024  the set's pages at 1024, int8 only: Embed 5's int8 cannot be derived from the float, and 1024 int8
             (1 KB) is the setting Cohere recommends. Questions stay float: only the index is stored
Usage:
  python scripts/embed_all.py                       status of every chunk: calls and minutes left
  python scripts/embed_all.py --next --go           run the next unfinished chunk, in the order below
  python scripts/embed_all.py hr pro vectors --go   run one chunk
Add --with-fast-int8 to include the optional Fast int8-1024 chunks.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import client as ec  # noqa: E402
import data  # noqa: E402

MODELS = {"pro": "embed-v5.0-pro", "fast": "embed-v5.0-fast"}
TOKEN_FACTOR = {"pro": 1.15, "fast": 1.25}        # x the fitted estimate (Fast counts ~9% more); then learned
SETS = ("hr", "computer_science", "finance_en")    # smallest first: a whole set is ready to score early
VECTOR_TYPES = ("float", "int8", "ubinary")


def chunks(with_fast_int8: bool = False) -> list[tuple[str, str, str]]:
    # Pro first, then its int8-1024 (the recommended re-scorer, so the headline two-stage runs on it), then Fast
    out = [(s, "pro", "vectors") for s in SETS] + [(s, "pro", "int8-1024") for s in SETS]
    out += [(s, "fast", "vectors") for s in SETS]
    return out + ([(s, "fast", "int8-1024") for s in SETS] if with_fast_int8 else [])


def work(set_name: str, job: str) -> list[tuple[list[str], str, int, tuple]]:
    """-> [(texts, input_type, dim, types)] for one chunk."""
    d = data.load(set_name)
    pages = list(dict.fromkeys(d.pages().text))
    questions = list(dict.fromkeys(d.queries.text))                      # all 6 languages
    if job == "int8-1024":           # pages only: a stored index is int8, while a question is never stored
        return [(pages, "search_document", 1024, ("int8",))]   # and stays float
    return [(pages, "search_document", 2048, VECTOR_TYPES), (questions, "search_query", 2048, VECTOR_TYPES)]


def embedder(model: str, client=None, ledger=None, cache=None) -> ec.Embedder:
    return ec.Embedder(client if client is not None else ec.make_client(ec.load_api_key()),
                       ledger or ec.Ledger(), cache or ec.Cache(), token_factor=TOKEN_FACTOR[model])


def plan(emb: ec.Embedder, chunk) -> dict:
    """Calls, texts and estimated tokens still to send for a chunk (0 calls: done)."""
    s, m, job = chunk
    tot = {"calls": 0, "texts": 0, "est_tokens": 0}
    for texts, itype, dim, types in work(s, job):
        p = emb.embed(texts, model=MODELS[m], input_type=itype, dim=dim, types=types, dry_run=True)
        tot = {k: tot[k] + p[k] for k in tot}
    # measured on chunk 1: a ~70k-token call waits for the last one to leave the minute window, so ~70k a minute
    tot["minutes"] = round(max(tot["est_tokens"] / (0.7 * emb.tokens_per_minute),
                               tot["calls"] * emb.min_interval_s / 60), 1)
    return tot


def wait_for_previous_minute(ledger: ec.Ledger) -> None:
    """A new process does not know what the last one sent: let the trial's per-minute token window clear."""
    last = ledger.db.execute("SELECT ts FROM calls ORDER BY id DESC LIMIT 1").fetchone()
    if last:
        import datetime as dt
        ago = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(last[0])).total_seconds()
        if ago < 65:
            print(f"waiting {65 - ago:.0f} s for the last run's minute window to clear", flush=True)
            ec._sleep(65 - ago)


def run(chunk, emb: ec.Embedder) -> None:
    s, m, job = chunk
    t0 = time.monotonic()
    for texts, itype, dim, types in work(s, job):
        print(f"  {itype:15s} {len(texts):5d} texts at {dim} ({', '.join(types)})", flush=True)
        emb.embed(texts, model=MODELS[m], input_type=itype, dim=dim, types=types, label=f"{s}/{m}/{job}")
    print(f"  done in {(time.monotonic() - t0) / 60:.1f} min; ledger: {emb.ledger.used_this_month()} calls this "
          f"month (cap {emb.ledger.monthly_cap})", flush=True)


def main(argv: list[str], client=None, ledger=None, cache=None) -> None:
    go, flags = "--go" in argv, {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    todo = chunks("--with-fast-int8" in flags)
    ledger, cache = ledger or ec.Ledger(), cache or ec.Cache()
    plans = {c: plan(embedder(c[1], client or object(), ledger, cache), c) for c in todo}
    if args:
        chunk = tuple(args)
        if chunk not in todo:
            sys.exit(f"unknown chunk {chunk}; one of {todo}")
        selected = [chunk]
    elif "--next" in flags:
        selected = [c for c in todo if plans[c]["calls"]][:1]
    else:
        selected = []
    print(f"ledger: {ledger.used_this_month()} calls this month (cap {ledger.monthly_cap})")
    for c in todo:
        p = plans[c]
        mark = "->" if c in selected else "  "
        state = "done" if not p["calls"] else f"{p['calls']:3d} calls  {p['texts']:5d} texts  ~{p['minutes']} min"
        print(f"{mark} {'/'.join(c):34s} {state}")
    left = sum(p["calls"] for p in plans.values())
    print(f"   total left: {left} calls, ~{sum(p['minutes'] for p in plans.values()):.0f} min")
    if not selected:
        print("nothing selected (all done)" if "--next" in flags else "")
        return
    if ledger.used_this_month() + plans[selected[0]]["calls"] > ledger.monthly_cap:
        sys.exit("this chunk would pass the monthly cap: stopping before any call")
    if not go:
        print("dry run: add --go to send these calls")
        return
    emb = embedder(selected[0][1], client, ledger, cache)
    wait_for_previous_minute(ledger)
    print(f"running {'/'.join(selected[0])}", flush=True)
    run(selected[0], emb)


if __name__ == "__main__":
    main(sys.argv[1:])
