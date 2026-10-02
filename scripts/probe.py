"""The probe: 9 Embed 5 calls that settle what the main run may assume, before it spends ~220 more.

Texts: 48 pages (16 per set: pages relevant to the chosen questions, then random ones; 48 keeps a call well under
the trial's 100k tokens a minute) and 96 questions (16 English questions in all 6 languages). The calls:
  P1 pro  pages     2048   which types Embed 5 returns; how binary and int8 relate to the float vector
  P2 pro  pages      256   is a shorter vector the first d numbers of the 2048 one, renormalized? (Matryoshka)
  P3 pro  pages      768   the same at 768 (which the SDK's docstring does not list)
  P4 fast pages     2048   Pro vs Fast on the same page (one shared space?)
  P5 pro  questions 2048   question vectors; Pro-index/Fast-query and Fast-index/Pro-query on the 48 pages
  P6 fast questions 2048
  P7 pro  pages     2048   P1 again, the batch reversed: deterministic and independent of batch order?
  P8 pro  questions  256   the Matryoshka check for questions
  P9 fast pages      256   the Matryoshka check for Fast
If P2/P3/P8/P9 match the trimmed 2048 vectors, every size comes from one 2048 call: ~220 calls for everything
instead of ~1,320 (over the trial's 1,000). Usage: python scripts/probe.py [--go]  (without --go: a dry run)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import client as ec  # noqa: E402
import data  # noqa: E402

PRO, FAST, DOC, QRY = "embed-v5.0-pro", "embed-v5.0-fast", "search_document", "search_query"
TYPES = ["float", "int8", "uint8", "binary", "ubinary"]
PLAN = [("P1", PRO, DOC, "pages", 2048), ("P2", PRO, DOC, "pages", 256), ("P3", PRO, DOC, "pages", 768),
        ("P4", FAST, DOC, "pages", 2048), ("P5", PRO, QRY, "questions", 2048), ("P6", FAST, QRY, "questions", 2048),
        ("P7", PRO, DOC, "pages_reversed", 2048), ("P8", PRO, QRY, "questions", 256),
        ("P9", FAST, DOC, "pages", 256)]
QUESTIONS_PER_SET = {"finance_en": 6, "computer_science": 5, "hr": 5}
OUT = ROOT / "probe_run"


PAGES_PER_SET = 16


def pick_texts():
    """-> pages [48 texts], page ids, questions [96 texts], question rows, relevant (en_id -> set of page ids)."""
    pages, page_ids, qrows, rel = [], [], [], {}
    for dom, k in QUESTIONS_PER_SET.items():
        d = data.load(dom)
        en = d.queries[d.queries.language == "english"].sort_values("id")
        chosen = en.sample(k, random_state=0).id.tolist()          # fixed seed: the probe is reproducible
        pos = d.qrels[(d.qrels.language == "english") & d.qrels["query-id"].isin(chosen) & (d.qrels.score > 0)]
        rel |= {q: set(g["corpus-id"]) for q, g in pos.groupby("query-id")}
        body = d.pages().set_index("id").text
        ids = list(dict.fromkeys(pos.sort_values("score", ascending=False)["corpus-id"]))[:PAGES_PER_SET // 2]
        others = body.index.difference(ids).to_series().sample(frac=1.0, random_state=0)
        ids += others[:PAGES_PER_SET - len(ids)].tolist()
        pages += body[ids].tolist()
        page_ids += ids
        qrows.append(d.queries[d.queries.en_id.isin(chosen)].assign(set=dom))
    q = pd.concat(qrows).sort_values(["language", "en_id"])
    assert len(pages) == 3 * PAGES_PER_SET and len(q) == 96
    return pages, page_ids, q.text.tolist(), q, rel


def renorm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def matryoshka(short, long, d):
    """How close the API's d-dim vectors are to the first d numbers of its 2048 vectors (renormalized)."""
    return {"min_cos": float((renorm(short) * renorm(long[:, :d])).sum(1).min()),
            "max_abs_diff_renorm": float(np.abs(short - renorm(long[:, :d])).max()),
            "max_abs_diff_raw": float(np.abs(short - long[:, :d]).max())}


def bits(f, ub, b):
    """Is ubinary the packed sign bits of the float (which bit order)? Is binary ubinary - 128 or its int8 view?"""
    return {"ubinary==signbits(big-endian)": float(np.mean(ub == np.packbits(f > 0, axis=-1))),
            "ubinary==signbits(little-endian)": float(np.mean(ub == np.packbits(f > 0, axis=-1, bitorder="little"))),
            "binary==ubinary-128": float(np.mean(b.astype(np.int16) == ub.astype(np.int16) - 128)),
            "binary==ubinary viewed as int8": float(np.mean(b == ub.view(np.int8))),
            "exact zeros in float": int((f == 0).sum())}


def int8_rule(f, q, u):
    """Which fixed rule turns the float into the API's int8? (exact match, within 1) per candidate rule."""
    m = np.abs(q) < 127
    s = float((q[m] * f[m]).sum() / (f[m] * f[m]).sum())                 # least-squares global scale
    rules = {f"global round(f * {s:.1f})": np.clip(np.round(f * s), -128, 127),
             "per-vector round(f / max|f| * 127)": np.round(f / np.abs(f).max(1, keepdims=True) * 127)}
    out = {k: (float(np.mean(p == q)), float(np.mean(np.abs(p - q) <= 1))) for k, p in rules.items()}
    out["uint8==int8+128"] = float(np.mean(u.astype(np.int16) == q.astype(np.int16) + 128))
    return out


def retrieval(doc, qry, q, page_ids, rel):
    """English questions only: mean rank (1 = best) of each question's relevant pages among the 48, by cosine."""
    en = (q.language == "english").to_numpy()
    s = renorm(qry[en]) @ renorm(doc).T
    ranks = [1 + int((s[i] > s[i, page_ids.index(p)]).sum())
             for i, e in enumerate(q.en_id[en]) for p in rel[e] if p in page_ids]
    return round(float(np.mean(ranks)), 2)


def analyze(R, q, page_ids, rel) -> dict:
    p1, p4, p5, p6 = (R[k]["float"] for k in ("P1", "P4", "P5", "P6"))
    A = {"types returned": sorted(R["P1"]),
         "norm of 2048 vectors (min, median, max)": np.percentile(np.linalg.norm(p1, axis=1), [0, 50, 100]).tolist(),
         "float values exactly fp16": float(np.mean(p1.astype(np.float16).astype(np.float32) == p1))}
    for lab, ref, d in (("P2", p1, 256), ("P3", p1, 768), ("P9", p4, 256), ("P8", p5, 256)):
        A[f"{lab} matryoshka"] = matryoshka(R[lab]["float"], ref, d)
        ref_lab = {"P9": "P4", "P8": "P5"}.get(lab, "P1")
        A[f"{lab} ubinary == first {d // 8} bytes of 2048"] = float(np.mean(R[lab]["ubinary"] ==
                                                                          R[ref_lab]["ubinary"][:, :d // 8]))
        A[f"{lab} int8 == first {d} of 2048"] = float(np.mean(R[lab]["int8"] == R[ref_lab]["int8"][:, :d]))
        A[f"{lab} int8 rule"] = int8_rule(R[lab]["float"], R[lab]["int8"], R[lab]["uint8"])
    A["P1 bits"] = bits(p1, R["P1"]["ubinary"], R["P1"]["binary"])
    A["P5 bits (questions)"] = bits(p5, R["P5"]["ubinary"], R["P5"]["binary"])
    A["P1 int8 rule"] = int8_rule(p1, R["P1"]["int8"], R["P1"]["uint8"])
    cos = lambda a, b: np.percentile((renorm(a) * renorm(b)).sum(1), [5, 50, 95]).round(4).tolist()
    A["cos pro vs fast, same page (p5, p50, p95)"] = cos(p1, p4)
    A["cos pro vs fast, same question (p5, p50, p95)"] = cos(p5, p6)
    A["mean rank of relevant pages among the 48 (index/question)"] = {
        f"{a}/{b}": retrieval(doc, qry, q, page_ids, rel)
        for a, doc in (("pro", p1), ("fast", p4)) for b, qry in (("pro", p5), ("fast", p6))}
    p7 = R["P7"]["float"][::-1]
    A["P7 == P1 bit for bit (reversed batch)"] = bool(np.array_equal(p7, p1))
    A["P7 vs P1 max abs diff"] = float(np.abs(p7 - p1).max())
    return A


def main(go: bool, emb: ec.Embedder | None = None, out: Path = OUT) -> dict | None:
    pages, page_ids, questions, q, rel = pick_texts()
    sets = {"pages": pages, "questions": questions, "pages_reversed": pages[::-1]}
    if emb is None:
        key = ec.load_api_key() if go else "dry-run"
        emb = ec.Embedder(ec.make_client(key), ec.Ledger(run_cap=11), ec.Cache(), raw_dir=out / "raw")
    print(f"ledger: {emb.ledger.used_this_month()} calls this month (cap {emb.ledger.monthly_cap})")
    R = {}
    for label, model, itype, which, dim in PLAN:
        tag = "P7" if label == "P7" else ""                    # P7 must not be served from P1's cache
        R[label] = emb.embed(sets[which], model=model, input_type=itype, dim=dim, types=TYPES, label=label,
                             tag=tag, dry_run=not go)
        print(label, model, itype, which, dim, "->", R[label] if not go else {k: v.shape for k, v in R[label].items()},
              flush=True)
    if go:
        A = analyze(R, q, page_ids, rel)
        A["ledger"] = emb.ledger.db.execute(
            "SELECT label, model, dim, n_texts, status, http_status, billed_input_tokens, latency_s, warnings, "
            "resp_headers FROM calls ORDER BY id").fetchall()
        out.mkdir(exist_ok=True)
        (out / "analysis.json").write_text(json.dumps(A, indent=1, default=str))
        print(json.dumps({k: v for k, v in A.items() if k != "ledger"}, indent=1, default=str))
        return A


if __name__ == "__main__":
    main("--go" in sys.argv)
