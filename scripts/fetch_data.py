"""Download the three ViDoRe v3 sets into data/ (about 28 MB over the network, ~4 min: the Hub's latency per file
dominates). Reads the page text by column projection, so the page images (99.6% of the corpus files, 2.2 GB)
are never fetched. Pinned revisions: rerunning gives the same files.

Writes per set: <set>.corpus / .queries / .qrels / .top_ranked / .documents .parquet (see src/data.py).
Usage: python scripts/fetch_data.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import DATA, LANGS, REPO, REV, UPSTREAM  # noqa: E402

fs = HfFileSystem()


def read(path: str, columns: list[str] | None = None, tries: int = 4) -> pd.DataFrame:
    for i in range(tries):              # the Hub occasionally drops a connection
        try:                            # cache_type="none": exact range requests, so only the asked columns arrive
            with fs.open(path, "rb", cache_type="none", block_size=1 << 20) as fh:
                return pq.ParquetFile(fh).read(columns=columns).to_pandas()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            print(f"  retry {i + 1} for {path}: {type(e).__name__}", flush=True)
            time.sleep(2 ** i)


def read_many(glob: str, columns: list[str]) -> pd.DataFrame:
    return pd.concat([read(f, columns) for f in sorted(fs.glob(glob))], ignore_index=True)


def align_to_english(q: pd.DataFrame) -> pd.DataFrame:
    """en_id for every question. Translations share no id: each language's ids come in the same contiguous blocks
    (two batches in some sets), so a translation is the question at the same (block, offset) in English."""
    q = q.assign(n=q.id.str.rsplit("-", n=1).str[1].astype(int)).sort_values(["language", "n"])
    block = q.groupby("language").n.transform(lambda s: (s.diff().fillna(1) != 1).cumsum())
    q["key"] = block.astype(str) + ":" + q.groupby(["language", block]).cumcount().astype(str)
    en = q[q.language == "english"].set_index("key").id
    return q.assign(en_id=q.key.map(en)).drop(columns=["n", "key"]).sort_values(["language", "en_id"], kind="stable")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    base = f"datasets/{REPO}@{REV}"
    for dom, up_rev in UPSTREAM.items():
        up = f"datasets/vidore/vidore_v3_{dom}@{up_rev}"
        corpus = read_many(f"{base}/corpus/{dom}/*.parquet", ["id", "text"])
        pages = read_many(f"{up}/corpus/*.parquet", ["corpus_id", "doc_id", "page_number_in_doc"])
        pages["id"] = "corpus-test-" + pages.pop("corpus_id").astype(str)
        corpus = corpus.merge(pages, on="id", how="left", validate="one_to_one")
        assert corpus.doc_id.notna().all()

        tabs = {name: [] for name in ("queries", "qrels", "top_ranked")}
        for lang in LANGS:
            for name, acc in tabs.items():
                acc.append(read(f"{base}/{dom}__{lang}/{name}.parquet").assign(language=lang))
        q = align_to_english(pd.concat(tabs["queries"], ignore_index=True))
        # check the alignment with a field that was left untranslated: each question's reference answers
        ans = read(f"{up}/queries/test-00000-of-00001.parquet", ["query_id", "raw_answers"])
        ans = pd.Series([tuple(a) for a in ans.raw_answers], index="query-test-" + ans.query_id.astype(str))
        assert q.en_id.notna().all() and ans[q[q.language == "english"].id].is_unique
        bad = int((ans[q.id].to_numpy() != ans[q.en_id].to_numpy()).sum())
        assert bad == 0, f"{dom}: {bad} translations not aligned with their English original"

        corpus.to_parquet(DATA / f"{dom}.corpus.parquet", index=False)
        read(f"{up}/documents_metadata/test-00000-of-00001.parquet").to_parquet(
            DATA / f"{dom}.documents.parquet", index=False)
        q[["language", "id", "text", "en_id"]].to_parquet(DATA / f"{dom}.queries.parquet", index=False)
        qrels = pd.concat(tabs["qrels"], ignore_index=True)
        qrels[["language", "query-id", "corpus-id", "score", "gain", "theta"]].to_parquet(
            DATA / f"{dom}.qrels.parquet", index=False)
        pools = pd.concat(tabs["top_ranked"], ignore_index=True)
        pools[["language", "query-id", "corpus-ids"]].to_parquet(DATA / f"{dom}.top_ranked.parquet", index=False)
        print(f"{dom:17s} pages {len(corpus):5d}  questions per language {len(q) // len(LANGS)}  "
              f"judged pairs {len(qrels)}", flush=True)


if __name__ == "__main__":
    main()
