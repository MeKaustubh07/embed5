"""The data "32 Bytes" measures: three ViDoRe v3 sets from Cohere's RCP-nDCG release
(fabianschmidt-cohere/rcp-ndcg-vidore-v3), as local parquet files written by scripts/fetch_data.py.

Per set: the pages' parsed text (the page images are never downloaded), the questions in 6 languages, and for every
question its judged shortlist ("pool") of 150 pages with an RCP gain per page. The documents of all three sets are
English, so the 5 other languages are cross-language search. A translated question has its own id; en_id is the
English original's. Shortlists and gains are per language too (a translation was pooled and judged on its own).
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = "fabianschmidt-cohere/rcp-ndcg-vidore-v3"
REV = "4e5ca46e2f9c050b18a7f170c23e3ade66f78a4e"     # pinned: the scores must use the same judged pools as the data
UPSTREAM = {                                          # vidore/vidore_v3_<set>: page -> document map, answers, licences
    "finance_en": "7f432c176d82e27546501ad8064a713ac3071809",
    "computer_science": "d5cc75883d92e294f0c0fc2662551c9708a06ebc",
    "hr": "0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3",
}
DOMAINS = tuple(UPSTREAM)                             # US 10-K finance, CS textbooks, EU HR reports
LANGS = ("english", "french", "german", "italian", "portuguese", "spanish")
DATA = Path(__file__).resolve().parents[1] / "data"


def subset(domain: str, lang: str) -> str:
    """rcp-ndcg's name for one set in one language. A wrong name scores 0.0 without an error, so build it here."""
    assert domain in DOMAINS and lang in LANGS, (domain, lang)
    return f"{domain}__{lang}"


@dataclass
class Domain:
    name: str
    corpus: pd.DataFrame      # id ('corpus-test-<n>'), text, doc_id, page_number_in_doc
    queries: pd.DataFrame     # language, id ('query-test-<n>'), text, en_id
    qrels: pd.DataFrame       # language, query-id, corpus-id, score (human grade 0/1/2), gain (RCP, 0..1), theta
    pools: pd.DataFrame       # language, query-id, corpus-ids (the 150-page judged shortlist)
    documents: pd.DataFrame   # the source documents: file_name, url, licence, ...

    def pages(self) -> pd.DataFrame:
        """The pages to embed: 33 of the 5,412 have no text (none is in any shortlist), and an empty input could
        fail a whole 96-text request."""
        return self.corpus[self.corpus.text.str.strip() != ""]


def load(domain: str, data_dir: Path = DATA) -> Domain:
    if not (data_dir / f"{domain}.corpus.parquet").exists():
        raise FileNotFoundError(f"{data_dir}/{domain}.*.parquet missing: run scripts/fetch_data.py first")
    read = lambda kind: pd.read_parquet(data_dir / f"{domain}.{kind}.parquet")
    return Domain(domain, read("corpus"), read("queries"), read("qrels"), read("top_ranked"), read("documents"))
