"""Regression test for a real bug found by code review: build_subset's
text-based dedup could silently drop a passage that duplicates earlier text
but is itself a *different query's gold passage* -- making that query's gold
reference permanently unretrievable, with no error anywhere. Verified by
constructing a synthetic two-row parquet file (no network needed) where a
duplicate passage's text is gold for the second row but not the first."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import src.data.loader as loader_module


def _make_synthetic_parquet(path):
    # Row 1 (query_id=1): passage 0's text is NOT gold for this query.
    # Row 2 (query_id=2): passage 0's text is IDENTICAL to row 1's passage 0
    # text, but IS gold for query 2. A naive text-dedup would keep only the
    # first occurrence (query 1's, non-gold) and silently drop query 2's own
    # doc_id ("q2_p0") from corpus.jsonl -- exactly the bug this test guards.
    shared_text_hi = "साझा पाठ जो दोहराया गया है।"
    shared_text_en = "Shared text that is duplicated."
    unique_text_hi = "अद्वितीय पाठ जो निगम के बारे में है।"
    unique_text_en = "Unique text about corporations."

    rows = [
        {
            "query_id": 1,
            "query": "प्रश्न एक",
            "Eng_Query": "question one",
            "query_type": "DESCRIPTION",
            "passages": {
                "is_selected": [0, 1],
                "English_passages": [shared_text_en, unique_text_en],
                "Translated_passages": [shared_text_hi, unique_text_hi],
            },
        },
        {
            "query_id": 2,
            "query": "प्रश्न दो",
            "Eng_Query": "question two",
            "query_type": "DESCRIPTION",
            "passages": {
                "is_selected": [1, 0],
                "English_passages": [shared_text_en, unique_text_en],
                "Translated_passages": [shared_text_hi, unique_text_hi],
            },
        },
    ]

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


@pytest.fixture
def synthetic_subset(tmp_path, monkeypatch):
    raw_path = tmp_path / "synthetic.parquet"
    _make_synthetic_parquet(raw_path)
    monkeypatch.setattr(loader_module, "ensure_raw", lambda lang="hin": raw_path)
    monkeypatch.setattr(loader_module, "SUBSET_DIR", tmp_path / "subset")

    corpus_path, eval_path = loader_module.build_subset(n_queries=10, lang="hin")
    return corpus_path, eval_path


def test_gold_passage_survives_text_dedup_across_queries(synthetic_subset):
    corpus_path, eval_path = synthetic_subset

    corpus_doc_ids = set()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            corpus_doc_ids.add(json.loads(line)["doc_id"])

    gold_ids = set()
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gold_ids.update(rec["gold_passage_ids"])

    # query 1's gold is q1_p1 (the unique passage); query 2's gold is q2_p0
    # (text-duplicate of q1_p0, but must still get its own corpus record).
    assert "q2_p0" in gold_ids
    assert "q2_p0" in corpus_doc_ids, (
        "query 2's gold passage was dropped by text-dedup even though its "
        "own query needs it -- this is the exact bug the fix addresses"
    )
    assert gold_ids <= corpus_doc_ids, "every gold_passage_id must have a corpus.jsonl record"
