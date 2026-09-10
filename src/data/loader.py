"""Data layer: download the MSMARCO-XI validation parquet and build a JSONL subset.

Why not `datasets.load_dataset()`: the HF datasets-server fails on this dataset's
nested `passages` struct ("Nested data conversions not implemented for chunked
array outputs"), and `load_dataset` hits the same nested-struct code path. We read
the parquet directly with pyarrow instead. See VOICE_RAG_BUILD_PLAN.md section 2.2.

The parquet file is a single row group, so range-reads still pull the whole file
(~40s to first row) -- we stream-download once to `data/raw/` and cache it there.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests

from src.config import RAW_DIR, SUBSET_DIR

HF_BASE = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main"


def ensure_raw(lang: str = "hin") -> Path:
    """Stream-download validation/{lang}val.parquet to data/raw/, skipped if cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{lang}val.parquet"
    if dest.exists() and dest.stat().st_size > 0:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[ensure_raw] cached: {dest} ({size_mb:.1f} MB) -- skipping download")
        return dest

    url = f"{HF_BASE}/validation/{lang}val.parquet"
    print(f"[ensure_raw] downloading {url}")
    t0 = time.monotonic()
    tmp = dest.with_suffix(".parquet.part")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                written += len(chunk)
        elapsed = time.monotonic() - t0
    tmp.rename(dest)
    size_mb = written / (1024 * 1024)
    expected_mb = total / (1024 * 1024) if total else float("nan")
    print(f"[ensure_raw] done: {size_mb:.1f} MB (expected {expected_mb:.1f} MB) in {elapsed:.1f}s")
    return dest


def build_subset(n_queries: int = 1000, lang: str = "hin") -> tuple[Path, Path]:
    """Read the cached parquet, filter to rows with >=1 selected passage, emit
    corpus.jsonl (one record per passage) and eval.jsonl (one record per query)."""
    raw_path = ensure_raw(lang)
    print(f"[build_subset] reading {raw_path} (single row group -- this pulls the whole file)")
    t0 = time.monotonic()
    table = pq.read_table(raw_path)
    print(f"[build_subset] loaded {table.num_rows} rows in {time.monotonic() - t0:.1f}s")

    cols = table.select(
        ["query_id", "query", "Eng_Query", "query_type", "passages"]
    ).to_pylist()

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = SUBSET_DIR / "corpus.jsonl"
    eval_path = SUBSET_DIR / "eval.jsonl"

    seen_texts: set[str] = set()
    n_eval_written = 0
    n_corpus_written = 0
    skipped_no_gold = 0
    skipped_malformed = 0

    with open(corpus_path, "w", encoding="utf-8") as corpus_f, open(
        eval_path, "w", encoding="utf-8"
    ) as eval_f:
        for row in cols:
            if n_eval_written >= n_queries:
                break

            passages = row["passages"]
            is_selected = passages.get("is_selected") or []
            eng = passages.get("English_passages") or []
            trans = passages.get("Translated_passages") or []

            if not (len(eng) == len(trans) == len(is_selected)):
                skipped_malformed += 1
                continue

            gold_passage_ids = [
                f"q{row['query_id']}_p{i}" for i, sel in enumerate(is_selected) if sel
            ]
            if not gold_passage_ids:
                skipped_no_gold += 1
                continue

            for i, (text_en, text_hi) in enumerate(zip(eng, trans)):
                doc_id = f"q{row['query_id']}_p{i}"
                is_gold = bool(is_selected[i])
                dedupe_key = (text_hi or "") + "||" + (text_en or "")
                # A gold passage (referenced by this row's own gold_passage_ids,
                # written below) must always get a corpus.jsonl record under
                # its own doc_id, even if its text duplicates an earlier
                # passage -- otherwise the text-dedup below can silently drop
                # the one doc_id this query's eval.jsonl entry actually needs,
                # making that query's gold reference permanently unretrievable
                # by any chunker/strategy, with no error anywhere. Only
                # non-gold candidate passages are deduped on text.
                if not is_gold and dedupe_key in seen_texts:
                    continue
                seen_texts.add(dedupe_key)
                corpus_f.write(
                    json.dumps(
                        {
                            "passage_id": doc_id,
                            "doc_id": doc_id,
                            "query_id": row["query_id"],
                            "text_hi": text_hi,
                            "text_en": text_en,
                            "query_type": row["query_type"],
                            "position": i,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_corpus_written += 1

            eval_f.write(
                json.dumps(
                    {
                        "query_id": row["query_id"],
                        "query_hi": row["query"],
                        "query_en": row["Eng_Query"],
                        "query_type": row["query_type"],
                        "gold_passage_ids": gold_passage_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_eval_written += 1

    assert n_eval_written > 0, "no eval queries survived filtering -- check the data"
    print(
        f"[build_subset] eval queries: {n_eval_written}  "
        f"corpus passages: {n_corpus_written}  "
        f"skipped (no gold): {skipped_no_gold}  skipped (malformed): {skipped_malformed}"
    )
    return corpus_path, eval_path


def _summarize(corpus_path: Path, eval_path: Path) -> None:
    from collections import Counter

    n_gold_per_query = []
    query_types = Counter()
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n_gold_per_query.append(len(rec["gold_passage_ids"]))
            query_types[rec["query_type"]] += 1
            assert len(rec["gold_passage_ids"]) >= 1, f"query {rec['query_id']} has zero gold passages"

    with open(corpus_path, encoding="utf-8") as f:
        n_corpus = sum(1 for _ in f)

    n_eval = len(n_gold_per_query)
    mean_gold = sum(n_gold_per_query) / n_eval if n_eval else 0.0
    print(f"\n=== subset summary ===")
    print(f"eval queries:            {n_eval}")
    print(f"corpus passages:         {n_corpus}")
    print(f"mean gold passages/query: {mean_gold:.3f}")
    print(f"query_type distribution:")
    for qt, count in query_types.most_common():
        print(f"  {qt:<15} {count:>5}  ({count / n_eval * 100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="hin")
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    corpus_path, eval_path = build_subset(n_queries=args.n, lang=args.lang)
    _summarize(corpus_path, eval_path)
