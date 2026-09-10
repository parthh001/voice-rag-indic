"""Phase 4 centerpiece: Recall@k / MRR for fixed vs semantic vs metadata-aware chunking,
across three corpus modes (hi->hi monolingual, hi->en cross-lingual, en->en baseline),
stratified by query_type. Every eval query has ground truth from `is_selected`
(see VOICE_RAG_BUILD_PLAN.md section 2.5) -- this is what makes the comparison real
numbers instead of opinion.

Fairness gotcha (plan section 7 Phase 4): identical embedding model, identical k,
identical eval set across all strategies -- and because chunk counts differ across
strategies, we report chunk counts alongside recall so a strategy that merely
produces more chunks (more shots at recall) isn't mistaken for a "better" one.

Corpus-mode index reuse: hi->en and en->en both search the SAME English-chunked
corpus (only the query language differs), so each strategy only needs two indices
built (Hindi-chunked, English-chunked), not three -- this is a real compute saving,
not a shortcut on the eval itself.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.chunking.fixed import FixedChunker
from src.chunking.metadata_aware import MetadataAwareChunker
from src.chunking.semantic import SemanticChunker
from src.config import RESULTS_DIR, SUBSET_DIR
from src.contracts import Chunk
from src.embeddings import encode_queries
from src.store import VectorStore

K_VALUES = (1, 3, 5, 10)
MAX_K = max(K_VALUES)
SEED = 42

STRATEGIES = {
    "fixed": FixedChunker,
    "semantic": SemanticChunker,
    "metadata_aware": MetadataAwareChunker,
}

CORPUS_MODES = {
    # mode_name: (query_field, corpus_lang)
    "hi_to_hi": ("query_hi", "hi"),
    "hi_to_en": ("query_hi", "en"),
    "en_to_en": ("query_en", "en"),
}


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _chunk_corpus(chunker, corpus: list[dict], lang: str) -> list[Chunk]:
    text_field = "text_hi" if lang == "hi" else "text_en"
    all_chunks: list[Chunk] = []
    for rec in corpus:
        text = rec[text_field]
        if not text or not text.strip():
            continue
        metadata = {
            "query_type": rec["query_type"],
            "position": rec["position"],
            "lang": lang,
        }
        all_chunks.extend(chunker.chunk(text, doc_id=rec["doc_id"], metadata=metadata))
    return all_chunks


def _build_index(strategy_name: str, chunker, corpus: list[dict], lang: str) -> tuple[VectorStore, float, int, float]:
    t0 = time.monotonic()
    chunks = _chunk_corpus(chunker, corpus, lang)
    store = VectorStore(f"cmp_{strategy_name}_{lang}", reset=True)
    store.index(chunks)
    build_s = time.monotonic() - t0
    mean_len = float(np.mean([len(c.text.split()) for c in chunks])) if chunks else 0.0
    return store, build_s, len(chunks), mean_len


def _search_with_precomputed(store: VectorStore, query_vec: np.ndarray, k: int) -> list[str]:
    """Bypass per-call query encoding (already batched by the caller) and return
    the list of doc_ids in rank order."""
    result = store.collection.query(query_embeddings=[query_vec.tolist()], n_results=k)
    metas = result["metadatas"][0]
    return [m["doc_id"] for m in metas]


def _metrics_for_ranked_doc_ids(ranked_doc_ids: list[str], gold_ids: set[str]) -> dict:
    out = {f"recall@{k}": 0.0 for k in K_VALUES}
    rr = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:MAX_K], start=1):
        if doc_id in gold_ids:
            if rr == 0.0:
                rr = 1.0 / rank
            for k in K_VALUES:
                if rank <= k:
                    out[f"recall@{k}"] = 1.0
    out["mrr@10"] = rr
    return out


def run_comparison(n_queries: int | None = None) -> dict:
    np.random.seed(SEED)
    corpus_path = SUBSET_DIR / "corpus.jsonl"
    eval_path = SUBSET_DIR / "eval.jsonl"
    corpus = _load_jsonl(corpus_path)
    eval_set = _load_jsonl(eval_path)
    if n_queries:
        eval_set = eval_set[:n_queries]

    print(f"[chunking_comparison] corpus={len(corpus)} passages, eval={len(eval_set)} queries")

    # Precompute all query embeddings once per language (reused across all strategies).
    hi_queries = [q["query_hi"] for q in eval_set]
    en_queries = [q["query_en"] for q in eval_set]
    print("[chunking_comparison] encoding all eval queries (hi + en)...")
    hi_query_vecs = encode_queries(hi_queries)
    en_query_vecs = encode_queries(en_queries)

    results: dict = {"strategies": {}}

    for strategy_name, ChunkerCls in STRATEGIES.items():
        chunker = ChunkerCls()
        print(f"\n=== strategy: {strategy_name} ===")

        store_hi, build_s_hi, n_chunks_hi, mean_len_hi = _build_index(
            strategy_name, chunker, corpus, "hi"
        )
        print(
            f"[{strategy_name}] hi index: {n_chunks_hi} chunks, "
            f"mean_len={mean_len_hi:.1f} words, build={build_s_hi:.1f}s"
        )

        store_en, build_s_en, n_chunks_en, mean_len_en = _build_index(
            strategy_name, chunker, corpus, "en"
        )
        print(
            f"[{strategy_name}] en index: {n_chunks_en} chunks, "
            f"mean_len={mean_len_en:.1f} words, build={build_s_en:.1f}s"
        )

        strategy_result: dict = {
            "n_chunks_hi": n_chunks_hi,
            "n_chunks_en": n_chunks_en,
            "mean_chunk_len_hi": mean_len_hi,
            "mean_chunk_len_en": mean_len_en,
            "build_s_hi": build_s_hi,
            "build_s_en": build_s_en,
            "modes": {},
        }

        for mode_name, (query_field, corpus_lang) in CORPUS_MODES.items():
            store = store_hi if corpus_lang == "hi" else store_en
            query_vecs = hi_query_vecs if query_field == "query_hi" else en_query_vecs

            per_query_metrics = []
            per_type_metrics: dict = defaultdict(list)

            for q_rec, q_vec in zip(eval_set, query_vecs):
                gold_ids = set(q_rec["gold_passage_ids"])
                ranked = _search_with_precomputed(store, q_vec, MAX_K)
                m = _metrics_for_ranked_doc_ids(ranked, gold_ids)
                per_query_metrics.append(m)
                per_type_metrics[q_rec["query_type"]].append(m)

            agg = {
                key: float(np.mean([m[key] for m in per_query_metrics]))
                for key in per_query_metrics[0]
            }
            by_type = {
                qt: {key: float(np.mean([m[key] for m in ms])) for key in ms[0]}
                for qt, ms in per_type_metrics.items()
            }

            strategy_result["modes"][mode_name] = {"overall": agg, "by_query_type": by_type}
            print(
                f"[{strategy_name}/{mode_name}] "
                f"R@1={agg['recall@1']:.3f} R@3={agg['recall@3']:.3f} "
                f"R@5={agg['recall@5']:.3f} R@10={agg['recall@10']:.3f} "
                f"MRR@10={agg['mrr@10']:.3f}"
            )

        results["strategies"][strategy_name] = strategy_result

    return results


def _write_markdown(results: dict, out_path: Path) -> None:
    lines = ["# Chunking Comparison — Recall@k / MRR", ""]
    lines.append(
        "Ground truth: `passages.is_selected` from MSMARCO-XI validation "
        "(real human relevance labels, not synthetic)."
    )
    lines.append("")
    lines.append(
        "| Strategy | Mode | Chunks | Mean chunk len (words) | R@1 | R@3 | R@5 | R@10 | MRR@10 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for strategy_name, s in results["strategies"].items():
        for mode_name in CORPUS_MODES:
            m = s["modes"][mode_name]["overall"]
            n_chunks = s["n_chunks_hi"] if mode_name == "hi_to_hi" else s["n_chunks_en"]
            mean_len = s["mean_chunk_len_hi"] if mode_name == "hi_to_hi" else s["mean_chunk_len_en"]
            lines.append(
                f"| {strategy_name} | {mode_name} | {n_chunks} | {mean_len:.1f} | "
                f"{m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | "
                f"{m['recall@10']:.3f} | {m['mrr@10']:.3f} |"
            )

    lines.append("")
    lines.append("## Stratified by query_type (recall@5)")
    lines.append("")
    lines.append("| Strategy | Mode | query_type | n | R@5 | MRR@10 |")
    lines.append("|---|---|---|---|---|---|")
    eval_set = _load_jsonl(SUBSET_DIR / "eval.jsonl")
    type_counts: dict = defaultdict(int)
    for rec in eval_set:
        type_counts[rec["query_type"]] += 1
    for strategy_name, s in results["strategies"].items():
        for mode_name in CORPUS_MODES:
            by_type = s["modes"][mode_name]["by_query_type"]
            for qt, m in sorted(by_type.items(), key=lambda kv: -type_counts.get(kv[0], 0)):
                lines.append(
                    f"| {strategy_name} | {mode_name} | {qt} | {type_counts.get(qt, 0)} | "
                    f"{m['recall@5']:.3f} | {m['mrr@10']:.3f} |"
                )

    lines.append("")
    lines.append(
        "**Fairness note:** strategies produce different chunk counts (see table above). "
        "A strategy with more chunks gets more independent shots at recall; chunk counts "
        "are reported alongside recall for exactly this reason -- read recall differences "
        "together with chunk count, not in isolation."
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="limit eval queries (default: all)")
    args = parser.parse_args()

    results = run_comparison(n_queries=args.n)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "chunking_comparison.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = RESULTS_DIR / "chunking_comparison.md"
    _write_markdown(results, md_path)

    print(f"\nWrote {json_path} and {md_path}")
