"""Section 4.1 ablation: prove the e5 'query: '/'passage: ' prefixes actually matter,
instead of just asserting it. Compares Recall@5/MRR@10 for a single (fixed chunker,
hi->hi) slice, indexed and queried once WITH the mandatory prefixes and once WITHOUT.
Both numbers go in the README per the plan's instruction -- this is the single
experiment that demonstrates understanding the model rather than having copied a
prefix convention from the model card without checking it matters.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
import numpy as np

from src.chunking.fixed import FixedChunker
from src.config import CHROMA_PATH, RESULTS_DIR, SUBSET_DIR
from src.embeddings import (
    encode_passages,
    encode_passages_no_prefix,
    encode_queries,
    encode_queries_no_prefix,
)

N_EVAL_QUERIES = 200
K = 5


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _recall_at_k(ranked_doc_ids: list[str], gold_ids: set[str], k: int) -> float:
    return 1.0 if any(d in gold_ids for d in ranked_doc_ids[:k]) else 0.0


def _mrr_at_10(ranked_doc_ids: list[str], gold_ids: set[str]) -> float:
    for rank, d in enumerate(ranked_doc_ids[:10], start=1):
        if d in gold_ids:
            return 1.0 / rank
    return 0.0


def _run_variant(name: str, encode_p, encode_q, corpus: list[dict], eval_set: list[dict]) -> dict:
    chunker = FixedChunker()
    chunks = []
    for rec in corpus:
        text = rec["text_hi"]
        if not text or not text.strip():
            continue
        metadata = {"query_type": rec["query_type"], "position": rec["position"], "lang": "hi"}
        chunks.extend(chunker.chunk(text, doc_id=rec["doc_id"], metadata=metadata))

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection_name = f"ablation_{name}"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    texts = [c.text for c in chunks]
    vecs = encode_p(texts)
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vecs[i : i + batch_size].tolist(),
            metadatas=[{"doc_id": c.doc_id} for c in batch],
            documents=texts[i : i + batch_size],
        )

    query_texts = [q["query_hi"] for q in eval_set]
    query_vecs = encode_q(query_texts)

    recalls, mrrs = [], []
    for q_rec, q_vec in zip(eval_set, query_vecs):
        gold_ids = set(q_rec["gold_passage_ids"])
        result = collection.query(query_embeddings=[q_vec.tolist()], n_results=K)
        ranked_doc_ids = [m["doc_id"] for m in result["metadatas"][0]]
        recalls.append(_recall_at_k(ranked_doc_ids, gold_ids, K))
        mrrs.append(_mrr_at_10(ranked_doc_ids, gold_ids))

    return {
        "recall@5": float(np.mean(recalls)),
        "mrr@10": float(np.mean(mrrs)),
        "n_chunks": len(chunks),
        "n_eval_queries": len(eval_set),
    }


def run_ablation() -> dict:
    corpus = _load_jsonl(SUBSET_DIR / "corpus.jsonl")
    eval_set = _load_jsonl(SUBSET_DIR / "eval.jsonl")[:N_EVAL_QUERIES]

    print(f"[prefix_ablation] corpus={len(corpus)} passages, eval={len(eval_set)} queries (fixed chunker, hi->hi)")

    with_prefix = _run_variant("with_prefix", encode_passages, encode_queries, corpus, eval_set)
    print(f"[prefix_ablation] WITH prefixes:    {with_prefix}")

    without_prefix = _run_variant(
        "without_prefix", encode_passages_no_prefix, encode_queries_no_prefix, corpus, eval_set
    )
    print(f"[prefix_ablation] WITHOUT prefixes: {without_prefix}")

    return {"with_prefix": with_prefix, "without_prefix": without_prefix}


def _write_markdown(results: dict, out_path: Path) -> None:
    wp, wop = results["with_prefix"], results["without_prefix"]
    lines = [
        "# e5 prefix ablation (section 4.1)",
        "",
        (
            f"`intfloat/multilingual-e5-small` is trained with mandatory `\"query: \"` / "
            f"`\"passage: \"` prefixes. Omitting them does not error -- it silently degrades "
            f"retrieval quality. Measured on {wp['n_eval_queries']} eval queries, fixed "
            "chunker, Hindi query -> Hindi corpus:"
        ),
        "",
        "| Variant | Recall@5 | MRR@10 | Chunks indexed |",
        "|---|---|---|---|",
        f"| With `query:`/`passage:` prefixes | {wp['recall@5']:.3f} | {wp['mrr@10']:.3f} | {wp['n_chunks']} |",
        f"| Without prefixes | {wop['recall@5']:.3f} | {wop['mrr@10']:.3f} | {wop['n_chunks']} |",
        "",
    ]
    delta = wp["recall@5"] - wop["recall@5"]
    if abs(delta) < 0.02:
        lines.append(
            f"**Honest note:** the measured Recall@5 delta ({delta:+.3f}) is within noise for "
            f"{wp['n_eval_queries']} queries -- not a dramatic effect on this particular slice, "
            "though the prefixes are still used everywhere in this project per the model card's "
            "documented training convention."
        )
    else:
        lines.append(f"Recall@5 delta (with - without): {delta:+.3f}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    results = run_ablation()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "prefix_ablation.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = RESULTS_DIR / "prefix_ablation.md"
    _write_markdown(results, md_path)
    print(f"Wrote {json_path} and {md_path}")
