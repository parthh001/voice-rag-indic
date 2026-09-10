"""Chroma-backed vector store: one persistent collection per (strategy, lang, corpus_mode).

Chroma's cosine space stores distance = 1 - cosine_similarity, so
score = 1 - distance recovers the cosine similarity the RetrievedChunk contract expects.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

from src.config import CHROMA_PATH
from src.contracts import Chunk, RetrievalResult, RetrievedChunk
from src.embeddings import encode_passages, encode_queries

_SCALAR_TYPES = (str, int, float, bool)


class VectorStore:
    def __init__(self, collection_name: str, persist_path: str | Path = CHROMA_PATH, reset: bool = False):
        self.client = chromadb.PersistentClient(path=str(persist_path))
        if reset:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _flatten_metadata(chunk: Chunk) -> dict:
        meta: dict = {"doc_id": chunk.doc_id, "strategy": chunk.strategy}
        for k, v in chunk.metadata.items():
            meta[k] = v if isinstance(v, _SCALAR_TYPES) else json.dumps(v, ensure_ascii=False)
        return meta

    def index(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        if not chunks:
            return
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = encode_passages(texts)
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings.tolist(),
                metadatas=[self._flatten_metadata(c) for c in batch],
                documents=texts,
            )

    def count(self) -> int:
        return self.collection.count()

    def search(self, query: str, k: int = 5, filters: dict | None = None) -> RetrievalResult:
        query_emb = encode_queries([query])[0]
        kwargs = {"where": filters} if filters else {}
        result = self.collection.query(
            query_embeddings=[query_emb.tolist()], n_results=k, **kwargs
        )

        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        retrieved: list[RetrievedChunk] = []
        strategy = "fixed"
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            strategy = meta["strategy"]
            chunk_metadata = {k: v for k, v in meta.items() if k not in ("doc_id", "strategy")}
            chunk = Chunk(
                chunk_id=chunk_id,
                text=doc,
                doc_id=meta["doc_id"],
                strategy=strategy,
                metadata=chunk_metadata,
            )
            score = 1.0 - dist
            retrieved.append(RetrievedChunk(chunk=chunk, score=score))

        return RetrievalResult(query=query, chunks=retrieved, strategy=strategy)
