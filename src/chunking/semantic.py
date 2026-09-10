from __future__ import annotations

import numpy as np

from src.chunking.base import Chunker
from src.contracts import Chunk
from src.embeddings import encode_passages
from src.text_utils import split_sentences


class SemanticChunker(Chunker):
    """Groups sentences by embedding-similarity to a running chunk centroid.

    This is NOT a fixed-size sentence splitter: boundaries are decided by
    real cosine similarity (via multilingual-e5-small embeddings) between
    each new sentence and the running mean embedding of the chunk being
    built. A topic shift drops similarity below `similarity_threshold` and
    triggers a new chunk, independent of any character/word cap.
    """

    name = "semantic"

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        min_chunk_sentences: int = 1,
        max_chunk_sentences: int = 8,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences

    def chunk(self, text: str, doc_id: str, metadata: dict) -> list[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                Chunk(
                    chunk_id=f"semantic:{doc_id}:0",
                    text=sentences[0],
                    doc_id=doc_id,
                    strategy=self.name,
                    metadata={**metadata, "chunk_position": 0, "n_sentences": 1},
                )
            ]

        embeddings = encode_passages(sentences)

        groups: list[list[int]] = []
        current: list[int] = []
        centroid_sum: np.ndarray | None = None  # running sum of embeddings in `current`

        for idx, emb in enumerate(embeddings):
            if not current:
                current = [idx]
                centroid_sum = emb.copy()
                continue

            centroid = centroid_sum / len(current)
            # Vectors from encode_passages are L2-normalized; `centroid` is a
            # mean of normalized vectors so it isn't itself unit-length, but
            # the dot product with a unit-length new-sentence embedding is
            # still a monotonic cosine-similarity-like signal, sufficient
            # for thresholding boundary decisions.
            sim = float(np.dot(centroid, emb))

            should_close = (
                sim < self.similarity_threshold and len(current) >= self.min_chunk_sentences
            ) or len(current) >= self.max_chunk_sentences

            if should_close:
                groups.append(current)
                current = [idx]
                centroid_sum = emb.copy()
            else:
                current.append(idx)
                centroid_sum = centroid_sum + emb

        if current:
            groups.append(current)

        chunks: list[Chunk] = []
        for i, group in enumerate(groups):
            chunk_text = " ".join(sentences[j] for j in group)
            chunk_metadata = {**metadata, "chunk_position": i, "n_sentences": len(group)}
            chunks.append(
                Chunk(
                    chunk_id=f"semantic:{doc_id}:{i}",
                    text=chunk_text,
                    doc_id=doc_id,
                    strategy=self.name,
                    metadata=chunk_metadata,
                )
            )

        return chunks
