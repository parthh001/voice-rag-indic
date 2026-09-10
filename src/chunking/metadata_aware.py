from __future__ import annotations

import re

from src.chunking.base import Chunker
from src.contracts import Chunk
from src.text_utils import split_sentences

_DIGIT_RE = re.compile(r"[0-9०-९]")

_TIGHT_QUERY_TYPES = {"NUMERIC", "ENTITY", "PERSON", "LOCATION"}


class MetadataAwareChunker(Chunker):
    """Chunker that adapts chunk size to `query_type` and carries rich,
    downstream-usable metadata (e.g. `contains_digit` for NUMERIC-query
    retrieval biasing) alongside each chunk.
    """

    name = "metadata_aware"

    def __init__(self, base_chunk_sentences: int = 6, tight_chunk_sentences: int = 3) -> None:
        self.base_chunk_sentences = base_chunk_sentences
        self.tight_chunk_sentences = tight_chunk_sentences

    def chunk(self, text: str, doc_id: str, metadata: dict) -> list[Chunk]:
        if not text or not text.strip():
            return []

        sentences = split_sentences(text)
        if not sentences:
            return []

        query_type = metadata.get("query_type")
        window_size = (
            self.tight_chunk_sentences
            if query_type in _TIGHT_QUERY_TYPES
            else self.base_chunk_sentences
        )

        # Sentence-aligned, non-overlapping windows: simplest way to guarantee
        # we never split a sentence, and adjacent chunks already share the
        # surrounding document context via retrieval, so overlap isn't needed.
        chunks: list[Chunk] = []
        for i, start in enumerate(range(0, len(sentences), window_size)):
            window = sentences[start : start + window_size]
            chunk_text = " ".join(window)

            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_position"] = i
            chunk_metadata["contains_digit"] = bool(_DIGIT_RE.search(chunk_text))
            chunk_metadata["n_sentences"] = len(window)

            chunks.append(
                Chunk(
                    chunk_id=f"metadata_aware:{doc_id}:{i}",
                    text=chunk_text,
                    doc_id=doc_id,
                    strategy=self.name,
                    metadata=chunk_metadata,
                )
            )

        return chunks
