from __future__ import annotations

from src.chunking.base import Chunker
from src.contracts import Chunk


class FixedChunker(Chunker):
    """Honest fixed-size baseline: whitespace-word windows with overlap."""

    name = "fixed"

    def __init__(self, chunk_size: int = 256, overlap_ratio: float = 0.15) -> None:
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio

    def chunk(self, text: str, doc_id: str, metadata: dict) -> list[Chunk]:
        # Whitespace-split "words" are used as a cheap, language-agnostic
        # approximation of tokens (avoids pulling in a tokenizer dependency).
        words = text.split()
        if not words:
            return []

        step = max(1, int(self.chunk_size * (1 - self.overlap_ratio)))

        chunks: list[Chunk] = []
        i = 0
        start = 0
        while start < len(words):
            window = words[start : start + self.chunk_size]
            chunk_text = " ".join(window)
            # "chunk_position" (this chunk's index within the doc), not
            # "position" -- the caller's own "position" (the passage's rank
            # among its query's candidates, a real signal per the build plan)
            # must survive untouched, not be clobbered by the chunk loop index.
            chunk_metadata = {**metadata, "chunk_position": i}
            chunks.append(
                Chunk(
                    chunk_id=f"fixed:{doc_id}:{i}",
                    text=chunk_text,
                    doc_id=doc_id,
                    strategy=self.name,
                    metadata=chunk_metadata,
                )
            )
            i += 1
            if start + self.chunk_size >= len(words):
                break
            start += step

        return chunks
