from __future__ import annotations

from abc import ABC, abstractmethod

from src.contracts import Chunk, ChunkStrategy


class Chunker(ABC):
    name: ChunkStrategy

    @abstractmethod
    def chunk(self, text: str, doc_id: str, metadata: dict) -> list[Chunk]: ...
