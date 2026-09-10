from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator

ChunkStrategy = Literal["fixed", "semantic", "metadata_aware"]

class Chunk(BaseModel):
    chunk_id: str
    text: str = Field(min_length=1)
    doc_id: str                      # source passage id, e.g. "q1185869_p5"
    strategy: ChunkStrategy
    metadata: dict = Field(default_factory=dict)   # query_type, lang, position, ...

class TranscriptionResult(BaseModel):
    text: str
    language: str
    provider: Literal["sarvam", "mock"]
    audio_duration_s: float | None = None

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float                     # cosine similarity, higher is better

class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    strategy: ChunkStrategy

class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    score: float | None = None

class GuardrailVerdict(BaseModel):
    passed: bool
    checks: list[CheckResult]
    @property
    def failed_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

class Answer(BaseModel):
    text: str
    citations: list[str]             # chunk_ids actually used
    grounded: bool
    grounding_score: float | None = None

class StageTiming(BaseModel):
    stage: str
    ms: float

class PipelineResult(BaseModel):
    transcript: TranscriptionResult | None
    retrieval: RetrievalResult
    guardrails: GuardrailVerdict
    answer: Answer | None            # None when guardrails reject
    timings: list[StageTiming]
    @property
    def total_ms(self) -> float:
        return sum(t.ms for t in self.timings)
