"""CLI entrypoint: voice or text query -> STT -> retrieve -> guardrail(pre) ->
generate -> guardrail(post, incl. grounding) -> answer. Wires the harness's typed
retry/timing around each external-facing stage (STT, generation); retrieval and
guardrails are local/deterministic and are timed but not retried.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking.base import Chunker
from src.chunking.fixed import FixedChunker
from src.chunking.metadata_aware import MetadataAwareChunker
from src.chunking.semantic import SemanticChunker
from src.config import SUBSET_DIR
from src.contracts import (
    CheckResult,
    GuardrailVerdict,
    PipelineResult,
    RetrievalResult,
    StageTiming,
    TranscriptionResult,
)
from src.generation import get_llm_provider
from src.guardrails import run_guardrails
from src.harness import FatalError, run_stage
from src.stt import get_stt_provider
from src.store import VectorStore

DEFAULT_STRATEGY = "fixed"

_CHUNKER_REGISTRY: dict[str, type[Chunker]] = {
    "fixed": FixedChunker,
    "semantic": SemanticChunker,
    "metadata_aware": MetadataAwareChunker,
}


def _ensure_pipeline_index(strategy: str) -> VectorStore:
    """Persistent per-strategy index over the Hindi-language corpus subset. Built
    once (lazily, on first use) and reused across runs via Chroma's persistence --
    only the first call after a fresh clone pays the indexing cost."""
    store = VectorStore(f"pipeline_{strategy}_hi")
    if store.count() > 0:
        return store

    corpus_path = SUBSET_DIR / "corpus.jsonl"
    if not corpus_path.exists():
        raise FatalError(
            f"{corpus_path} not found. Run `python -m src.data.loader --lang hin --n 1000` first."
        )

    chunker = _CHUNKER_REGISTRY[strategy]()
    chunks = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["text_hi"]
            if not text or not text.strip():
                continue
            metadata = {"query_type": rec["query_type"], "position": rec["position"], "lang": "hi"}
            chunks.extend(chunker.chunk(text, doc_id=rec["doc_id"], metadata=metadata))
    store.index(chunks)
    return store


def _validate_query_text(query_text: str | None) -> CheckResult:
    """The architecture diagram's 'validate' stage: catches an empty/whitespace
    query before it ever reaches retrieval or generation -- notably, the case
    where a real STT call succeeds but transcribes non-speech audio (silence,
    noise, a test tone) into an empty string. Found via live testing against
    the real Sarvam API with a synthetic test tone: without this check, an
    empty transcript silently proceeded through retrieval and generation and
    produced a nonsense answer instead of a clear rejection."""
    if query_text is None or not query_text.strip():
        return CheckResult(
            name="input_validation",
            passed=False,
            detail="query text is empty (STT may have transcribed non-speech audio, "
            "or an empty query was provided directly)",
        )
    return CheckResult(name="input_validation", passed=True, detail="query text is non-empty")


def run_pipeline(
    query_text: str | None = None,
    audio_path: str | None = None,
    strategy: str = DEFAULT_STRATEGY,
    k: int = 5,
) -> PipelineResult:
    if query_text is None and audio_path is None:
        raise ValueError("either query_text or audio_path must be provided")
    if strategy not in _CHUNKER_REGISTRY:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {list(_CHUNKER_REGISTRY)}")

    timings: list[StageTiming] = []
    transcript: TranscriptionResult | None = None

    if audio_path is not None:
        stt = get_stt_provider()
        # Real providers (SarvamSTT) raise harness.TransientError/FatalError
        # directly, already classified by status code -- MockSTT never raises.
        transcript = run_stage("stt", lambda: stt.transcribe(audio_path), timings)
        query_text = transcript.text

    t0 = time.monotonic()
    validation = _validate_query_text(query_text)
    timings.append(StageTiming(stage="validate", ms=(time.monotonic() - t0) * 1000))

    if not validation.passed:
        return PipelineResult(
            transcript=transcript,
            retrieval=RetrievalResult(query=query_text or "", chunks=[], strategy=strategy),
            guardrails=GuardrailVerdict(passed=False, checks=[validation]),
            answer=None,
            timings=timings,
        )

    store = _ensure_pipeline_index(strategy)

    def _retrieve_call():
        return store.search(query_text, k=k)

    retrieval = run_stage("retrieve", _retrieve_call, timings)

    t0 = time.monotonic()
    pre_verdict = run_guardrails(query_text, retrieval.chunks, answer=None)
    timings.append(StageTiming(stage="guardrail_pre", ms=(time.monotonic() - t0) * 1000))

    if not pre_verdict.passed:
        return PipelineResult(
            transcript=transcript,
            retrieval=retrieval,
            guardrails=pre_verdict,
            answer=None,
            timings=timings,
        )

    llm = get_llm_provider()
    # Real providers (GroqLLM) raise harness.TransientError/FatalError
    # directly, already classified by status code -- MockLLM never raises.
    answer = run_stage("generate", lambda: llm.generate(query_text, retrieval.chunks), timings)

    t0 = time.monotonic()
    post_verdict = run_guardrails(query_text, retrieval.chunks, answer=answer)
    timings.append(StageTiming(stage="guardrail_post", ms=(time.monotonic() - t0) * 1000))

    if not post_verdict.passed:
        # A failed post-generation verdict (invented citation, ungrounded
        # sentence) must null the answer, matching the contract's own
        # invariant ("answer: Answer | None  # None when guardrails reject")
        # -- returning the rejected answer anyway would defeat the guardrail.
        return PipelineResult(
            transcript=transcript,
            retrieval=retrieval,
            guardrails=post_verdict,
            answer=None,
            timings=timings,
        )

    # Providers hardcode grounded/grounding_score as placeholders (see
    # src/generation.py) -- overwrite with the real value the hallucination
    # check just computed, so a caller reading Answer.grounding_score sees
    # the actual measured score, not a stale placeholder.
    hallucination_check = next(
        (c for c in post_verdict.checks if c.name == "hallucination_detection"), None
    )
    if hallucination_check is not None and hallucination_check.score is not None:
        answer.grounding_score = hallucination_check.score
    answer.grounded = True

    return PipelineResult(
        transcript=transcript,
        retrieval=retrieval,
        guardrails=post_verdict,
        answer=answer,
        timings=timings,
    )


def _print_result(result: PipelineResult) -> None:
    if result.transcript is not None:
        print(f"Transcript ({result.transcript.provider}): {result.transcript.text}")
    print(f"Query: {result.retrieval.query}")
    print(f"Strategy: {result.retrieval.strategy}")
    if result.retrieval.chunks:
        print(
            f"Retrieved {len(result.retrieval.chunks)} chunks "
            f"(top score: {result.retrieval.chunks[0].score:.3f})"
        )
    else:
        print("Retrieved 0 chunks")
    print()
    print(f"Guardrails passed: {result.guardrails.passed}")
    for c in result.guardrails.checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"  [{status}] {c.name}: {c.detail}")
    print()
    if result.answer is not None:
        print(f"Answer: {result.answer.text}")
        print(f"Citations: {result.answer.citations}")
        print(f"Grounded: {result.answer.grounded} (score={result.answer.grounding_score})")
    else:
        print("Answer: REJECTED by guardrails -- no answer generated.")
    print()
    print(f"Total latency: {result.total_ms:.1f}ms")
    for t in result.timings:
        print(f"  {t.stage}: {t.ms:.1f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="text query (skips STT)")
    group.add_argument("--audio", type=str, help="path to a .wav file")
    parser.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY, choices=list(_CHUNKER_REGISTRY))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    result = run_pipeline(query_text=args.query, audio_path=args.audio, strategy=args.strategy, k=args.k)
    _print_result(result)
