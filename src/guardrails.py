"""Four guardrail checks: off-topic, inappropriate input, grounding verification,
hallucination detection. Cheap checks (keyword match, retrieval-score threshold) run
before any embedding/LLM work -- both faster and more defensible per the plan's own
guidance (section 7 Phase 6).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.contracts import Answer, CheckResult, GuardrailVerdict, RetrievedChunk
from src.embeddings import encode_passages
from src.text_utils import split_sentences

OFF_TOPIC_SIM_THRESHOLD = 0.75
HALLUCINATION_SIM_THRESHOLD = 0.75

# Heuristic keyword blocklist for the "inappropriate input" check. This is a coarse,
# disclosed limitation (see README Known Limitations) -- a real deployment would use
# a dedicated moderation model, not substring matching.
_UNSAFE_PATTERNS = [
    "kill yourself",
    "how to make a bomb",
    "how to build a bomb",
    "child sexual abuse",
    "how to synthesize meth",
    "how to make meth",
    "commit suicide",
    "self harm",
    "self-harm",
]


def check_inappropriate(query: str) -> CheckResult:
    lowered = query.lower()
    for pattern in _UNSAFE_PATTERNS:
        if pattern in lowered:
            return CheckResult(
                name="inappropriate_input",
                passed=False,
                detail=f"matched unsafe pattern: {pattern!r}",
                score=0.0,
            )
    return CheckResult(
        name="inappropriate_input", passed=True, detail="no unsafe pattern matched", score=1.0
    )


def check_off_topic(
    query: str,
    retrieved: list[RetrievedChunk],
    threshold: float = OFF_TOPIC_SIM_THRESHOLD,
    llm_judge: Callable[[str], bool] | None = None,
) -> CheckResult:
    """Cheap check first: if the corpus's best match for this query is weak, the
    corpus likely cannot answer it. Only escalate to an LLM judge (if one is
    supplied) when the cheap check is ambiguous/negative -- keeps this check
    testable and fast with no API key required in the default path.
    """
    max_score = max((rc.score for rc in retrieved), default=0.0)
    if max_score >= threshold:
        return CheckResult(
            name="off_topic",
            passed=True,
            detail=f"max retrieval score {max_score:.3f} >= {threshold}",
            score=max_score,
        )
    if llm_judge is not None:
        is_on_topic = llm_judge(query)
        return CheckResult(
            name="off_topic",
            passed=is_on_topic,
            detail=f"retrieval score {max_score:.3f} below threshold, escalated to LLM judge",
            score=max_score,
        )
    return CheckResult(
        name="off_topic",
        passed=False,
        detail=f"max retrieval score {max_score:.3f} < {threshold}, no LLM judge configured",
        score=max_score,
    )


def check_grounding(answer: Answer, retrieved: list[RetrievedChunk]) -> CheckResult:
    """Every citation the answer makes must exist in the retrieved set. An answer
    with zero citations trivially passes this check (nothing to invent) -- it is
    the hallucination check below that catches ungrounded prose."""
    retrieved_ids = {rc.chunk.chunk_id for rc in retrieved}
    invented = [c for c in answer.citations if c not in retrieved_ids]
    passed = len(invented) == 0
    score = 1.0 - (len(invented) / len(answer.citations)) if answer.citations else 1.0
    detail = (
        f"invented citations not in retrieved set: {invented}"
        if invented
        else "all citations verified against retrieved set"
    )
    return CheckResult(name="grounding_verification", passed=passed, detail=detail, score=score)


def check_hallucination(
    answer: Answer, retrieved: list[RetrievedChunk], threshold: float = HALLUCINATION_SIM_THRESHOLD
) -> CheckResult:
    """Sentence-level: embed each answer sentence, compare against the retrieved
    chunk embeddings, flag sentences whose best match falls below threshold."""
    sentences = split_sentences(answer.text)
    if not sentences:
        return CheckResult(
            name="hallucination_detection", passed=True, detail="no sentences to check", score=1.0
        )
    if not retrieved:
        return CheckResult(
            name="hallucination_detection",
            passed=False,
            detail="answer has content but nothing was retrieved to ground it in",
            score=0.0,
        )

    sentence_vecs = encode_passages(sentences)
    chunk_texts = [rc.chunk.text for rc in retrieved]
    chunk_vecs = encode_passages(chunk_texts)

    sims: list[float] = []
    flagged: list[str] = []
    for sent, svec in zip(sentences, sentence_vecs):
        sim = float(np.max(chunk_vecs @ svec))
        sims.append(sim)
        if sim < threshold:
            flagged.append(sent)

    grounding_score = float(np.mean(sims))
    passed = len(flagged) == 0
    detail = (
        f"{len(flagged)}/{len(sentences)} sentence(s) below similarity {threshold}: {flagged}"
        if flagged
        else f"all {len(sentences)} sentence(s) grounded (mean sim {grounding_score:.3f})"
    )
    return CheckResult(
        name="hallucination_detection", passed=passed, detail=detail, score=grounding_score
    )


def run_guardrails(
    query: str,
    retrieved: list[RetrievedChunk],
    answer: Answer | None,
    off_topic_threshold: float = OFF_TOPIC_SIM_THRESHOLD,
    hallucination_threshold: float = HALLUCINATION_SIM_THRESHOLD,
    llm_judge: Callable[[str], bool] | None = None,
) -> GuardrailVerdict:
    checks = [
        check_inappropriate(query),
        check_off_topic(query, retrieved, threshold=off_topic_threshold, llm_judge=llm_judge),
    ]
    if answer is not None:
        checks.append(check_grounding(answer, retrieved))
        checks.append(check_hallucination(answer, retrieved, threshold=hallucination_threshold))

    passed = all(c.passed for c in checks)
    return GuardrailVerdict(passed=passed, checks=checks)
