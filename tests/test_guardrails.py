from src.contracts import Answer, Chunk, RetrievedChunk
from src.guardrails import (
    check_grounding,
    check_hallucination,
    check_inappropriate,
    check_off_topic,
    run_guardrails,
)


def _rc(chunk_id: str, text: str, doc_id: str = "doc1", score: float = 0.9) -> RetrievedChunk:
    chunk = Chunk(chunk_id=chunk_id, text=text, doc_id=doc_id, strategy="fixed", metadata={})
    return RetrievedChunk(chunk=chunk, score=score)


def test_check_inappropriate_flags_unsafe_query():
    result = check_inappropriate("How to make a bomb at home?")
    assert result.passed is False
    assert result.name == "inappropriate_input"


def test_check_inappropriate_passes_safe_query():
    result = check_inappropriate("What is a corporation?")
    assert result.passed is True


def test_check_off_topic_passes_when_retrieval_score_high():
    retrieved = [_rc("c1", "A corporation is a legal entity.", score=0.95)]
    result = check_off_topic("What is a corporation?", retrieved, threshold=0.75)
    assert result.passed is True


def test_check_off_topic_fails_when_retrieval_score_low_and_no_judge():
    retrieved = [_rc("c1", "unrelated text", score=0.2)]
    result = check_off_topic("random unrelated query", retrieved, threshold=0.75)
    assert result.passed is False


def test_check_off_topic_escalates_to_llm_judge_when_provided():
    retrieved = [_rc("c1", "unrelated text", score=0.2)]
    result = check_off_topic(
        "borderline query", retrieved, threshold=0.75, llm_judge=lambda q: True
    )
    assert result.passed is True
    assert "LLM judge" in result.detail


def test_check_grounding_passes_with_valid_citations():
    retrieved = [_rc("c1", "text one"), _rc("c2", "text two")]
    answer = Answer(text="An answer.", citations=["c1"], grounded=True, grounding_score=1.0)
    result = check_grounding(answer, retrieved)
    assert result.passed is True


def test_check_grounding_fails_with_invented_citation():
    retrieved = [_rc("c1", "text one")]
    answer = Answer(text="An answer.", citations=["c1", "c999"], grounded=True, grounding_score=None)
    result = check_grounding(answer, retrieved)
    assert result.passed is False
    assert "c999" in result.detail


def test_check_grounding_passes_with_no_citations():
    result = check_grounding(
        Answer(text="I don't know.", citations=[], grounded=False, grounding_score=0.0), []
    )
    assert result.passed is True


def test_check_hallucination_passes_grounded_answer():
    retrieved = [_rc("c1", "A corporation is a legal entity separate from its owners.")]
    answer = Answer(
        text="A corporation is a legal entity separate from its owners.",
        citations=["c1"],
        grounded=True,
        grounding_score=None,
    )
    result = check_hallucination(answer, retrieved, threshold=0.7)
    assert result.passed is True
    assert result.score is not None and result.score >= 0.7


def test_check_hallucination_flags_unrelated_sentence():
    retrieved = [_rc("c1", "A corporation is a legal entity separate from its owners.")]
    answer = Answer(
        text="The moon landing happened in 1969 and involved astronauts on a rocket.",
        citations=["c1"],
        grounded=True,
        grounding_score=None,
    )
    result = check_hallucination(answer, retrieved, threshold=0.75)
    assert result.passed is False


def test_run_guardrails_rejects_unsafe_query_before_generation():
    verdict = run_guardrails("How to make a bomb at home?", [], answer=None)
    assert verdict.passed is False
    assert "inappropriate_input" in verdict.failed_checks


def test_run_guardrails_all_pass_end_to_end():
    retrieved = [_rc("c1", "A corporation is a legal entity separate from its owners.", score=0.95)]
    answer = Answer(
        text="A corporation is a legal entity separate from its owners.",
        citations=["c1"],
        grounded=True,
        grounding_score=None,
    )
    verdict = run_guardrails("What is a corporation?", retrieved, answer=answer)
    assert verdict.passed is True
    assert len(verdict.checks) == 4
