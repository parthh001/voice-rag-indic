from src.contracts import GuardrailVerdict, CheckResult
from src.pipeline import _validate_query_text, run_pipeline


def test_validate_rejects_empty_string():
    result = _validate_query_text("")
    assert result.passed is False
    assert result.name == "input_validation"


def test_validate_rejects_whitespace_only():
    result = _validate_query_text("   \n\t  ")
    assert result.passed is False


def test_validate_rejects_none():
    result = _validate_query_text(None)
    assert result.passed is False


def test_validate_passes_real_query():
    result = _validate_query_text("कॉर्पोरेशन क्या है?")
    assert result.passed is True


def test_run_pipeline_nulls_answer_when_post_guardrail_fails(monkeypatch):
    """Regression test: a failed post-generation guardrail verdict (invented
    citation, ungrounded sentence) must null the returned answer, matching
    the contract's own invariant ("answer: Answer | None  # None when
    guardrails reject"). Previously the pipeline returned the rejected
    answer anyway, silently defeating the guardrail."""
    passing_verdict = GuardrailVerdict(passed=True, checks=[])
    failing_verdict = GuardrailVerdict(
        passed=False,
        checks=[CheckResult(name="hallucination_detection", passed=False, detail="forced failure", score=0.1)],
    )
    # run_guardrails is called twice: once pre-generation (must pass so the
    # pipeline actually reaches generation) and once post-generation (forced
    # to fail here, to isolate the bug this test targets).
    calls = {"n": 0}

    def fake_run_guardrails(*args, **kwargs):
        calls["n"] += 1
        return passing_verdict if calls["n"] == 1 else failing_verdict

    monkeypatch.setattr("src.pipeline.run_guardrails", fake_run_guardrails)

    result = run_pipeline(query_text="कॉर्पोरेशन क्या है?", strategy="fixed")

    assert result.answer is None
    assert result.guardrails.passed is False
