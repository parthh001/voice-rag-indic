import pytest

from src.contracts import StageTiming
from src.harness import FatalError, MaxRetriesExceededError, TransientError, run_stage


class FlakyProvider:
    """Fails with a given exception N times, then succeeds."""

    def __init__(self, fail_times: int, exc_factory):
        self.fail_times = fail_times
        self.exc_factory = exc_factory
        self.calls = 0

    def call(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc_factory()
        return "ok"


def test_retries_on_429_then_succeeds():
    provider = FlakyProvider(fail_times=2, exc_factory=lambda: TransientError("rate limited", status_code=429))
    timings: list[StageTiming] = []

    result = run_stage("stt", provider.call, timings, max_attempts=3, sleep_fn=lambda s: None)

    assert result == "ok"
    assert provider.calls == 3
    # 2 failed attempts + 1 successful attempt recorded
    assert len(timings) == 3
    assert timings[0].stage == "stt:attempt1:transient"
    assert timings[1].stage == "stt:attempt2:transient"
    assert timings[2].stage == "stt"


def test_does_not_retry_on_401():
    provider = FlakyProvider(fail_times=5, exc_factory=lambda: FatalError("unauthorized", status_code=401))
    timings: list[StageTiming] = []

    with pytest.raises(FatalError):
        run_stage("stt", provider.call, timings, max_attempts=3, sleep_fn=lambda s: None)

    assert provider.calls == 1, "a 401 must never be retried"
    assert len(timings) == 1
    assert timings[0].stage == "stt:attempt1:fatal"


def test_surfaces_typed_error_after_max_attempts():
    provider = FlakyProvider(fail_times=10, exc_factory=lambda: TransientError("timeout"))
    timings: list[StageTiming] = []

    with pytest.raises(MaxRetriesExceededError) as exc_info:
        run_stage("retrieve", provider.call, timings, max_attempts=3, sleep_fn=lambda s: None)

    assert provider.calls == 3
    assert exc_info.value.stage == "retrieve"
    assert exc_info.value.attempts == 3
    assert isinstance(exc_info.value.last_error, TransientError)


def test_timings_recorded_even_when_stage_ultimately_fails():
    provider = FlakyProvider(fail_times=10, exc_factory=lambda: TransientError("timeout"))
    timings: list[StageTiming] = []

    with pytest.raises(MaxRetriesExceededError):
        run_stage("generate", provider.call, timings, max_attempts=3, sleep_fn=lambda s: None)

    assert len(timings) == 3
    assert all(t.ms >= 0 for t in timings)
    assert all(t.stage.startswith("generate:") for t in timings)


def test_backoff_delay_grows_exponentially():
    provider = FlakyProvider(fail_times=3, exc_factory=lambda: TransientError("timeout"))
    timings: list[StageTiming] = []
    delays: list[float] = []

    run_stage(
        "embed",
        provider.call,
        timings,
        max_attempts=4,
        base_delay_s=0.1,
        sleep_fn=lambda s: delays.append(s),
    )

    assert delays == [0.1, 0.2, 0.4]
