"""Orchestration harness: typed stages, retry with exponential backoff on transient
errors only, structured typed exceptions, and per-stage timing recorded even for
failed attempts. Never retry a 401 or a validation error (section 7 Phase 7,
trap #10 in the plan's trap table) -- that burns rate limit and hides the real error.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from src.contracts import StageTiming

T = TypeVar("T")


class PipelineError(Exception):
    """Base class for all harness-raised errors."""


class TransientError(PipelineError):
    """Retryable: timeouts, HTTP 429, HTTP 5xx."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FatalError(PipelineError):
    """Not retryable: HTTP 401, validation errors, and anything else that will not
    succeed on retry."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class MaxRetriesExceededError(PipelineError):
    def __init__(self, stage: str, attempts: int, last_error: Exception) -> None:
        super().__init__(f"stage {stage!r} failed after {attempts} attempt(s): {last_error}")
        self.stage = stage
        self.attempts = attempts
        self.last_error = last_error


def run_stage(
    stage_name: str,
    fn: Callable[[], T],
    timings: list[StageTiming],
    max_attempts: int = 3,
    base_delay_s: float = 0.01,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Run `fn`, recording a StageTiming for every attempt (including failures).
    Retries only on TransientError, with exponential backoff (base_delay_s * 2**n).
    A FatalError is raised immediately, no retry. After `max_attempts` transient
    failures, raises MaxRetriesExceededError."""
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        t0 = time.monotonic()
        try:
            result = fn()
        except FatalError:
            elapsed_ms = (time.monotonic() - t0) * 1000
            timings.append(StageTiming(stage=f"{stage_name}:attempt{attempt}:fatal", ms=elapsed_ms))
            raise
        except TransientError as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            timings.append(
                StageTiming(stage=f"{stage_name}:attempt{attempt}:transient", ms=elapsed_ms)
            )
            last_error = e
            if attempt < max_attempts:
                sleep_fn(base_delay_s * (2 ** (attempt - 1)))
            continue
        else:
            elapsed_ms = (time.monotonic() - t0) * 1000
            timings.append(StageTiming(stage=stage_name, ms=elapsed_ms))
            return result

    assert last_error is not None
    raise MaxRetriesExceededError(stage_name, max_attempts, last_error)
