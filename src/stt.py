"""Speech-to-text providers.

Callers: src/pipeline.py and src/harness.py obtain a provider via
get_stt_provider() and call .transcribe(audio_path) -> TranscriptionResult.
"""
from __future__ import annotations

import os
import wave
from abc import ABC, abstractmethod

import requests

from src import config
from src.contracts import TranscriptionResult
from src.harness import FatalError, TransientError


class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        ...


def _wav_duration_s(audio_path: str) -> float | None:
    """Best-effort, stdlib-only WAV duration lookup. Never raises."""
    try:
        with wave.open(audio_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except Exception:
        return None


class MockSTT(STTProvider):
    """Deterministic, offline STT stand-in. No network, no API key required.

    Always returns a fixed transcript so downstream pipeline stages and
    tests have something deterministic to work with, regardless of whether
    ``audio_path`` refers to a real file.
    """

    FIXED_TEXT = "कॉर्पोरेशन क्या है?"
    FIXED_LANGUAGE = "hin"

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        duration = _wav_duration_s(audio_path)
        return TranscriptionResult(
            text=self.FIXED_TEXT,
            language=self.FIXED_LANGUAGE,
            provider="mock",
            audio_duration_s=duration,
        )


class SarvamSTT(STTProvider):
    """Real STT via Sarvam AI's speech-to-text API (saaras:v3 model).

    Verified live against the real API on 2026-09-11: the model originally
    specified here (saarika:v2) is deprecated server-side and returns HTTP 400
    with an explicit "use saaras:v3 instead" message -- not a guess, the API's
    own error response named the replacement.
    """

    API_URL = "https://api.sarvam.ai/speech-to-text"

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        api_key = config.SARVAM_API_KEY
        if not api_key:
            raise FatalError(
                "SARVAM_API_KEY is not set. Get one at sarvam.ai and add it "
                "to .env, or set STT_PROVIDER=mock."
            )

        headers = {"api-subscription-key": api_key}
        data = {"model": "saaras:v3"}

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                response = requests.post(
                    self.API_URL,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=60,
                )
            response.raise_for_status()
        except FileNotFoundError as e:
            raise FatalError(f"Audio file not found: {audio_path}") from e
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            # 429 and 5xx are worth retrying; everything else (401, 400, 404,
            # ...) will not succeed on retry -- never retry those
            # (harness.py trap #10).
            if status_code == 429 or (status_code is not None and 500 <= status_code < 600):
                raise TransientError(f"Sarvam STT request failed: {e}", status_code=status_code) from e
            raise FatalError(f"Sarvam STT request failed: {e}", status_code=status_code) from e
        except requests.RequestException as e:
            # network-level issues with no HTTP status (timeout, connection
            # error) are worth retrying.
            raise TransientError(f"Sarvam STT request failed: {e}") from e

        try:
            payload = response.json()
        except ValueError as e:
            # HTTP 200 with a malformed/non-JSON body (truncated response, an
            # HTML error page, empty body under provider-side incident) is a
            # real possibility that raise_for_status() does not catch --
            # worth retrying rather than crashing uncaught.
            raise TransientError(f"Sarvam STT returned a non-JSON response: {e}") from e
        text = payload.get("transcript", "") or ""
        language = payload.get("language_code") or payload.get("language") or "unknown"

        return TranscriptionResult(
            text=text,
            language=language,
            provider="sarvam",
            audio_duration_s=_wav_duration_s(audio_path),
        )


def get_stt_provider() -> STTProvider:
    provider = config.STT_PROVIDER
    if provider == "mock":
        return MockSTT()
    if provider == "sarvam":
        return SarvamSTT()
    raise ValueError(
        f"Unknown STT_PROVIDER={provider!r}. Expected 'mock' or 'sarvam'."
    )
