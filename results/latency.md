# Latency Benchmark — P50/P70/P100 per stage

Strategy: `fixed`. Percentiles via `numpy.percentile(..., method="nearest")`. Cold start (first query: model load + index build) is reported separately and excluded from the warm P50/P70/P100 pool below.

## mock

STT provider: `mock`  |  LLM provider: `mock`  |  warm queries: 29  |  cold start: 7343.0ms

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---|---|---|
| **end-to-end** | 31.9 | 35.1 | 43.8 |
| stt | 0.0 | 0.0 | 0.0 |
| validate | 0.0 | 0.0 | 0.0 |
| retrieve | 8.1 | 9.0 | 10.5 |
| guardrail_pre | 0.0 | 0.0 | 0.0 |
| generate | 0.0 | 0.0 | 0.0 |
| guardrail_post | 23.7 | 25.8 | 33.3 |

## real_llm_mock_stt

STT provider: `mock`  |  LLM provider: `groq`  |  warm queries: 29  |  cold start: 1488.0ms

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---|---|---|
| **end-to-end** | 5355.8 | 7967.6 | 9009.4 |
| stt | 0.0 | 0.0 | 0.1 |
| validate | 0.0 | 0.0 | 0.0 |
| retrieve | 12.2 | 12.5 | 16.2 |
| guardrail_pre | 0.0 | 0.0 | 0.0 |
| generate | 5251.4 | 7849.0 | 8915.9 |
| guardrail_post | 87.7 | 94.3 | 113.1 |

## real_stt

**Not measured.** SARVAM_API_KEY and a real recorded .wav sample are both required and neither is available in this environment -- not measured, not estimated.

## The 200ms target — honest analysis

Measured on this machine: query embedding + vector retrieval (`retrieve` stage above) lands at P50=8.1ms, P100=10.5ms over a ~10k-chunk corpus. This is the genuinely sub-200ms part of the pipeline, and the part the 200ms target most plausibly refers to.

The full voice-to-answer pipeline does **not** hit 200ms end-to-end, even in mock mode, once generation and guardrail stages are included -- and it cannot, in real-API mode, once network-bound STT (300-1500ms) and LLM decode (400-3000ms) are added on top of what's measured here. See VOICE_RAG_BUILD_PLAN.md section 3 for the stage-by-stage floor analysis this was measured against.

**What it would take to get closer to 200ms end-to-end with real STT+LLM:** a local quantized STT model (e.g. whisper.cpp tiny/base) instead of a network round trip; a small local generator instead of a hosted API; a streaming-first design that measures time-to-first-token rather than time-to-full-answer; and aggressive caching for repeated queries. These are estimates, not measurements, and are stated as such.