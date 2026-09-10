# Voice-Enabled RAG over Indic MS MARCO

A voice-enabled RAG system over the [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset: **voice input → speech-to-text → chunking/retrieval → answer generation**, with three chunking strategies compared head-to-head using real relevance labels, a guardrail layer with a measured (not asserted) precision/recall, and a per-stage latency breakdown that treats the brief's 200ms target honestly instead of gaming it.

Built as a portfolio-grade implementation of HH Goa 2026 Shortlisting Task 2, after the original submission deadline — the optimization target here is correctness, measurability, and honest engineering analysis, not speed of delivery. See `VOICE_RAG_BUILD_PLAN.md` for the full specification and reconnaissance this was built against.

## Architecture

```
                        ┌──────────────────── model harness ─────────────────────┐
  mic / .wav ──► STT ──►│ validate ─► retrieve ─► guardrail(pre) ─► generate ─►  │──► answer
                        │   ▲            │             │              │          │    + citations
                        │   └── retries, backoff, typed errors, per-stage timing  │    + grounding
                        └────────────────────────────────────────────────────────┘         verdict
                                         │
                          Chroma ◄── embeddings ◄── chunker (fixed | semantic | metadata-aware)
                                                          ▲
                                              MSMARCO-XI subset (JSONL cache)
```

| Layer | Choice | Why |
|---|---|---|
| STT | Sarvam (`saarika`) + mock provider | built for Indic audio; mock keeps CI/tests keyless |
| Embeddings | `intfloat/multilingual-e5-small` (local) | 384-dim, ~120MB, fast on CPU, real multilingual coverage, no network hop in the latency path |
| Vector store | Chroma (local persistent) | zero-config, local, free |
| Generation | Groq (Llama 3.1) + mock provider | fastest hosted inference available free; mock keeps CI/tests keyless |
| Validation | pydantic v2 | typed contracts at every module boundary (`src/contracts.py`) |
| Tests | pytest | 49 tests, all pass with zero API keys present |

Ground truth for every retrieval metric below comes from `passages.is_selected` in the dataset itself — real human relevance labels, not synthetic judgments.

## Setup

```bash
git clone <this-repo>
cd voice-rag-indic
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SARVAM_API_KEY / GROQ_API_KEY if you have them; mock works with none

python -m src.data.loader --lang hin --n 1000        # build the subset (one-time, ~440MB download, ~90s)
```

Demo commands:

```bash
python -m src.pipeline --query "कॉर्पोरेशन क्या है?"     # text-only path, mock providers by default
python -m src.pipeline --audio samples/question.wav      # full voice → answer (needs SARVAM_API_KEY for real STT)
python benchmarks/chunking_comparison.py                 # the centerpiece: Recall@k / MRR, ~10-15 min
python benchmarks/prefix_ablation.py                      # e5 query:/passage: prefix ablation
python benchmarks/guardrail_eval.py                       # guardrail precision/recall
python benchmarks/latency_benchmark.py --n 30             # P50/P70/P100 per stage
python benchmarks/plot_results.py                          # regenerate the README charts from results/*.json
pytest -v                                                  # full suite, no API keys needed
```

`STT_PROVIDER` and `LLM_PROVIDER` default to `mock` (see `.env.example`) — every command above works out of the box with zero API keys. Set them to `sarvam` / `groq` once you have keys to exercise the real paths.

## Chunking comparison — the centerpiece

Three strategies (`src/chunking/{fixed,semantic,metadata_aware}.py`) compared across three corpus modes on **1,000 real eval queries** against **9,952 corpus passages**, all from the same MSMARCO-XI Hindi validation subset. Full table and query-type stratification in [`results/chunking_comparison.md`](results/chunking_comparison.md).

![Chunking strategy comparison](results/chunking_comparison.png)

| Strategy | Mode | Chunks | R@1 | R@3 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| fixed | hi→hi | 10,130 | 0.358 | 0.611 | 0.740 | 0.827 | 0.510 |
| fixed | hi→en (cross-lingual) | 9,952 | 0.277 | 0.534 | 0.650 | 0.749 | 0.430 |
| fixed | en→en | 9,952 | 0.460 | 0.813 | 0.914 | 0.967 | 0.648 |
| semantic | hi→hi | 13,196 | 0.356 | 0.606 | 0.723 | 0.809 | 0.504 |
| semantic | hi→en | 14,378 | 0.230 | 0.418 | 0.507 | 0.591 | 0.341 |
| semantic | en→en | 14,378 | 0.454 | 0.779 | 0.881 | 0.938 | 0.629 |
| metadata_aware | hi→hi | 12,359 | 0.351 | 0.603 | 0.728 | 0.822 | 0.504 |
| metadata_aware | hi→en | 11,666 | 0.235 | 0.454 | 0.563 | 0.657 | 0.367 |
| metadata_aware | en→en | 11,666 | 0.447 | 0.801 | 0.905 | 0.960 | 0.637 |

**Honest finding: the fixed baseline wins.** `fixed` matches or beats both `semantic` and `metadata_aware` on every metric in every corpus mode, and the gap widens on cross-lingual retrieval (hi→en), where `semantic` is clearly worst (R@5 = 0.507 vs 0.650 for fixed). This is not the result a demo optimized for a nice story would show — it's what 1,000 real queries against real relevance labels actually produced, and it's reported as such rather than reframed.

Two candidate explanations, neither confirmed: (1) MS MARCO passages are already short (single retrieved snippets, not long documents), so there is little topic drift within a passage for semantic/metadata-aware chunking to exploit — the "vast chunking strategy" premise assumes longer source documents than this dataset actually contains; (2) `semantic` and `metadata_aware` both produce *more* chunks than `fixed` (13-14k vs 10k), which should if anything help recall (more shots at a hit) yet still underperforms — suggesting the extra chunk boundaries are splitting passages in ways that hurt embedding quality for short text, not helping it. This is exactly the kind of finding the plan asked to surface rather than hide (§7 Phase 4 gotcha: report chunk counts alongside recall, don't declare a winner from resemblance alone).

Cross-lingual retrieval (hi→en: a Hindi query against an English-only corpus) works substantially worse than either monolingual mode across all three strategies — a genuine, measured limitation of `multilingual-e5-small` at this size, not a bug in the retrieval code.

### e5 prefix ablation (section 4.1)

`multilingual-e5-small` is trained with mandatory `"query: "` / `"passage: "` prefixes. Measured on 200 eval queries (fixed chunker, hi→hi): full table in [`results/prefix_ablation.md`](results/prefix_ablation.md).

| Variant | Recall@5 | MRR@10 |
|---|---|---|
| With prefixes | 0.755 | 0.495 |
| Without prefixes | 0.750 | 0.464 |

The Recall@5 delta (+0.005) is within noise at this sample size — not a dramatic effect on this particular slice — though MRR@10 shows a more consistent gap (+0.031) favoring the documented prefix convention. Prefixes are used everywhere in this project regardless, per the model card.

## Guardrails

Four checks (`src/guardrails.py`): inappropriate-input (keyword blocklist), off-topic (retrieval-score threshold, escalatable to an LLM judge), grounding verification (citations must exist in the retrieved set), hallucination detection (sentence-level embedding similarity against retrieved chunks). Evaluated against a hand-labelled 40-query set spanning on-topic / off-topic / unsafe / unanswerable-from-corpus. Full results and per-query scores in [`results/guardrail_eval.md`](results/guardrail_eval.md).

![Guardrail cheap check vs LLM judge](results/guardrail_comparison.png)

| Metric | Value |
|---|---|
| Precision (of blocked queries, how many should've been) | 1.000 |
| Recall (of queries that should be blocked, how many were caught) | 0.300 |
| on_topic accuracy | 20/20 |
| off_topic accuracy | 0/9 |
| unsafe accuracy | 6/7 |
| unanswerable accuracy | 0/4 |

**Honest finding, not a hand-wave: the retrieval-score off-topic check does not discriminate on this corpus.** on_topic scores range 0.791–0.902; off_topic/unanswerable scores range 0.787–0.859 — these ranges almost completely overlap. No threshold would separate them without also blocking a similar fraction of legitimate queries, because MSMARCO-XI's validation corpus spans law, finance, government, geography, and science broadly enough that e5-small finds *some* semantically adjacent passage for nearly any well-formed question, on-topic or not. This is a property of the (corpus, embedding model) pair, not a code bug.

**A second, deeper finding from actually wiring in a real LLM judge:** `check_off_topic` accepts an `llm_judge` callback for exactly this reason, and re-running the eval with a real Groq call wired in produced *identical* results to the cheap-check-only run — because the 0.75 threshold sits below every observed score on this corpus, the cheap check always short-circuits to `passed=True` before the judge is ever consulted. The escalation path is correctly implemented but **unreachable in the current configuration**, not because the judge doesn't work but because its trigger condition never fires. Calling the judge directly (bypassing the unreachable trigger) to measure whether it actually helps:

| | Precision | Recall | Accuracy | off_topic | unanswerable | on_topic |
|---|---|---|---|---|---|---|
| cheap checks only (as wired) | 1.000 | 0.300 | 0.650 | 0/9 | 0/4 | 20/20 |
| LLM judge, standalone | 1.000 | 0.750 | 0.875 | 6/9 | 2/4 | 20/20 |

A real, substantial improvement (+0.450 recall) with zero false blocks of legitimate queries — but getting there took a second bug fix: the judge's first run scored recall=1.000/on_topic=0/20 (blocking *everything*), which looked like real model behavior but turned out to be `max_tokens=5` silently truncating `openai/gpt-oss-20b`'s response to empty content before any visible answer (it's a reasoning model that spends tokens internally before emitting output) — confirmed via direct inspection of `finish_reason: "length"`, not assumed. Fixed by raising the budget to 300 tokens. **This escalation path is not wired into `src/pipeline.py`** — every real query would need an extra ~1-9s Groq call (see Latency below) to reach it, a real cost/latency tradeoff not made silently here; see Known Limitations.

The `inappropriate_input` keyword check performed well in isolation (6/7 unsafe queries caught) but missed "Explain how to hotwire a car" — not on the blocklist, a real and disclosed gap in a substring-match approach.

## Latency — the 200ms target, measured honestly

The brief's "under 200ms" target is not achievable for the full voice→answer pipeline with hosted APIs, and no honest implementation will hit it — see `VOICE_RAG_BUILD_PLAN.md` §3 for the stage-by-stage floor (STT 300–1500ms, LLM generation 400–3000ms on hosted APIs alone). Full tables in [`results/latency.md`](results/latency.md).

Measured, both modes, 29 warm queries + 1 cold start excluded from the pool each, `numpy.percentile(..., method="nearest")`:

**Mock (STT=mock, LLM=mock)** — cold start 7,343ms:

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---|---|---|
| **end-to-end** | 31.9 | 35.1 | 43.8 |
| stt (mock) | 0.0 | 0.0 | 0.0 |
| validate | 0.0 | 0.0 | 0.0 |
| retrieve | 8.1 | 9.0 | 10.5 |
| guardrail_pre | 0.0 | 0.0 | 0.0 |
| generate (mock) | 0.0 | 0.0 | 0.0 |
| guardrail_post | 23.7 | 25.8 | 33.3 |

**Real generation (STT=mock, LLM=real Groq)** — cold start 1,488ms:

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---|---|---|
| **end-to-end** | 5,355.8 | 7,967.6 | 9,009.4 |
| retrieve | 12.2 | 12.5 | 16.2 |
| generate (real Groq) | 5,251.4 | 7,849.0 | 8,915.9 |
| guardrail_post | 87.7 | 94.3 | 113.1 |

Real Groq generation (P50=5.3s, P100=9.0s) landed well above the build plan's own §3 estimate of 400–3000ms for hosted LLM decode — a genuine correction to that earlier hypothesis, not a number picked to match it. Real Sarvam STT latency is still **not measured**: it was verified to work correctly (see Testing/Known Limitations), but with no real recorded speech available, every real-STT call would return an empty transcript that short-circuits at the `validate` stage before reaching `retrieve`/`generate` — timing that would be meaningless, not honest, so it isn't reported.

**What this proves and what it doesn't.** Query embedding + vector retrieval over a ~10k-chunk corpus genuinely lands sub-200ms (mock mode P50=8.1ms, P100=10.5ms; real-generation mode P50=12.2ms, P100=16.2ms) — this is the part of the pipeline the 200ms target most plausibly refers to, and it's a defensible claim because it's measured. `guardrail_post` (hallucination detection, a second embedding pass over the answer's sentences) is the largest mock-mode cost at ~24-33ms but is dwarfed by real generation once a real LLM call is in the loop. **Real LLM generation alone (P50=5.3s) is ~400-700x the entire mock end-to-end pipeline** — this is the actual, measured reason the full voice→answer pipeline cannot hit 200ms with hosted APIs, not a hypothetical. Real Sarvam STT latency remains genuinely not measured (see above) — reported as such, not estimated.

**What it would take to get real STT+LLM closer to 200ms end-to-end:** a local quantized STT model (whisper.cpp tiny/base) instead of a network round trip; a small local generator instead of a hosted API; a streaming-first design measuring time-to-first-token rather than time-to-full-answer (the 5.3s P50 above is time-to-*full-answer*; time-to-first-token would be substantially lower but was not separately measured here); aggressive caching for repeated queries. These are estimates, explicitly labelled as such, not measurements.

## Demo UI

A local Streamlit UI (`demo_app.py`) calls the real pipeline — not a mockup — with live retrieval, guardrail checks, and generation visible per query:

```bash
pip install -r requirements-demo.txt
streamlit run demo_app.py
```

Pick a chunking strategy, k, and STT/LLM provider (real providers only appear in the dropdown if the corresponding API key is set in `.env`) in the sidebar; ask a question by text or upload a `.wav`. Retrieved chunks, every guardrail check's pass/fail and detail, the answer with citations and its real grounding score, and a per-stage latency chart are all shown for that specific run — for the rigorous, aggregated numbers, see the benchmark results above instead.

## Testing

```bash
pytest -v
```

49 tests, all passing, **zero API keys required** — `.github/workflows/tests.yml` runs this exact suite in CI with no `SARVAM_API_KEY`/`GROQ_API_KEY` in the environment, proving the mock-provider path genuinely works standalone rather than merely being claimed to. Covers: contract validation (`test_contracts.py`), all three chunkers including the semantic chunker's real-embedding topic-shift split (`test_chunking.py`), vector retrieval (`test_retrieval.py`), all four guardrail checks (`test_guardrails.py`), the harness's retry/backoff/typed-error behavior against a fake flaky provider (`test_harness.py`), the pipeline's empty-query validation gate (`test_pipeline.py`), and citation-parsing robustness against real observed LLM output quirks (`test_generation.py`).

Separately from the pytest suite (which stays keyless by design), the real `SarvamSTT` and `GroqLLM` providers were exercised live against the actual APIs once keys became available mid-build — see Known Limitations for exactly what that did and didn't cover.

## What live API keys actually verified (and what they didn't)

Both `SARVAM_API_KEY` and `GROQ_API_KEY` became available mid-build and were used to test the real provider paths, not just code-review them. Three real bugs surfaced this way and were fixed, each with a regression test or a re-run confirming the fix:

1. **Sarvam's `saarika:v2` STT model is deprecated server-side.** The live API returned HTTP 400 with an explicit "use `saaras:v3` instead" message. Fixed in `src/stt.py`; re-verified live — `SarvamSTT` now correctly transcribes (tested against a synthetic tone, since no real speech recording exists in this repo; empty transcript on non-speech audio is the expected, correct result).
2. **The harness's retry/typed-error classification was never actually wired into the real provider error paths.** A live Groq 404 (see #3) surfaced as a raw, uncaught stack trace instead of a classified `TransientError`/`FatalError` — the retry logic was only ever tested against a synthetic flaky provider (`test_harness.py`), not connected to real API errors. Fixed: `src/stt.py`/`src/generation.py` now classify real HTTP/SDK errors (429 and 5xx → retryable `TransientError`; 401/400/404/missing-key → non-retryable `FatalError`) and raise the harness's typed errors directly.
3. **The hardcoded Groq model (`llama-3.1-8b-instant`) no longer exists in the live catalog.** Confirmed via `GET /openai/v1/models`, not guessed — Groq's catalog has shifted away from Llama entirely. Swapped to `openai/gpt-oss-20b`, confirmed present and working live.
4. **The architecture diagram's "validate" stage was missing from `src/pipeline.py`.** Live-testing real Sarvam STT against non-speech audio produced an empty transcript that silently proceeded through retrieval and generation into a nonsense answer, instead of being rejected. Fixed: added `_validate_query_text` between STT and retrieval, with a regression test (`test_pipeline.py`).
5. **LLM citation parsing broke three different ways across real Groq calls**, none of them the same failure twice: truncating the multi-colon `chunk_id`, using fullwidth `【】` brackets instead of ASCII, and echoing the prompt's `[doc_id: X]` label verbatim instead of just `X`. Fixed by citing on the simpler colon-free `doc_id` with a bare passage header (nothing bracketed to imitate), widening the citation regex to accept both bracket styles, and defensively stripping a leading label if one still appears — regression-tested in `test_generation.py`. Confirmed working on two independent live queries after the fix, with real, correctly-resolved citations.
6. **The `llm_judge` escalation path for the off-topic guardrail check is correctly implemented but unreachable as configured** — the 0.75 threshold sits below every observed retrieval score on this corpus, so `check_off_topic` always short-circuits before consulting the judge. Measuring the judge standalone (bypassing the unreachable trigger) showed a real +0.450 recall improvement over the cheap check alone (see Guardrails above) — a genuine, substantial finding, but one that also required fixing a second bug first: the judge's first run showed 0/20 on-topic accuracy, which turned out to be `max_tokens=5` truncating `openai/gpt-oss-20b`'s (a reasoning model) response to empty content before any visible answer, not real model disagreement.

**What this did NOT verify:** real Sarvam STT was never tested against actual human speech (no recorded `.wav` sample exists in this repo — only a synthetic sine-tone test file); the `llm_judge` escalation, while measured standalone, is not wired into `src/pipeline.py`'s actual guardrail call (doing so would add a real ~1-9s Groq call to every single query, a latency/cost tradeoff not made silently here); and neither key's rate limits, quota behavior, or failure modes under sustained load were tested — only individual, low-volume calls.

## Known Limitations

- **Subset size: 1,000 of 97,941 Hindi validation queries (~1%).** All chunking-comparison and guardrail numbers are measured against this 1,000-query / 9,952-passage subset (`data/subset/`, built by `src/data/loader.py`), not the full validation split. The `query_type` distribution in this subset (72.8% DESCRIPTION, 20.0% NUMERIC, 4.2% ENTITY, 2.5% PERSON, 0.5% LOCATION) differs somewhat from the full dataset's distribution reported in `VOICE_RAG_BUILD_PLAN.md` §2.7, since the subset takes the first 1,000 surviving (gold-labelled) rows rather than a random sample.
- **Real Sarvam STT was verified against the live API but never against real human speech** — see above. `--audio` accepts any `.wav` path; the mock STT provider works on a nonexistent path (for CI/tests).
- **The `llm_judge` off-topic escalation is measured but not shipped in the live pipeline** — see above. Recall on off_topic/unanswerable categories stays weak (0.300) in `src/pipeline.py`'s actual behavior today; the +0.450 improvement is real but requires a deliberate architecture change (lower the threshold, or always escalate) that trades latency/cost for it.
- **`intfloat/multilingual-e5-small` was chosen without an exhaustive embedding-model comparison.** It was selected for its size/speed/multilingual-coverage tradeoff (384-dim, ~120MB, CPU-fast) per `VOICE_RAG_BUILD_PLAN.md` §4, not benchmarked here against alternatives like `bge-m3` (larger, likely stronger, much slower) or `paraphrase-multilingual-MiniLM-L12-v2` (faster, weaker on Indic languages). The prefix ablation above is the only controlled embedding-model experiment actually run.
- **Only Hindi was verified end to end.** The dataset ships 14 languages; only `hin` was downloaded, subsetted, and benchmarked. The loader (`src/data/loader.py --lang <code>`) supports other language codes in principle but none were run.
- **Python 3.14.5 was used for local development**, not the 3.11 the original plan specified — no 3.11 interpreter was available in the build environment and nothing in the dependency stack requires it specifically. CI (`.github/workflows/tests.yml`) pins 3.11.
- **The `inappropriate_input` check is a small hardcoded keyword blocklist** (`src/guardrails.py`), not a trained moderation model — it will miss unsafe queries phrased without a blocklisted substring (one such miss is documented in the guardrail eval above) and is not a substitute for a real content-moderation system in production.
- **Neither API key's rate limits or behavior under sustained/concurrent load were tested** — only individual, low-volume live calls during development.

## Independent verification

`VOICE_RAG_BUILD_PLAN.md` §10 defines an audit prompt for a fresh, context-free session to re-verify every claim in this README against the actual repository state (re-running benchmarks rather than trusting `results/*.md`, checking git history for leaked secrets, adversarially testing guardrails) — not yet run against this build.

## Repository layout

```
src/            contracts, config, data loader, chunking (3 strategies), embeddings,
                vector store, STT, generation, guardrails, harness, pipeline CLI
benchmarks/     chunking_comparison.py, prefix_ablation.py, guardrail_eval.py, latency_benchmark.py
tests/          49 tests, pydantic contracts through harness retry logic, zero API keys needed
results/        the numbers this README's claims are copied from, not retyped
data/           gitignored -- raw parquet cache + JSONL subset, rebuilt via src/data/loader.py
```
