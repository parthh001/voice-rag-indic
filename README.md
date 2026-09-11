# Voice-Enabled RAG over Indic MS MARCO

[![tests](https://github.com/parthh001/voice-rag-indic/actions/workflows/tests.yml/badge.svg)](https://github.com/parthh001/voice-rag-indic/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A voice-enabled RAG system built on the [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset: **voice in → speech-to-text → retrieval → answer out**. Three chunking strategies are compared head-to-head using real relevance labels, the guardrails have a measured precision/recall (not just a claim that they exist), and the latency numbers are broken down stage by stage instead of one convenient headline figure.

The goal here was correctness and honest measurement over speed — every claim below is backed by a number in `results/`, and every limitation is written down rather than left out. See `VOICE_RAG_BUILD_PLAN.md` for the full spec and dataset notes this was built against.

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

Every retrieval number below is checked against `passages.is_selected` in the dataset itself — real human relevance labels, not something I made up.

## Setup

```bash
git clone <this-repo>
cd voice-rag-indic
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add SARVAM_API_KEY / GROQ_API_KEY if you have them; mock works with none

python -m src.data.loader --lang hin --n 1000        # build the subset (one-time, ~440MB download, ~90s)
```

Try it out:

```bash
python -m src.pipeline --query "कॉर्पोरेशन क्या है?"     # text-only, mock providers by default
python -m src.pipeline --audio samples/question.wav      # full voice → answer (needs SARVAM_API_KEY for real STT)
python benchmarks/chunking_comparison.py                 # the main benchmark: Recall@k / MRR, ~10-15 min
python benchmarks/prefix_ablation.py                      # e5 query:/passage: prefix ablation
python benchmarks/guardrail_eval.py                       # guardrail precision/recall
python benchmarks/latency_benchmark.py --n 30             # P50/P70/P100 per stage
python benchmarks/plot_results.py                          # regenerate the README charts from results/*.json
pytest -v                                                  # full suite, no API keys needed
```

`STT_PROVIDER` and `LLM_PROVIDER` default to `mock`, so everything above runs out of the box with zero API keys. Set them to `sarvam` / `groq` once you have keys to try the real thing.

## Chunking comparison — the main event

Three strategies (`src/chunking/{fixed,semantic,metadata_aware}.py`) tested across three corpus modes on **1,000 real queries** against **9,952 passages**, all from the same MSMARCO-XI Hindi validation subset. Full breakdown in [`results/chunking_comparison.md`](results/chunking_comparison.md).

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

**The simple approach wins.** Plain fixed-size chunking matches or beats the fancier semantic and metadata-aware strategies on every single metric, and the gap gets bigger on cross-lingual queries (semantic drops to R@5 = 0.507 vs. 0.650 for fixed). That's not the result you'd pick if you wanted a good story — it's just what 1,000 real queries against real answers actually showed, so that's what's reported.

Two guesses at why, neither confirmed: MS MARCO passages are already short, so there isn't much topic drift within one passage for semantic chunking to catch — the whole idea of "smart chunking helps" assumes longer documents than this dataset has. Also, semantic and metadata-aware both produce *more* chunks than fixed (13-14k vs. 10k), which should help recall if anything, and it still doesn't — suggesting the extra splits are hurting embedding quality on short text, not helping it.

Cross-lingual retrieval (asking in Hindi, searching English passages) is noticeably worse than either same-language mode, across all three strategies — a real limit of this embedding model at this size, not a bug.

### e5 prefix ablation

`multilingual-e5-small` needs a `"query: "` / `"passage: "` prefix on every input to work properly. Tested on 200 queries (fixed chunker, hi→hi): full numbers in [`results/prefix_ablation.md`](results/prefix_ablation.md).

| Variant | Recall@5 | MRR@10 |
|---|---|---|
| With prefixes | 0.755 | 0.495 |
| Without prefixes | 0.750 | 0.464 |

The Recall@5 difference is too small to mean much at this sample size, though MRR@10 shows a more consistent gap in favor of using the prefixes. They're used everywhere in this project regardless, since that's how the model was trained.

## Guardrails

Four checks (`src/guardrails.py`): is the input inappropriate (keyword list), is the question off-topic (retrieval score, can escalate to an LLM), do the citations actually exist in what was retrieved, and does the answer actually match the retrieved text (embedding similarity). Tested against 40 hand-labelled questions covering on-topic, off-topic, unsafe, and unanswerable cases. Full results in [`results/guardrail_eval.md`](results/guardrail_eval.md).

![Guardrail cheap check vs LLM judge](results/guardrail_comparison.png)

| Metric | Value |
|---|---|
| Precision (of blocked questions, how many should've been) | 1.000 |
| Recall (of questions that should be blocked, how many were caught) | 0.300 |
| on_topic accuracy | 20/20 |
| off_topic accuracy | 0/9 |
| unsafe accuracy | 6/7 |
| unanswerable accuracy | 0/4 |

**The off-topic check doesn't actually work well on this corpus.** On-topic questions score 0.791–0.902; off-topic ones score 0.787–0.859 — nearly the same range. No cutoff point would separate them without also blocking a lot of legitimate questions, because this dataset covers law, finance, government, geography, and science broadly enough that the embedding model finds *something* close to almost any question, related or not. That's a property of this particular dataset and model, not a code bug.

**A deeper check turned up something else.** The off-topic function can escalate to a real LLM for a second opinion, but wiring in a real Groq call and re-running the test gave *identical* results — because the score cutoff is set low enough that the cheap check always says "fine" before the LLM ever gets asked. The escalation path works, it's just never triggered. Calling the LLM directly (skipping the broken trigger) to see if it actually helps:

| | Precision | Recall | Accuracy | off_topic | unanswerable | on_topic |
|---|---|---|---|---|---|---|
| cheap check only (as currently wired) | 1.000 | 0.300 | 0.650 | 0/9 | 0/4 | 20/20 |
| LLM judge, called directly | 1.000 | 0.750 | 0.875 | 6/9 | 2/4 | 20/20 |

A real improvement (+0.450 recall) with zero legitimate questions wrongly blocked. Getting a clean read on this took fixing one more bug first: the LLM's first test run blocked *everything*, which looked like a real opinion but was actually `max_tokens=5` cutting off the model's response before it could answer (it's a reasoning model that uses tokens thinking before it replies) — confirmed by checking the raw response, not guessed. Raising the token limit fixed it. **This escalation isn't turned on in the live pipeline** — every real question would need an extra ~1-9 second LLM call to use it, and that's a real cost/speed tradeoff worth making on purpose, not silently.

The keyword-based unsafe-content check did well on its own (6/7 caught) but missed "Explain how to hotwire a car" since that phrase isn't on the list — a real, disclosed gap in a simple keyword approach.

## Latency — the 200ms target, honestly measured

Hitting 200ms for the *whole* voice-to-answer pipeline isn't realistic once you're calling hosted APIs — see `VOICE_RAG_BUILD_PLAN.md` §3 for why. Full numbers in [`results/latency.md`](results/latency.md).

Measured over 29 warm queries (first query excluded as cold start), both with mock providers and with a real Groq call:

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

The real Groq generation time (5.3s median) came in much higher than the plan's original estimate of 400–3000ms — a real correction to that guess, not a number picked to look good. Real Sarvam speech-to-text speed wasn't measured, since there's no real recorded question to test it with — a synthetic tone always returns an empty transcript, so timing that would be meaningless.

**What actually is fast:** turning a question into a search and getting results back takes about 8-16ms, even with a real LLM in the loop — genuinely well under 200ms, and this is probably the part the 200ms target is really about. Everything after that (the actual answer generation) is 400-700x slower once a real hosted LLM is involved, and that's the honest reason the full pipeline can't hit 200ms — not a guess, a measurement.

**What it would actually take:** a small model running locally instead of a hosted API, for both speech-to-text and generation, plus measuring time-to-first-word instead of time-to-full-answer, plus caching repeated questions. These are estimates, not something tested here.

## Try it in a browser

A small local Streamlit app (`demo_app.py`) runs the real pipeline — not a mockup — and shows retrieval, guardrail checks, and the generated answer for each question you ask:

```bash
pip install -r requirements-demo.txt
streamlit run demo_app.py
```

Pick a chunking strategy and provider in the sidebar (real STT/LLM options only show up if you've added the matching key to `.env`), then type a question or upload a `.wav`. You'll see the retrieved passages, each guardrail's pass/fail with detail, the answer with citations and its grounding score, and a timing chart for that run. For the real aggregated numbers, use the benchmark results above instead.

## Testing

```bash
pytest -v
```

49 tests, all passing, **no API keys needed** — CI runs this exact suite with no keys in the environment, so the mock-provider path is proven to work on its own rather than just assumed to. Covers the data contracts, all three chunkers (including proving the semantic chunker actually uses embeddings to split text, not just sentence length), retrieval, all four guardrail checks, the retry/backoff logic, the pipeline's empty-question handling, and citation parsing against real quirks a live LLM produced.

Separately from the test suite (which stays keyless on purpose), the real Sarvam and Groq providers were tested against the live APIs once keys became available — see below for what that did and didn't cover.

## What testing with real API keys found

Once real Sarvam and Groq keys were available, I tested the actual providers, not just reviewed the code. That turned up six real bugs, each one fixed and verified:

1. **Sarvam's `saarika:v2` speech model is discontinued.** The live API returned an error naming the replacement (`saaras:v3`). Fixed and re-tested live.
2. **Retry logic wasn't actually connected to real errors.** A real Groq error crashed with a raw stack trace instead of being caught and retried — the retry code had only ever been tested against a fake provider, never a real one. Fixed so real errors get properly classified as "retry this" or "don't bother."
3. **The Groq model I'd hardcoded doesn't exist anymore.** Confirmed by checking the live model list, not guessed. Swapped to a model that's actually available.
4. **A missing validation step let empty transcripts through.** Testing real speech-to-text against non-speech audio returned an empty transcript that then produced a nonsense answer instead of being rejected. Added the missing check, with a test to catch it happening again.
5. **Citations broke three different ways across real LLM calls** — a long ID getting cut short, unusual bracket characters, and the model echoing back a label instead of just the value. Fixed by using a simpler ID format and making the parsing more forgiving, tested against all three failure modes.
6. **The LLM-judge safety net exists but never actually turns on** — see Guardrails above. Measuring it directly showed a real +0.450 recall improvement, but that took fixing a second bug (a too-small token limit that silently blocked every answer) to measure properly.

**What this didn't cover:** real speech-to-text was never tested against an actual human voice recording, since none exists in this repo. The LLM-judge safety net is measured but not turned on by default. And neither API key was tested under heavy or sustained use — just individual calls during development.

## Known limitations

- **Only 1,000 of 97,941 Hindi questions were used** for every number above (`data/subset/`, built by `src/data/loader.py`), not the full dataset.
- **Real Sarvam speech-to-text works but was never tested on real human speech** — see above.
- **The LLM-judge safety net is measured but not turned on** in the live pipeline — turning it on would add a real few-second delay to every question, a tradeoff worth making deliberately.
- **The embedding model wasn't picked after comparing alternatives** — it was chosen for being small and fast, not benchmarked against bigger/slower options.
- **Only Hindi was tested end to end**, though the dataset covers 14 languages and the code should work with others.
- **Built and tested on Python 3.14**, not 3.11 as originally planned — nothing in the code actually needs 3.11 specifically. CI still uses 3.11.
- **The unsafe-content check is a simple keyword list**, not a trained model — it will miss things phrased in ways not on the list.
- **API rate limits and behavior under heavy load weren't tested** — just individual calls.

## Repository layout

```
src/            contracts, config, data loader, chunking (3 strategies), embeddings,
                vector store, STT, generation, guardrails, harness, pipeline CLI
benchmarks/     chunking_comparison.py, prefix_ablation.py, guardrail_eval.py, latency_benchmark.py
tests/          49 tests, pydantic contracts through harness retry logic, zero API keys needed
results/        the numbers this README's claims are copied from, not retyped
data/           gitignored -- raw parquet cache + JSONL subset, rebuilt via src/data/loader.py
```
