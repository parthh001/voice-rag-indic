# Voice-Enabled RAG over Indic MS MARCO — Complete Build Plan

**For: Claude Code.** This document is the full specification, verified reconnaissance, architecture, and phase-by-phase build plan. Read it end to end before writing any code.

**Author's note to the agent:** Sections marked `VERIFIED` were actually executed against the live dataset/network on 2026-09-10 — trust them. Sections marked `UNVERIFIED` are recommendations that have *not* been tested; treat them as hypotheses to confirm, not facts. Do not blur that line, and do not add to the VERIFIED list unless you ran it yourself.

---

## 1. Context

This is a voice-enabled RAG system over Indic MS MARCO, built as a portfolio project rather than a rushed submission. The goal is correctness, measurability, and honest engineering analysis over speed of delivery. The finished repo should survive a skeptical senior engineer reading it in an interview.

### Original task requirements (verbatim from the brief)

- Build a voice-enabled RAG system: **Voice input → Speech-to-text → Chunking/Retrieval (vector DB) → Answer generation**, end to end.
- Dataset: `https://huggingface.co/datasets/ai4bharat/MSMARCO-XI`
- Speech-to-text: **Sarvam or ElevenLabs**
- Chunking: *"Chunking strategy should be vast — don't submit a single naive fixed-size chunking approach."* Demonstrate semantic vs fixed-size and metadata-aware techniques.
- Performance target: **complete processing in under 200ms**
- Latency analytics: **P50/P70/P100** across multiple test queries
- Model harness: structured orchestration with **tool calls, retries, error handling, input/output validation**
- Guardrails: off-topic queries, inappropriate inputs, **hallucination detection, grounding verification**
- Deliverables: GitHub repo, live working demo, two videos (90-second process video; full end-to-end demo)

---

## 2. Reconnaissance — what is actually true about this dataset

This section exists because these findings cost real time to discover and each one invalidates an obvious-but-wrong approach. **Do not re-litigate them; build on them.**

### 2.1 `VERIFIED` — Network reachability (2026-09-10)

| Host | Status |
|---|---|
| `huggingface.co` | 200 |
| `pypi.org` | 200 |
| `api.groq.com` | 200 (reachable; key required) |
| `api.sarvam.ai` | resolves (404 at root — expected, no route at `/`) |

### 2.2 `VERIFIED` — The HuggingFace datasets-server is BROKEN for this dataset

```
GET https://datasets-server.huggingface.co/rows?dataset=ai4bharat/MSMARCO-XI&config=default&split=validation
→ {"error": "The dataset generation failed",
   "cause_message": "Nested data conversions not implemented for chunked array outputs"}
```

**Consequence:** the dataset viewer, the `/rows` API, and very likely `load_dataset("ai4bharat/MSMARCO-XI", ...)` all fail on the nested `passages` struct. **Do not build the data layer on `datasets.load_dataset()`.** Read the parquet files directly with `pyarrow`. If you try `load_dataset` and it works, great — but budget zero time for making it work, and fall back immediately.

### 2.3 `VERIFIED` — File layout and sizes

Files are **per-language parquet**, not a single blob:

```
train/{asm,ben,guj,hin,kan,mal,mar,nep,ori,pan,san,tam,urd}train.parquet
validation/{...}val.parquet
```

Measured sizes:

| File | Size |
|---|---|
| `validation/hinval.parquet` | **440 MB** |
| `validation/marval.parquet` | 451 MB |
| `train/hintrain.parquet` | **3,547 MB** |

**Never download a `train/` file.** Validation alone is far more than enough.

Languages available (14): Assamese, Bengali, Gujarati, **Hindi**, Kannada, Malayalam, **Marathi**, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu.

### 2.4 `VERIFIED` — Schema, confirmed by reading real rows

`validation/hinval.parquet` columns:

```
source_lang, target_lang, meta, Answer, query_id, query_type,
passages{is_selected[], English_passages[], Translated_passages[]},
Eng_Query, Eng_Answer, query
```

A real row:

```
target_lang : hin_Deva
query_type  : DESCRIPTION
query       : कॉर्पोरेशन क्या है?
Eng_Query   : what is a corporation?
is_selected : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
English_passages    : 10 passages
Translated_passages : 10 passages (parallel to the English ones)
```

### 2.5 `VERIFIED` — The single most important finding: **free ground truth**

`passages.is_selected` is a **relevance label**. For each query, it marks which of the 10 candidate passages actually answers it.

**This is what makes the whole project defensible.** It means the chunking comparison is not an opinion — you can compute real **Recall@k and MRR** for every strategy against human-labelled relevance. Almost every other submission will say "we implemented three chunking strategies" and show three functions. You will show a table of numbers proving which one wins and by how much. Build the entire evaluation around this.

### 2.6 `VERIFIED` — Three traps in the data

1. **The parquet file is ONE row group** (97,941 rows in `hinval.parquet`). Range-reading a slice does not work — pyarrow must pull the entire 440 MB to decode anything. Measured: **~40s to first row.** Therefore: **download once, cache to `data/raw/`, gitignore it, and build a small JSONL subset from it.** Do not re-download per run.
2. **Only ~49.6% of rows have any selected passage** (measured: 248 of the first 500). Rows where `sum(is_selected) == 0` have **no ground truth**. You must filter these out when constructing the eval set, or half your metrics will be silently meaningless.
3. Every row has exactly **10 passages**, and `English_passages[i]` is parallel to `Translated_passages[i]`. The alignment is positional — rely on it, but assert it (`len(eng) == len(trans) == len(is_selected)`) and skip malformed rows.

### 2.7 `VERIFIED` — `query_type` distribution (97,941 Hindi validation rows)

| query_type | Count | Share |
|---|---|---|
| DESCRIPTION | 52,912 | 54.0% |
| NUMERIC | 24,741 | 25.3% |
| ENTITY | 8,427 | 8.6% |
| PERSON | 6,206 | 6.3% |
| LOCATION | 5,655 | 5.8% |

This is real, free metadata. Use it two ways: (a) as a signal in the **metadata-aware chunker**, and (b) to report **retrieval quality stratified by query type** — which is the kind of analysis that separates a serious project from a demo. Expect NUMERIC and ENTITY queries to behave differently from DESCRIPTION ones; if they don't, that itself is a finding worth stating.

### 2.8 `VERIFIED` — Working parquet reader (use this, it is tested)

`fsspec`'s HTTP filesystem requires `aiohttp`. Avoid the dependency — this minimal reader works with only `requests`:

```python
import io, requests, pyarrow.parquet as pq

class HttpRangeFile(io.RawIOBase):
    """Minimal seekable file over HTTP range requests. Only needs `requests`."""
    def __init__(self, url, session=None):
        self.url = url
        self.s = session or requests.Session()
        self.pos = 0
        h = self.s.head(url, allow_redirects=True, timeout=60)
        h.raise_for_status()
        self.size = int(h.headers["Content-Length"])
        self._final = h.url                      # HF redirects to a CDN host
    def readable(self):  return True
    def seekable(self):  return True
    def tell(self):      return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        r = self.s.get(self._final, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
        r.raise_for_status()
        data = r.content
        self.pos += len(data)
        return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
```

Given 2.6.1 (single row group), prefer a **plain streamed download to disk** for the cache step and use `HttpRangeFile` only for cheap metadata probes (schema, row counts) where you do not want the full file.

---

## 3. The 200ms problem — read this before designing anything

The brief demands "complete processing in under 200ms". **As literally stated, for the full voice→answer pipeline, this is not achievable with hosted APIs, and no honest implementation will hit it.** Rough floor for each stage:

| Stage | Realistic latency | Why |
|---|---|---|
| Speech-to-text (hosted API) | 300–1500 ms | network round trip + audio upload + model inference |
| Embedding the query (local, small model) | 5–25 ms | CPU forward pass on a short string |
| Vector search (Chroma, ~10k chunks) | 1–15 ms | in-memory ANN over a small corpus |
| LLM generation (hosted, streaming) | 400–3000 ms | dominated by decode; Groq is fastest-in-class but not sub-200ms for a full answer |
| Guardrail checks (if LLM-based) | +300–1000 ms each | every check is another model call |

**The mandated response to this is not to fake it and not to ignore it.** It is to:

1. Instrument **every stage separately** and report per-stage P50/P70/P100 alongside end-to-end.
2. State plainly in the README which stages dominate, with your measured numbers.
3. Show what **is** achievable under 200ms — realistically **query embedding + vector retrieval**, which is the part the 200ms target most plausibly refers to. If your retrieval path lands at, say, P50 = 12ms / P100 = 40ms, that is a genuine and defensible "sub-200ms retrieval" claim.
4. Write a short **"what it would take"** analysis: local quantized STT (e.g. whisper.cpp tiny/base), a small local generator, streaming-first design measuring **time-to-first-token** rather than time-to-full-answer, aggressive caching. Give estimates and say they are estimates.

An honest decomposition plus a "here is the achievable subset and here is what full compliance would require" analysis is **strictly more impressive** than a fabricated 199ms. Anyone competent reading a claimed sub-200ms end-to-end voice-RAG number will assume it is measured wrong, and they will be right.

**Hard rule: never report a latency number you did not measure. Never redefine what is being measured in order to hit a target without saying so in the same sentence.**

---

## 4. Architecture and stack decisions

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

| Layer | Choice | Rationale | Rejected alternative |
|---|---|---|---|
| Language | Python 3.11 | ecosystem | — |
| STT | **Sarvam** (`saarika` family) | built for Indic audio; dataset is Indic | ElevenLabs — strong, but weaker Indic fit for this dataset |
| Embeddings | **`intfloat/multilingual-e5-small`** (local) | 384-dim, ~120MB, fast on CPU, real multilingual coverage; local = no network hop in the latency path and zero cost | `bge-m3` (better, ~2.2GB, much slower — try as an upgrade experiment if time allows); `paraphrase-multilingual-MiniLM-L12-v2` (faster, weaker Indic) |
| Vector store | **Chroma** (local persistent) | zero-config, local, free | FAISS (faster, but more plumbing for metadata filtering) |
| Generation | **Groq** free tier (Llama 3.x) | fastest hosted inference available free — directly helps the latency story | OpenAI/Gemini free tiers are viable fallbacks; keep the client behind an interface so it is swappable |
| Validation | **pydantic v2** | typed contracts between stages | dataclasses (no runtime validation) |
| Tests | **pytest** | — | unittest |

### 4.1 `UNVERIFIED` — the e5 prefix gotcha, do not skip this

`multilingual-e5-*` models are trained with **mandatory prefixes**: `"query: "` before queries and `"passage: "` before documents. Omitting them silently degrades retrieval quality — nothing errors, the numbers just get worse and you will not know why.

**Action:** encode documents as `f"passage: {text}"` and queries as `f"query: {text}"`. Then **prove it matters**: run the chunking comparison once with prefixes and once without, and put both numbers in the README. That single experiment demonstrates you understand the model rather than having copied a snippet. Verify the exact prefix convention against the model card before relying on it.

### 4.2 Cross-lingual retrieval — the free differentiator

The dataset carries **parallel English and translated passages for every row**. That means you can evaluate three retrieval modes at no extra data cost:

1. Hindi query → Hindi passages (monolingual)
2. Hindi query → **English passages** (cross-lingual)
3. English query → English passages (baseline)

Cross-lingual retrieval — ask in Marathi, retrieve from an English corpus, answer in Marathi — is a genuinely useful capability and almost nobody else will report it. Make it a first-class evaluation axis, not an afterthought.

---

## 5. Interface contracts — define these first, before any component

Everything else depends on these being stable. Write `src/contracts.py` first and do not change it casually afterwards; if a stage needs a change, change it here and re-run all tests.

```python
# src/contracts.py
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator

ChunkStrategy = Literal["fixed", "semantic", "metadata_aware"]

class Chunk(BaseModel):
    chunk_id: str
    text: str = Field(min_length=1)
    doc_id: str                      # source passage id, e.g. "q1185869_p5"
    strategy: ChunkStrategy
    metadata: dict = Field(default_factory=dict)   # query_type, lang, position, ...

class TranscriptionResult(BaseModel):
    text: str
    language: str
    provider: Literal["sarvam", "mock"]
    audio_duration_s: float | None = None

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float                     # cosine similarity, higher is better

class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    strategy: ChunkStrategy

class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    score: float | None = None

class GuardrailVerdict(BaseModel):
    passed: bool
    checks: list[CheckResult]
    @property
    def failed_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

class Answer(BaseModel):
    text: str
    citations: list[str]             # chunk_ids actually used
    grounded: bool
    grounding_score: float | None = None

class StageTiming(BaseModel):
    stage: str
    ms: float

class PipelineResult(BaseModel):
    transcript: TranscriptionResult | None
    retrieval: RetrievalResult
    guardrails: GuardrailVerdict
    answer: Answer | None            # None when guardrails reject
    timings: list[StageTiming]
    @property
    def total_ms(self) -> float:
        return sum(t.ms for t in self.timings)
```

**Every module boundary speaks these types.** No dicts crossing stage boundaries.

---

## 6. Repository structure

```
voice-rag-indic/
├── README.md                     # honest, number-backed, written LAST
├── requirements.txt
├── .env.example                  # names only, never values
├── .gitignore                    # .env, data/, *.wav, .chroma/, __pycache__
├── pytest.ini
├── .github/workflows/tests.yml   # CI: runs the suite with NO api keys
├── src/
│   ├── __init__.py
│   ├── config.py                 # env loading, model names, paths
│   ├── contracts.py              # section 5 — write first
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py             # parquet cache → JSONL subset + eval set
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── base.py               # Chunker ABC
│   │   ├── fixed.py
│   │   ├── semantic.py
│   │   └── metadata_aware.py
│   ├── embeddings.py             # e5 wrapper, prefix handling, batching
│   ├── store.py                  # Chroma index + retrieval
│   ├── stt.py                    # Sarvam client + mock provider
│   ├── generation.py             # Groq client + mock provider
│   ├── guardrails.py             # 4 checks
│   ├── harness.py                # orchestration, retries, timing
│   └── pipeline.py               # CLI entrypoint
├── benchmarks/
│   ├── chunking_comparison.py    # Recall@k / MRR per strategy
│   ├── latency_benchmark.py      # P50/P70/P100 per stage
│   └── guardrail_eval.py         # precision/recall of the guardrails themselves
├── tests/
│   ├── test_contracts.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_guardrails.py
│   └── test_harness.py
├── results/                      # committed: the numbers that back the README
│   ├── chunking_comparison.md
│   ├── latency.md
│   └── guardrail_eval.md
└── data/                         # GITIGNORED
    ├── raw/hinval.parquet
    └── subset/{corpus.jsonl,eval.jsonl}
```

---

## 7. Build plan — phases, each with a hard definition of done

Do these in order. **Do not start a phase until the previous phase's verification command actually passes and you have seen its output.** "Should work" is not done.

### Phase 0 — Scaffold and contracts
**Do:** repo skeleton, `requirements.txt`, `.gitignore` (`.env`, `data/`, `.chroma/`, `*.wav`, `__pycache__`), `.env.example`, `src/config.py`, `src/contracts.py` exactly as section 5, `pytest.ini`.
**DoD:** `pytest tests/test_contracts.py -v` passes — tests that a valid `Chunk` constructs and an empty-text `Chunk` raises `ValidationError`.
**Gotcha:** put `.gitignore` in place **before** the first `git add`, so a stray `.env` never enters history.

### Phase 1 — Data layer
**Do:** `src/data/loader.py` with two functions:
- `ensure_raw(lang="hin") -> Path` — streamed download of `validation/{lang}val.parquet` to `data/raw/`, skipped if cached. Print size and elapsed. ~440MB / ~40s first time.
- `build_subset(n_queries=1000, lang="hin") -> (corpus_path, eval_path)` — read the parquet, **filter to rows where `sum(is_selected) >= 1`** (section 2.6.2), assert `len(English_passages) == len(Translated_passages) == len(is_selected)`, take the first `n_queries` surviving rows, and emit:
  - `corpus.jsonl` — one record per passage: `{passage_id, doc_id, query_id, text_hi, text_en, query_type, position}`, deduped on text.
  - `eval.jsonl` — one record per query: `{query_id, query_hi, query_en, query_type, gold_passage_ids: [...]}`.

**DoD:** run it, then print and eyeball: number of eval queries, number of corpus passages, mean gold passages per query, `query_type` distribution. **Every eval query must have ≥1 gold passage id — assert this in code, not by hope.**
**Verify:** `python -m src.data.loader --lang hin --n 1000 && wc -l data/subset/*.jsonl`
**Expect:** ~1,000 eval queries, ~9–10k corpus passages (10 per query, minus dedupe).

### Phase 2 — Embeddings + vector store
**Do:** `src/embeddings.py` (e5 wrapper — **prefixes per 4.1**, batched encode, normalized vectors) and `src/store.py` (Chroma persistent collection per `(strategy, lang, corpus_mode)`, `index(chunks)`, `search(query, k, filters)` returning `RetrievalResult`).
**DoD:** index 200 chunks, query with the exact text of a known chunk, and assert that chunk comes back rank 1 with score > 0.9. If it doesn't, the prefix handling or normalization is wrong — fix it before moving on.
**Verify:** `pytest tests/test_retrieval.py -v`

### Phase 3 — The three chunkers
`src/chunking/base.py` defines:
```python
class Chunker(ABC):
    name: ChunkStrategy
    @abstractmethod
    def chunk(self, text: str, doc_id: str, metadata: dict) -> list[Chunk]: ...
```

- **`fixed.py`** — token/word-count windows with configurable overlap (default ~256 tokens / 15% overlap). This is the honest baseline; do not cripple it to make the others look good.
- **`semantic.py`** — split into sentences, embed each, walk them in order and start a new chunk when cosine similarity to the running chunk centroid drops below a threshold; enforce min/max chunk size so it cannot degenerate. **Must use real embedding similarity** — a sentence splitter with a size cap is *not* semantic chunking, and calling it that is the exact failure mode the brief warns about. Indic sentence splitting: handle `।` (danda) as well as `.` `?` `!`.
- **`metadata_aware.py`** — carry and exploit structure: `query_type`, passage `position` (MS MARCO passages are ranked; position is signal), source language, and `is_selected`-independent document identity. Chunks keep this in `Chunk.metadata`, and `store.search` can filter on it (e.g. NUMERIC queries bias toward chunks containing digits/units).

**DoD:** `pytest tests/test_chunking.py -v` covering, per strategy: normal text; text shorter than one chunk; text with an abrupt topic shift (semantic chunker **must** split there — assert it); Devanagari text with `।` separators; metadata survives into every emitted `Chunk`.
**Gotcha:** chunk_ids must be deterministic and unique across strategies (`f"{strategy}:{doc_id}:{i}"`), or the comparison silently compares mush.

### Phase 4 — The chunking comparison (the centerpiece)
**Do:** `benchmarks/chunking_comparison.py`. For each strategy × each corpus mode (hi→hi, hi→en, en→en): chunk the corpus, index it, run all eval queries, compute **Recall@1/@3/@5/@10 and MRR@10** against `gold_passage_ids` (a chunk counts as a hit if it derives from a gold passage — map `chunk → doc_id → passage_id`). Also report index build time, chunk count, mean chunk length, and **metrics stratified by `query_type`**.
**Output:** a markdown table written to `results/chunking_comparison.md`, plus raw JSON.
**DoD:** the table exists, is filled with real measured numbers, and the README's claim about which strategy wins is copied *from* this file — never typed from memory.
**Gotcha:** identical embedding model, identical `k`, identical eval set across all strategies. Fix random seeds. If one strategy produces far more chunks than another it gets more shots at recall — **report chunk counts alongside recall** so the comparison stays honest, and consider also reporting recall at equal chunk budget.

### Phase 5 — STT and generation, with mock providers
**Do:** `src/stt.py` and `src/generation.py`, each behind a small interface with **two implementations: real API and mock**. Selection via env (`STT_PROVIDER=sarvam|mock`). Mocks return deterministic canned output so tests and CI never need a key. Missing key → a clear, actionable error message, never a stack trace or a silent fallback that pretends to be real.
**DoD:** `STT_PROVIDER=mock python -m src.pipeline --query "कॉर्पोरेशन क्या है?"` runs end to end and returns an answer. Real-API paths stay untested until keys exist — **say so in the README rather than implying they were tested.**
**Gotcha:** never log the key. `.env` must already be gitignored (Phase 0).

### Phase 6 — Guardrails
Four checks in `src/guardrails.py`, each returning a `CheckResult`:
1. **Off-topic** — retrieval-score based first (if top-k max similarity < threshold, the corpus likely cannot answer it) and only then an LLM judgment. Cheap check first is both faster and more defensible.
2. **Inappropriate input** — refuse/deflect unsafe or abusive queries.
3. **Grounding verification** — the answer must cite `chunk_id`s that were actually retrieved; verify every citation exists in the retrieved set and reject invented ones.
4. **Hallucination detection** — sentence-level: embed each answer sentence, check max similarity against the retrieved chunks, flag sentences below threshold. Report a `grounding_score`.

**DoD:** `benchmarks/guardrail_eval.py` runs a **hand-labelled set of ~40 queries** (on-topic, off-topic, unsafe, unanswerable-from-corpus) and reports the guardrails' own **precision/recall** into `results/guardrail_eval.md`. Stating "guardrails implemented" without this number is exactly the hand-wave to avoid.
**Gotcha:** tune thresholds on a split you are willing to disclose, and disclose them. A guardrail with a 40% false-positive rate that blocks legitimate questions is a bug, not a safety feature — if yours does that, report it.

### Phase 7 — Harness
**Do:** `src/harness.py` — an orchestrator running typed stages, with per-stage retry + exponential backoff (only on transient errors: timeouts, 429, 5xx — **never** retry a 401 or a validation error), structured typed exceptions, pydantic validation at every boundary, and a timing hook recording `StageTiming` for every stage including failed attempts.
**DoD:** `pytest tests/test_harness.py -v` proving with a fake flaky provider: retries on 429 then succeeds; does **not** retry on 401; surfaces a typed error after max attempts; timings are recorded even when a stage fails.

### Phase 8 — Latency benchmark
**Do:** `benchmarks/latency_benchmark.py` — ≥30 real queries through the real pipeline, recording per-stage and end-to-end latency, reporting **P50/P70/P100** for each (`numpy.percentile`, `method="nearest"`, and say so). Run in both mock-provider and real-API modes; label which is which in the output. Report cold-start (first query, model load) separately from warm — mixing them is a classic way to produce a meaningless P100.
**DoD:** `results/latency.md` exists with real per-stage tables, plus the section-3 honest analysis of the 200ms target written against your own measured numbers.

### Phase 9 — Tests, CI, README
**Do:** full suite green; `.github/workflows/tests.yml` running `pytest` on push **with no API keys present** (proving the mock path genuinely works); README written last, from `results/*.md`.

README must contain: what it is; architecture diagram; **real measured numbers** for chunking comparison and latency; the honest 200ms analysis; setup that a stranger can follow from clone to demo; and a **Known Limitations** section naming every one of: real STT/LLM paths tested only if keys were supplied, subset size used vs full dataset, guardrail false-positive rate, embedding model chosen without exhaustive comparison, single-language verification if only Hindi was run.

**DoD:** `pytest -v` fully green, CI green on GitHub, and `git ls-files | grep -Ei '\.env$|\.wav$|parquet'` returns **nothing**.

---

## 8. Running this with parallel agents

Most of this is a dependency chain and must be sequential. Exactly one part parallelizes cleanly.

**Sequential (single session):** Phases 0, 1, 2 — everything downstream depends on the contracts, the data, and the embedding layer being settled.

**Parallel (git worktrees):** Phase 3's three chunkers are independent — they share only `base.py` and `contracts.py`, both frozen by then.

```bash
git worktree add ../vr-chunk-fixed     -b chunk-fixed
git worktree add ../vr-chunk-semantic  -b chunk-semantic
git worktree add ../vr-chunk-metadata  -b chunk-metadata
```

Open three terminals, run `claude` in each, and give each agent **only its slice** plus the frozen `Chunker` ABC and `Chunk` contract. End every scoped prompt with: *"Write only `src/chunking/<yours>.py` and `tests/test_chunking_<yours>.py`. Do not modify any other file — other agents own the sibling strategies on separate branches. Run your tests and show real passing output before you finish."*

Merge back, **read the diffs** (a clean merge is not a correct merge), then run Phase 4 in a single session.

**Sequential again:** Phases 4–9.

---

## 9. Proof-of-work standard — non-negotiable

1. **Run it, don't claim it.** Every "it works" is backed by pasted command output. No "this should pass".
2. **Never fabricate a number.** Latency, recall, precision — measured or absent. If a benchmark was not run, the README says "not measured", never an estimate formatted like a result.
3. **Mocks are labelled as mocks**, everywhere they appear in results.
4. **Secrets:** `.gitignore` before first commit; verify with `git ls-files` — never assume the ignore worked.
5. **Bugs found → regression test written**, in the same commit as the fix.
6. **Borderline results are reported as borderline.** A 2% recall difference on 1,000 queries is noise, not a win — say so rather than declaring a winner.
7. **Every limitation goes in the README**, including the unflattering ones.
8. **The README's numbers are copied from `results/*.md`**, never retyped from memory.

---

## 10. Independent verification pass

When the build is "done", open a **fresh session with no build context** and give it exactly this:

```
You are auditing this repository, not building it. You did not write this code and
have no context beyond the repo itself. Verify every claim in the README against
reality — trust nothing the README, code comments, or results/ files assert.

1. Run the full test suite. Report actual output, not a summary.
2. Run `git ls-files` against .gitignore. Confirm no .env, keys, audio, or parquet
   files are tracked. Check git history too, not just the working tree.
3. Re-run benchmarks/latency_benchmark.py yourself. Compare your numbers to
   results/latency.md. Flag any discrepancy beyond run-to-run variance.
4. Re-run benchmarks/chunking_comparison.py. Verify the README's claimed winning
   strategy is what the script actually produces — do not read the results file,
   regenerate it.
5. Adversarially test the guardrails: an off-topic question, an empty string, a
   1000-word input, a prompt-injection attempt inside the query, and a question the
   corpus provably cannot answer. Report what got through that should not have.
6. Check whether the semantic chunker is actually semantic: confirm it uses
   embedding similarity to decide boundaries and is not a sentence splitter with a
   size cap wearing a different name.
7. Confirm the eval set contains no query with zero gold passages.
8. List every README claim you could NOT independently verify, and why.

Report each finding as: CONFIRMED / COULD NOT VERIFY / CONTRADICTED, with evidence.
```

---

## 11. Traps, ranked by how much time they will cost

| # | Trap | Cost if missed |
|---|---|---|
| 1 | Building on `load_dataset()` — broken for this dataset (§2.2) | hours |
| 2 | Not filtering rows with no selected passage — half the eval has no ground truth (§2.6.2) | invalidates every metric |
| 3 | Missing e5 `query:`/`passage:` prefixes — silent quality loss (§4.1) | invisible, poisons all comparisons |
| 4 | Re-downloading the 440MB parquet per run (§2.6.1) | ~40s every run, forever |
| 5 | "Semantic" chunker that is really a sentence splitter | the exact thing the brief calls out |
| 6 | Comparing strategies with different chunk counts and calling it a fair win (§Phase 4) | conclusion is wrong |
| 7 | Chasing a literal sub-200ms end-to-end number (§3) | days, and it still fails |
| 8 | Mixing cold-start into P100 latency | meaningless tail numbers |
| 9 | Committing `.env` or the parquet cache | credential leak / bloated repo |
| 10 | Retrying on 401 in the harness | burns rate limit, hides the real error |

---

## 12. Setup appendix

**API keys (free tiers, sign up before starting Phase 5):**
- Sarvam — `sarvam.ai` → dashboard → API key → `SARVAM_API_KEY`
- Groq — `console.groq.com` → API keys → `GROQ_API_KEY`

**`.env.example` (names only, no values):**
```
SARVAM_API_KEY=
GROQ_API_KEY=
STT_PROVIDER=mock          # sarvam | mock
LLM_PROVIDER=mock          # groq | mock
EMBEDDING_MODEL=intfloat/multilingual-e5-small
DATASET_LANG=hin
CHROMA_PATH=.chroma
```

**Dependencies:**
```
sentence-transformers
chromadb
pydantic>=2
pytest
numpy
requests
python-dotenv
groq
pyarrow
pandas
```

**Demo commands the README must document:**
```bash
python -m src.data.loader --lang hin --n 1000        # build the subset (one-time)
python -m src.pipeline --audio samples/question.wav  # full voice → answer
python -m src.pipeline --query "कॉर्पोरेशन क्या है?"   # text-only path
python benchmarks/chunking_comparison.py             # the centerpiece numbers
python benchmarks/latency_benchmark.py               # P50/P70/P100
pytest -v                                            # full suite
```

**Recording checklist (for the two videos):** the 90-second process video covers why each architectural choice was made — especially the 200ms analysis, which is the most interesting thing in the project. The full demo shows a real voice question going in and a grounded, cited answer coming out, plus the benchmark tables on screen. Show a guardrail correctly *rejecting* an off-topic question — deliberately demonstrating a refusal reads as far more competent than only showing happy paths.

---

## 13. Definition of done for the whole project

- [ ] `pytest -v` green locally and in GitHub Actions with no API keys present
- [ ] `results/chunking_comparison.md` — real Recall@k/MRR for 3 strategies × 3 corpus modes, stratified by query_type
- [ ] `results/latency.md` — real per-stage P50/P70/P100 + honest 200ms analysis
- [ ] `results/guardrail_eval.md` — real precision/recall of the guardrails on a labelled set
- [ ] End-to-end demo runs from a clean clone using only the README
- [ ] `git ls-files` shows no secrets, audio, or dataset files
- [ ] Known Limitations section is specific and unflattering where warranted
- [ ] Independent verification pass (§10) run, findings addressed or documented
