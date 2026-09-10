"""LLM answer-generation providers.

Callers: src/pipeline.py and src/harness.py obtain a provider via
get_llm_provider() and call .generate(query, retrieved_chunks) -> Answer.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from src import config
from src.contracts import Answer, RetrievedChunk
from src.harness import FatalError, TransientError


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> Answer:
        ...


def _first_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    match = re.search(r"[.!?।]", text)
    if match:
        return text[: match.end()].strip()
    return text


class MockLLM(LLMProvider):
    """Deterministic, offline extractive answer generator.

    No network, no API key required. Builds the answer from the top 1-2
    retrieved chunks so behavior is fully deterministic given the same
    input, and citations are guaranteed to be a subset of the chunk_ids
    actually passed in.
    """

    def generate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> Answer:
        if not retrieved_chunks:
            return Answer(
                text="I don't have enough information to answer that.",
                citations=[],
                grounded=False,
                grounding_score=0.0,
            )

        top = retrieved_chunks[:2]
        pieces = [_first_sentence(rc.chunk.text) for rc in top]
        answer_text = " ".join(p for p in pieces if p)

        return Answer(
            text=answer_text,
            citations=[rc.chunk.chunk_id for rc in top],
            grounded=True,
            grounding_score=1.0,
        )


# Matches ASCII [doc_id] as instructed, and also the CJK/fullwidth corner
# bracket style (｟U+3010/U+3011｠) observed live from real Groq output despite
# the prompt asking for ASCII brackets -- models don't always follow bracket
# style instructions exactly, so both are accepted rather than silently
# dropping every citation on a stylistic mismatch.
_CITATION_RE = re.compile(r"[\[【]([^\[\]【】\s]+)[\]】]")


class GroqLLM(LLMProvider):
    """Real LLM generation via Groq's hosted models.

    Verified live on 2026-09-11: `llama-3.1-8b-instant` (originally specified
    here per the build plan's "fastest hosted Llama inference" rationale) is no
    longer in this account's model catalog (`GET /openai/v1/models` confirmed
    it absent; Groq's current catalog has shifted to gpt-oss/qwen/compound
    models). Swapped to `openai/gpt-oss-20b`, a real available fast
    instruction-following model, confirmed present in the live catalog.
    """

    MODEL = "openai/gpt-oss-20b"

    def generate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> Answer:
        api_key = config.GROQ_API_KEY
        if not api_key:
            raise FatalError(
                "GROQ_API_KEY is not set. Get one at console.groq.com and "
                "add it to .env, or set LLM_PROVIDER=mock."
            )

        from groq import APIConnectionError, APIStatusError, Groq

        client = Groq(api_key=api_key)

        # Cite on doc_id (e.g. "q1057779_p2"), not the full chunk_id (e.g.
        # "fixed:q1057779_p2:0"). Verified live across several real Groq
        # calls: the model (a) drops/garbles segments of the multi-colon
        # chunk_id when citing it, and (b) when the context block itself
        # labels the id as "[doc_id: X]", sometimes echoes that whole label
        # back as the citation instead of just X. Passage IDs are now shown
        # as a bare "Passage X:" header (nothing bracketed to imitate) and
        # doc_id -> chunk_id(s) is resolved after parsing.
        context_blocks = "\n\n".join(
            f"Passage {rc.chunk.doc_id}:\n{rc.chunk.text}" for rc in retrieved_chunks
        )
        doc_id_to_chunk_ids: dict[str, list[str]] = {}
        for rc in retrieved_chunks:
            doc_id_to_chunk_ids.setdefault(rc.chunk.doc_id, []).append(rc.chunk.chunk_id)

        system_prompt = (
            "You are a question-answering assistant. Each context passage "
            "below is headed by its own bare ID, e.g. 'Passage q123_p4:'. "
            "Answer the user's question using ONLY the information in these "
            "passages. Do not use outside knowledge. For every factual claim "
            "you make, cite the passage ID it came from inline in square "
            "brackets containing ONLY that ID and nothing else, e.g. "
            "[q123_p4] -- not [doc_id: q123_p4], not the full passage text. "
            "If the context does not contain enough information to answer, "
            "say so plainly."
        )
        user_prompt = (
            f"Context:\n{context_blocks}\n\nQuestion: {query}\n\n"
            "Answer using only the context above, citing each passage's bare "
            "ID inline in brackets, like [q123_p4]."
        )

        try:
            completion = client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except APIStatusError as e:
            # 429 and 5xx are worth retrying (rate limit, transient server
            # error); everything else (401, 400, 404, ...) will not succeed
            # on retry -- never retry those (harness.py trap #10).
            if e.status_code == 429 or 500 <= e.status_code < 600:
                raise TransientError(f"Groq API error {e.status_code}: {e}", status_code=e.status_code) from e
            raise FatalError(f"Groq API error {e.status_code}: {e}", status_code=e.status_code) from e
        except APIConnectionError as e:
            raise TransientError(f"Groq connection error: {e}") from e

        if not completion.choices:
            # Empty choices (content filtering, provider-side edge case) is a
            # real possibility not covered by an HTTP-level error -- worth
            # retrying rather than crashing uncaught with an IndexError.
            raise TransientError("Groq API returned no completion choices")

        answer_text = completion.choices[0].message.content or ""

        found = _CITATION_RE.findall(answer_text)
        # Preserve order, drop duplicates and hallucinated doc_ids not in the
        # retrieved set (never crash on an unmatched citation marker); resolve
        # each cited doc_id to its actual chunk_id(s) from the retrieved set.
        # Defensively strip a "label: " prefix the model may echo back from
        # the prompt (e.g. "[doc_id: q123_p4]") even when told not to --
        # sampling variance means the prompt alone doesn't guarantee this.
        citations: list[str] = []
        seen: set[str] = set()
        for raw in found:
            doc_id = raw.split(":", 1)[-1].strip() if ":" in raw else raw
            for chunk_id in doc_id_to_chunk_ids.get(doc_id, []):
                if chunk_id not in seen:
                    citations.append(chunk_id)
                    seen.add(chunk_id)

        return Answer(
            text=answer_text,
            citations=citations,
            grounded=True,
            grounding_score=None,
        )


def get_llm_provider() -> LLMProvider:
    provider = config.LLM_PROVIDER
    if provider == "mock":
        return MockLLM()
    if provider == "groq":
        return GroqLLM()
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Expected 'mock' or 'groq'."
    )
