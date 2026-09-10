"""Shared text helpers used by both the chunkers and the guardrails -- kept in
one place so fixing/extending sentence-boundary detection (e.g. another Indic
terminator, abbreviation handling) can't drift out of sync between the
semantic/metadata-aware chunkers and the hallucination-detection guardrail
that validates against their output.
"""

from __future__ import annotations

import re

# Devanagari danda (।) plus Latin .?! as sentence terminators.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.?!])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
