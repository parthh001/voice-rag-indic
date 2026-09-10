"""multilingual-e5-small wrapper: mandatory query/passage prefixes, batched, normalized.

e5 models are trained with "query: " / "passage: " prefixes (verified against the
model card: intfloat/multilingual-e5-small). Omitting them does not error -- it
silently degrades retrieval quality. See VOICE_RAG_BUILD_PLAN.md section 4.1.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode documents with the mandatory 'passage: ' prefix. Returns normalized vectors."""
    model = _get_model()
    prefixed = [f"passage: {t}" for t in texts]
    return model.encode(
        prefixed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )


def encode_queries(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode queries with the mandatory 'query: ' prefix. Returns normalized vectors."""
    model = _get_model()
    prefixed = [f"query: {t}" for t in texts]
    return model.encode(
        prefixed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )


def encode_passages_no_prefix(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode documents WITHOUT the prefix -- exists only for the ablation in section 4.1
    (README documents the recall difference with vs without prefixes)."""
    model = _get_model()
    return model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )


def encode_queries_no_prefix(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Encode queries WITHOUT the prefix -- ablation-only, see encode_passages_no_prefix."""
    model = _get_model()
    return model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
