import pytest
from pydantic import ValidationError

from src.contracts import Chunk


def test_valid_chunk_constructs():
    chunk = Chunk(
        chunk_id="fixed:q1185869_p5:0",
        text="A corporation is a legal entity separate from its owners.",
        doc_id="q1185869_p5",
        strategy="fixed",
    )
    assert chunk.chunk_id == "fixed:q1185869_p5:0"
    assert chunk.text.startswith("A corporation")
    assert chunk.doc_id == "q1185869_p5"
    assert chunk.strategy == "fixed"
    assert chunk.metadata == {}


def test_empty_text_chunk_raises_validation_error():
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="fixed:q1185869_p5:0",
            text="",
            doc_id="q1185869_p5",
            strategy="fixed",
        )
