import pytest

from src.chunking.fixed import FixedChunker
from src.chunking.metadata_aware import MetadataAwareChunker
from src.chunking.semantic import SemanticChunker

ALL_CHUNKERS = [FixedChunker(), SemanticChunker(), MetadataAwareChunker()]

DEVANAGARI_TEXT = (
    "कॉर्पोरेशन एक कानूनी इकाई है। यह अपने मालिकों से अलग है। "
    "भारत में गंगा नदी उत्तर से बहती है। यह बंगाल की खाड़ी में गिरती है।"
)

TOPIC_SHIFT_TEXT = (
    "Photosynthesis is the process by which plants convert sunlight into chemical energy. "
    "Chlorophyll in the leaves absorbs light primarily in the blue and red wavelengths. "
    "This process produces oxygen as a byproduct which is released into the atmosphere. "
    "The Reserve Bank of India is the central banking institution of the country. "
    "It regulates the issuance and supply of the Indian rupee and monetary policy. "
    "The RBI also oversees the country's principal payment systems and banking sector."
)

NORMAL_TEXT = (
    "A corporation is a legal entity separate from its owners. "
    "It can own property, enter contracts, and be sued in its own name. "
    "Shareholders elect a board of directors to oversee major decisions. "
    "Profits may be distributed as dividends or reinvested in the business."
)

SHORT_TEXT = "This is a short passage."


@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: c.name)
def test_normal_text_produces_chunks(chunker):
    chunks = chunker.chunk(NORMAL_TEXT, doc_id="doc1", metadata={"query_type": "DESCRIPTION"})
    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == chunker.name
        assert c.doc_id == "doc1"
        assert len(c.text) > 0


@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: c.name)
def test_short_text_yields_single_chunk(chunker):
    chunks = chunker.chunk(SHORT_TEXT, doc_id="doc2", metadata={})
    assert len(chunks) == 1
    assert chunks[0].text.strip() != ""


@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: c.name)
def test_empty_text_yields_no_chunks(chunker):
    assert chunker.chunk("", doc_id="doc3", metadata={}) == []
    assert chunker.chunk("   ", doc_id="doc3", metadata={}) == []


@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: c.name)
def test_devanagari_danda_handled(chunker):
    chunks = chunker.chunk(DEVANAGARI_TEXT, doc_id="doc4", metadata={"query_type": "DESCRIPTION"})
    assert len(chunks) >= 1
    for c in chunks:
        assert c.text.strip() != ""


@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: c.name)
def test_metadata_survives_into_every_chunk(chunker):
    input_metadata = {"query_type": "NUMERIC", "position": 3, "lang": "en"}
    chunks = chunker.chunk(NORMAL_TEXT, doc_id="doc5", metadata=input_metadata)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.metadata.get("query_type") == "NUMERIC"
        assert c.metadata.get("lang") == "en"
    # caller's dict must not be mutated
    assert input_metadata == {"query_type": "NUMERIC", "position": 3, "lang": "en"}


@pytest.mark.parametrize("chunker", ALL_CHUNKERS, ids=lambda c: c.name)
def test_chunk_ids_unique_and_deterministic(chunker):
    chunks_a = chunker.chunk(NORMAL_TEXT, doc_id="doc6", metadata={})
    chunks_b = chunker.chunk(NORMAL_TEXT, doc_id="doc6", metadata={})
    ids_a = [c.chunk_id for c in chunks_a]
    ids_b = [c.chunk_id for c in chunks_b]
    assert len(ids_a) == len(set(ids_a)), "chunk_ids must be unique within one doc"
    assert ids_a == ids_b, "chunk_ids must be deterministic for the same input"
    for cid in ids_a:
        assert cid.startswith(f"{chunker.name}:doc6:")


def test_semantic_chunker_splits_on_abrupt_topic_shift():
    chunker = SemanticChunker()
    chunks = chunker.chunk(TOPIC_SHIFT_TEXT, doc_id="doc7", metadata={})
    assert len(chunks) >= 2, (
        "semantic chunker must split on an abrupt topic shift using real embedding "
        "similarity -- a sentence splitter with a size cap would keep this as one chunk"
    )
    # the photosynthesis sentences and the RBI sentences must land in different chunks
    first_chunk_text = chunks[0].text.lower()
    last_chunk_text = chunks[-1].text.lower()
    assert "photosynthesis" in first_chunk_text or "chlorophyll" in first_chunk_text
    assert "rbi" in last_chunk_text or "reserve bank" in last_chunk_text.lower()


def test_metadata_aware_chunker_sizes_numeric_tighter_than_description():
    chunker = MetadataAwareChunker()
    numeric_chunks = chunker.chunk(
        NORMAL_TEXT, doc_id="doc8", metadata={"query_type": "NUMERIC"}
    )
    description_chunks = chunker.chunk(
        NORMAL_TEXT, doc_id="doc8", metadata={"query_type": "DESCRIPTION"}
    )
    max_numeric_sentences = max(c.metadata["n_sentences"] for c in numeric_chunks)
    max_description_sentences = max(c.metadata["n_sentences"] for c in description_chunks)
    assert max_numeric_sentences <= max_description_sentences


def test_metadata_aware_chunker_flags_digits():
    chunker = MetadataAwareChunker()
    digit_text = "The population was 1.4 billion in 2023. It grew steadily over the decade."
    chunks = chunker.chunk(digit_text, doc_id="doc9", metadata={"query_type": "NUMERIC"})
    assert any(c.metadata.get("contains_digit") for c in chunks)
