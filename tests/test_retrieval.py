import shutil
import uuid

import pytest

from src.contracts import Chunk
from src.store import VectorStore

TEST_CHROMA_PATH = ".chroma_test"


@pytest.fixture(scope="module")
def store():
    collection_name = f"test_{uuid.uuid4().hex[:8]}"
    s = VectorStore(collection_name, persist_path=TEST_CHROMA_PATH, reset=True)
    yield s
    shutil.rmtree(TEST_CHROMA_PATH, ignore_errors=True)


def _make_chunks(n: int) -> list[Chunk]:
    topics = [
        "A corporation is a legal entity separate from its owners.",
        "The Ganges river flows through northern India into the Bay of Bengal.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The Reserve Bank of India regulates monetary policy and currency.",
        "Machine learning models learn patterns from labelled training data.",
    ]
    chunks = []
    for i in range(n):
        topic = topics[i % len(topics)]
        chunks.append(
            Chunk(
                chunk_id=f"fixed:doc{i}:0",
                text=f"{topic} (variant {i})",
                doc_id=f"doc{i}",
                strategy="fixed",
                metadata={"position": i},
            )
        )
    return chunks


def test_index_and_exact_text_retrieval(store):
    chunks = _make_chunks(200)
    store.index(chunks)
    assert store.count() == 200

    known = chunks[42]
    result = store.search(known.text, k=5)

    assert len(result.chunks) == 5
    top = result.chunks[0]
    assert top.chunk.chunk_id == known.chunk_id
    assert top.score > 0.9
