from src.contracts import Chunk, RetrievedChunk
from src.generation import MockLLM, _CITATION_RE


def test_citation_regex_matches_ascii_brackets():
    assert _CITATION_RE.findall("see [q123_p4] and [q567_p8] for details") == ["q123_p4", "q567_p8"]


def test_citation_regex_matches_fullwidth_brackets():
    # Observed live from real Groq output: the model sometimes cites using
    # CJK/fullwidth corner brackets instead of the ASCII brackets requested
    # in the prompt.
    assert _CITATION_RE.findall("देखें 【q123_p4】 और 【q567_p8】") == ["q123_p4", "q567_p8"]


def test_citation_regex_matches_mixed_bracket_styles():
    assert _CITATION_RE.findall("[q1] and 【q2】") == ["q1", "q2"]


def test_mock_llm_citations_are_subset_of_input_chunk_ids():
    chunk = Chunk(chunk_id="fixed:doc1:0", text="A corporation is a legal entity.", doc_id="doc1", strategy="fixed")
    retrieved = [RetrievedChunk(chunk=chunk, score=0.9)]
    answer = MockLLM().generate("what is a corporation?", retrieved)
    assert set(answer.citations) <= {"fixed:doc1:0"}
