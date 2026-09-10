"""Local demo UI for the voice-RAG pipeline -- calls the real src.pipeline.run_pipeline,
not a mockup. Run with:

    pip install -r requirements-demo.txt
    streamlit run demo_app.py

Set STT_PROVIDER=sarvam/LLM_PROVIDER=groq (or pick them in the sidebar) with real
API keys in .env to exercise the real STT/LLM paths; otherwise everything runs on
the deterministic mock providers, same as the CLI and CI.
"""

from __future__ import annotations

import tempfile

import streamlit as st

from src import config
from src.pipeline import _CHUNKER_REGISTRY, run_pipeline

st.set_page_config(page_title="Voice RAG — Indic MS MARCO", page_icon="🎙️", layout="wide")

st.title("🎙️ Voice-Enabled RAG over Indic MS MARCO")
st.caption(
    "Ask a question in Hindi or English. Retrieval, guardrails, and generation all "
    "run for real against the MSMARCO-XI subset -- this is not a mockup of the pipeline."
)

with st.sidebar:
    st.header("Settings")
    strategy = st.selectbox("Chunking strategy", list(_CHUNKER_REGISTRY.keys()), index=0)

    stt_options = ["mock"] + (["sarvam"] if config.SARVAM_API_KEY else [])
    llm_options = ["mock"] + (["groq"] if config.GROQ_API_KEY else [])
    stt_provider = st.selectbox(
        "STT provider", stt_options, help="'sarvam' only appears if SARVAM_API_KEY is set in .env"
    )
    llm_provider = st.selectbox(
        "LLM provider", llm_options, help="'groq' only appears if GROQ_API_KEY is set in .env"
    )
    k = st.slider("Retrieved chunks (k)", 1, 10, 5)

    st.markdown("---")
    st.caption(f"Active: STT=`{stt_provider}`  LLM=`{llm_provider}`  strategy=`{strategy}`")
    st.markdown(
        "Real measured numbers (not this UI) live in [`results/*.md`](results/) -- "
        "see the README for the full chunking/guardrail/latency analysis."
    )

tab_text, tab_audio = st.tabs(["Text query", "Audio (.wav)"])

with tab_text:
    query_text_input = st.text_input("Query", value="कॉर्पोरेशन क्या है?")
    run_text = st.button("Run", key="run_text")

with tab_audio:
    st.caption(
        "With STT=mock, any .wav (even one with no real speech) returns a fixed "
        "deterministic transcript -- useful for testing the pipeline without a real "
        "recording. With STT=sarvam, real speech is actually transcribed."
    )
    uploaded = st.file_uploader("Upload a .wav file", type=["wav"])
    run_audio = st.button("Run", key="run_audio")


def _run(*, query_text: str | None = None, audio_path: str | None = None):
    config.STT_PROVIDER = stt_provider
    config.LLM_PROVIDER = llm_provider
    with st.spinner(
        "Running the real pipeline (first query is slower: loading the embedding "
        "model and building the vector index)..."
    ):
        return run_pipeline(query_text=query_text, audio_path=audio_path, strategy=strategy, k=k)


result = None
if run_text and query_text_input.strip():
    result = _run(query_text=query_text_input)
elif run_audio and uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    result = _run(audio_path=tmp_path)

if result is not None:
    if result.transcript is not None:
        st.info(f"**Transcript** ({result.transcript.provider}): {result.transcript.text}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Retrieval")
        st.write(f"Strategy: `{result.retrieval.strategy}` · {len(result.retrieval.chunks)} chunks retrieved")
        for rc in result.retrieval.chunks:
            with st.expander(f"{rc.chunk.chunk_id}  ·  score={rc.score:.3f}"):
                st.write(rc.chunk.text)
                st.json(rc.chunk.metadata)

    with col2:
        st.subheader("Guardrails")
        st.write("✅ PASSED" if result.guardrails.passed else "❌ REJECTED")
        for check in result.guardrails.checks:
            icon = "✅" if check.passed else "❌"
            score_str = f" (score={check.score:.3f})" if check.score is not None else ""
            st.write(f"{icon} **{check.name}**{score_str}: {check.detail}")

    st.subheader("Answer")
    if result.answer is not None:
        st.success(result.answer.text)
        st.write(f"Citations: `{result.answer.citations}`")
        st.write(f"Grounded: **{result.answer.grounded}**  ·  grounding_score: `{result.answer.grounding_score}`")
    else:
        st.error("No answer — guardrails rejected this query before generation was reached.")

    st.subheader("Latency (this run, not a benchmark)")
    st.write(f"Total: {result.total_ms:.1f}ms")
    st.bar_chart({t.stage: t.ms for t in result.timings})
