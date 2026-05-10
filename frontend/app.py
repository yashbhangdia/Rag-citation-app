"""
frontend/app.py

Streamlit frontend for the RAG Citation Assistant.
Run with: streamlit run frontend/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import tempfile
import os
from loguru import logger

from src.ingestion import load_single_document, chunk_documents
from src.retrieval.vector_store import VectorStoreManager
from src.generation.rag_pipeline import RAGPipeline
from src.utils.citation_builder import CitationResult
from configs.settings import settings

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.app_title,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    .main { background: #0d1117; }
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

    /* Header */
    .app-header {
        background: linear-gradient(90deg, #1a2332 0%, #1e2d40 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #79c0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .app-header p { color: #8b949e; margin: 4px 0 0; font-size: 0.9rem; }

    /* Citation card */
    .citation-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #c9d1d9;
    }
    .citation-card .ci-index {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 600;
        color: #58a6ff;
    }
    .citation-card .ci-file { font-weight: 600; color: #e6edf3; }
    .citation-card .ci-page { color: #8b949e; font-size: 0.8rem; }
    .citation-card .ci-snippet { color: #8b949e; font-style: italic; margin-top: 6px; font-size: 0.8rem; }

    /* Answer block */
    .answer-block {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 24px;
        color: #e6edf3;
        line-height: 1.75;
        font-size: 0.95rem;
    }

    /* Stats pill */
    .stat-pill {
        display: inline-block;
        background: #1f6feb22;
        border: 1px solid #1f6feb55;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        color: #58a6ff;
        font-family: 'JetBrains Mono', monospace;
        margin: 2px;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        border: 2px dashed #30363d !important;
        border-radius: 10px !important;
        background: #161b2255 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        padding: 10px 20px;
        transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.85; }

    /* Input */
    .stTextInput>div>div>input, .stTextArea textarea {
        background: #1c2128 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
        font-family: 'Sora', sans-serif !important;
    }

    /* Divider */
    hr { border-color: #30363d; }

    .section-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8b949e;
        font-weight: 600;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "vsm" not in st.session_state:
    st.session_state.vsm = VectorStoreManager()
if "pipeline" not in st.session_state:
    try:
        st.session_state.pipeline = RAGPipeline(
            vector_store=st.session_state.vsm
        )
        st.session_state.pipeline_ready = True
    except ValueError as e:
        st.session_state.pipeline = None
        st.session_state.pipeline_ready = False
        st.session_state.pipeline_error = str(e)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Helpers ───────────────────────────────────────────────────────────────────
def render_citation_card(citation) -> str:
    return f"""
<div class="citation-card">
    <span class="ci-index">[{citation.index}]</span>
    <span class="ci-file"> {citation.file_name}</span>
    <span class="ci-page"> — Page {citation.page}/{citation.total_pages}</span>
    <div class="ci-snippet">{citation.snippet}</div>
</div>
"""


def render_stat_pill(label: str, value) -> str:
    return f'<span class="stat-pill">{label}: {value}</span>'


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <h1>📚 RAG Citation Assistant</h1>
        <p>Ask questions about your documents — every answer is backed by precise citations.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">⚙ Configuration</div>', unsafe_allow_html=True)

    model_display = settings.llm_model.split("/")[-1]
    embed_display = settings.embedding_model.split("/")[-1]

    st.markdown(
        render_stat_pill("LLM", model_display) +
        render_stat_pill("Embed", embed_display) +
        render_stat_pill("Chunks", f"top-{settings.top_k_results}"),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Document upload
    st.markdown('<div class="section-title">📂 Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, DOCX or MD files",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("🔄  Index Uploaded Files", use_container_width=True):
            progress = st.progress(0, text="Starting ingestion…")
            total = len(uploaded_files)
            indexed = 0
            for i, uf in enumerate(uploaded_files):
                try:
                    progress.progress((i + 1) / total, text=f"Processing {uf.name}…")
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uf.name).suffix
                    ) as tmp:
                        tmp.write(uf.getbuffer())
                        tmp_path = tmp.name

                    docs = load_single_document(tmp_path)
                    # Preserve original filename in metadata
                    for d in docs:
                        d.metadata["file_name"] = uf.name
                        d.metadata["source"] = uf.name

                    chunks = chunk_documents(docs)
                    st.session_state.vsm.add_documents(chunks)
                    indexed += len(chunks)
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Error with {uf.name}: {e}")

            progress.empty()
            st.success(f"✅  Indexed {total} file(s) → {indexed} chunks")
            st.rerun()

    st.markdown("---")

    # Indexed documents list
    st.markdown('<div class="section-title">🗂 Indexed Documents</div>', unsafe_allow_html=True)
    docs_list = st.session_state.vsm.list_documents()
    vector_count = st.session_state.vsm.count()

    if docs_list:
        for d in docs_list:
            st.markdown(f"- 📄 `{d['file_name']}`")
        st.markdown(
            render_stat_pill("Total vectors", vector_count),
            unsafe_allow_html=True,
        )
    else:
        st.caption("No documents indexed yet. Upload files above.")

    st.markdown("---")

    # Settings
    st.markdown('<div class="section-title">🎛 Query Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Retrieved chunks (k)", 1, 10, settings.top_k_results)
    strategy = st.selectbox(
        "Retrieval strategy",
        ["mmr", "similarity"],
        help="MMR gives diverse results; similarity is fastest."
    )
    use_streaming = st.toggle("Stream response", value=True)

    st.markdown("---")
    if st.button("🗑  Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Token warning ─────────────────────────────────────────────────────────────
if not st.session_state.get("pipeline_ready", True):
    st.error(
        f"⚠️  Pipeline not ready: {st.session_state.get('pipeline_error', 'Unknown error')}\n\n"
        "Add your `HUGGINGFACE_API_TOKEN` to the `.env` file and restart the app.\n\n"
        "Get a free token at: https://huggingface.co/settings/tokens"
    )
    st.stop()

# ── Chat history display ──────────────────────────────────────────────────────
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(f'<div class="answer-block">{entry["answer"]}</div>', unsafe_allow_html=True)
        if entry.get("citations"):
            with st.expander(f"📎 {len(entry['citations'])} source(s) cited", expanded=False):
                for c in entry["citations"]:
                    st.markdown(render_citation_card(c), unsafe_allow_html=True)


# ── Chat input ────────────────────────────────────────────────────────────────
if vector_count == 0:
    st.info("👆 Upload and index documents using the sidebar to get started.")
else:
    question = st.chat_input("Ask a question about your documents…")

    if question:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                citations_placeholder = st.empty()

                with st.spinner("Retrieving relevant passages…"):
                    try:
                        pipeline: RAGPipeline = st.session_state.pipeline

                        if use_streaming:
                            token_stream, citations = pipeline.query_stream(
                                question, top_k=top_k
                            )
                            full_answer = ""
                            for token in token_stream:
                                full_answer += token
                                answer_placeholder.markdown(
                                    f'<div class="answer-block">{full_answer}▌</div>',
                                    unsafe_allow_html=True
                                )
                            answer_placeholder.markdown(
                                f'<div class="answer-block">{full_answer}</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            result: CitationResult = pipeline.query(
                                question,
                                top_k=top_k,
                                strategy=strategy,
                            )
                            full_answer = result.answer
                            citations = result.used_citations()
                            answer_placeholder.markdown(
                                f'<div class="answer-block">{full_answer}</div>',
                                unsafe_allow_html=True
                            )

                        # Show citations
                        if citations:
                            with citations_placeholder.expander(
                                f"📎 {len(citations)} source(s) cited", expanded=True
                            ):
                                for c in citations:
                                    st.markdown(render_citation_card(c), unsafe_allow_html=True)

                        # Save to history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer":   full_answer,
                            "citations": citations,
                        })

                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
                        logger.exception(e)
