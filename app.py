from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import streamlit as st

from rag_utils import (
    CHAT_MODEL,
    DOCS_DIR,
    EMBEDDING_MODEL,
    UPLOADED_DOCS_DIR,
    build_token_usage_summary,
    generate_answer,
    get_chroma_collection,
    get_collection_stats,
    ingest_folder,
    ingest_pdf,
    reset_vector_db,
    retrieve_context_with_usage,
    rerank_chunks_with_usage,
    rewrite_query_result,
    ensure_project_dirs,
)
from ui_components import (
    inject_custom_css,
    render_chat_message,
    render_document_table,
    render_empty_state,
    render_error_state,
    render_header,
    render_ingestion_status_cards,
    render_metric_card,
    render_overview,
    render_sidebar,
)


st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    ensure_project_dirs()
    defaults: dict[str, Any] = {
        "messages": [],
        "ingestion_events": [],
        "last_ingestion_results": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "pending_nav" in st.session_state:
        st.session_state.nav_section = st.session_state.pop("pending_nav")


def add_ingestion_event(message: str) -> None:
    st.session_state.ingestion_events.append(message)
    st.session_state.ingestion_events = st.session_state.ingestion_events[-20:]


def ingest_all_known_pdfs() -> None:
    collection = get_chroma_collection()
    status_area = st.empty()

    def status(message: str) -> None:
        add_ingestion_event(message)
        status_area.info(message)

    results = []
    try:
        with st.spinner("Indexing PDFs into ChromaDB..."):
            results.extend(ingest_folder(DOCS_DIR, collection=collection, status_callback=status))
            results.extend(ingest_folder(UPLOADED_DOCS_DIR, collection=collection, status_callback=status))
        st.session_state.last_ingestion_results = results
        indexed = len([result for result in results if result.get("status") == "indexed"])
        skipped = len([result for result in results if result.get("status") == "skipped"])
        if not results:
            st.warning("No PDFs found in docs/ or uploaded_docs/.")
        else:
            st.success(f"Ingestion complete: {indexed} indexed, {skipped} skipped as duplicates.")
    except Exception as exc:
        render_error_state("Ingestion failed", str(exc))


def save_uploaded_files(uploaded_files) -> list[Path]:
    saved_paths: list[Path] = []
    target_dir = Path(UPLOADED_DOCS_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files or []:
        destination = target_dir / uploaded_file.name
        destination.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(destination)
    return saved_paths


def answer_question(question: str) -> None:
    question = question.strip()
    if not question:
        return

    collection = get_chroma_collection()
    st.session_state.messages.append({"role": "user", "content": question})

    if collection.count() == 0:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "I don't know based on the uploaded documents. Upload and ingest PDFs first, then ask again.",
                "sources": [],
                "debug": {
                    "original_query": question,
                    "rewritten_query": question,
                    "retrieved_chunks": [],
                    "reranked_chunks": [],
                    "model": CHAT_MODEL,
                    "response_time": 0,
                    "prompt_tokens_estimate": 0,
                    "completion_tokens_estimate": 0,
                    "token_usage": build_token_usage_summary(
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    ),
                },
            }
        )
        st.session_state.pending_nav = "Chat / Answer"
        st.rerun()

    try:
        with st.spinner("Rewriting query, retrieving evidence, reranking chunks, and generating an answer..."):
            history = st.session_state.messages[:-1]
            rewrite_result = rewrite_query_result(question, history)
            rewritten_query = rewrite_result["query"]
            retrieval_result = retrieve_context_with_usage(rewritten_query, collection=collection, top_k=10)
            retrieved = retrieval_result["chunks"]
            rerank_result = rerank_chunks_with_usage(rewritten_query, retrieved, top_n=5)
            reranked = rerank_result["chunks"]
            result = generate_answer(question, rewritten_query, reranked, history)
            token_usage = build_token_usage_summary(
                rewrite_result.get("usage", {}),
                retrieval_result.get("usage", {}),
                rerank_result.get("usage", {}),
                result.get("usage", {}),
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "debug": {
                    "original_query": question,
                    "rewritten_query": rewritten_query,
                    "retrieved_chunks": retrieved,
                    "reranked_chunks": reranked,
                    "model": result.get("model", CHAT_MODEL),
                    "response_time": result.get("response_time", 0),
                    "prompt_tokens_estimate": result.get("prompt_tokens_estimate", 0),
                    "completion_tokens_estimate": result.get("completion_tokens_estimate", 0),
                    "answer_prompt_tokens": result.get("answer_prompt_tokens", 0),
                    "answer_completion_tokens": result.get("answer_completion_tokens", 0),
                    "answer_total_tokens": result.get("answer_total_tokens", 0),
                    "token_usage": token_usage,
                },
            }
        )
    except Exception as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"I could not generate an answer because: {exc}",
                "sources": [],
                "debug": {
                    "original_query": question,
                    "rewritten_query": question,
                    "retrieved_chunks": [],
                    "reranked_chunks": [],
                    "model": CHAT_MODEL,
                    "response_time": 0,
                    "prompt_tokens_estimate": 0,
                    "completion_tokens_estimate": 0,
                    "token_usage": build_token_usage_summary(
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    ),
                },
            }
        )

    st.session_state.pending_nav = "Chat / Answer"
    st.rerun()


def render_chat_screen(stats: dict[str, Any]) -> None:
    st.markdown('<div class="section-title">Chat / Answer</div>', unsafe_allow_html=True)
    if stats.get("total_chunks", 0) == 0:
        render_empty_state(
            "No indexed documents yet",
            "Upload PDFs in Documents, then click Ingest. The assistant will answer only from indexed files.",
        )

    for message in st.session_state.messages:
        render_chat_message(message)

    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        answer_question(prompt)


def render_documents_screen(stats: dict[str, Any]) -> None:
    st.markdown(
        """
<div class="hero-card">
  <h2 class="hero-title">Ingestion &amp; document management</h2>
  <div class="hero-copy">Upload, ingest, and manage your PDFs. The system extracts text, chunks it semantically, and stores embeddings for retrieval.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    upload_col, status_col = st.columns([1, 1.35])
    with upload_col:
        st.markdown(
            """
<div class="upload-zone">
  <div class="upload-glyph">PDF</div>
  <strong>Drag and drop PDFs here</strong>
  <p>Files are saved to uploaded_docs/ for persistent local ingestion.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            saved = save_uploaded_files(uploaded)
            st.success(f"Saved {len(saved)} file(s) to uploaded_docs/.")
        if st.button("Ingest uploaded and docs folder", type="primary", use_container_width=True):
            ingest_all_known_pdfs()

    with status_col:
        st.markdown('<div class="section-card"><div class="section-title">Ingestion progress</div>', unsafe_allow_html=True)
        events = st.session_state.ingestion_events[-6:]
        if not events:
            st.caption("No ingestion events yet.")
        for event in events:
            st.info(event)
        if st.session_state.last_ingestion_results:
            st.dataframe(st.session_state.last_ingestion_results, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card("Total documents", str(stats.get("total_documents", 0)), "Deduped by SHA-256", "warm")
    with metric_cols[1]:
        render_metric_card("Total chunks", f'{stats.get("total_chunks", 0):,}', "Stored in ChromaDB")
    with metric_cols[2]:
        render_metric_card("Storage estimate", f'{stats.get("storage_estimate_mb", 0):.2f} MB', "Source PDFs only", "gold")

    render_document_table(stats.get("documents", []))


def render_ingestion_status(stats: dict[str, Any]) -> None:
    st.markdown('<div class="ingestion-status-title">Ingestion status</div>', unsafe_allow_html=True)
    render_ingestion_status_cards(stats)
    render_document_table(stats.get("documents", []))


def render_models_screen() -> None:
    st.markdown('<div class="section-title">Models</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
<div class="section-card">
  <div class="section-title">Chat model</div>
  <h2>{CHAT_MODEL}</h2>
  <p>Used for query rewriting, beginner-friendly reranking, and final answer generation.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
<div class="section-card">
  <div class="section-title">Embedding model</div>
  <h2>{EMBEDDING_MODEL}</h2>
  <p>Used for document chunk vectors and query vectors stored in ChromaDB.</p>
</div>
""",
            unsafe_allow_html=True,
        )


def render_examples_screen() -> None:
    st.markdown('<div class="section-title">Example questions</div>', unsafe_allow_html=True)
    questions = [
        "What are the key risks mentioned in the report?",
        "Summarize the employee onboarding process.",
        "What is the refund policy for our products?",
        "Compare Q1 and Q2 performance metrics.",
        "Explain the architecture of the system.",
        "What policy details should leadership review?",
    ]
    for question in questions:
        if st.button(question, key=f"example_{question}", use_container_width=True):
            answer_question(question)


def render_settings_screen(stats: dict[str, Any]) -> None:
    st.markdown('<div class="section-title">Settings / Debug</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="section-card">
  <div class="section-title">Environment</div>
  <p>Secrets are loaded from .env. The app never hardcodes API keys.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.code("OPENAI_API_KEY=your_openai_api_key_here", language="bash")
    st.markdown("**Collection stats**")
    st.json(stats)

    st.warning("Resetting the vector database deletes all indexed chunks. Source PDFs are not deleted.")
    confirmed = st.checkbox("I understand this will delete the local ChromaDB collection.")
    if st.button("Reset vector DB", disabled=not confirmed):
        reset_vector_db()
        st.session_state.messages = []
        st.success("Vector database reset.")
        st.rerun()

    if st.button("Delete uploaded PDFs", disabled=not confirmed):
        shutil.rmtree(UPLOADED_DOCS_DIR, ignore_errors=True)
        Path(UPLOADED_DOCS_DIR).mkdir(parents=True, exist_ok=True)
        st.success("uploaded_docs/ cleared.")
        st.rerun()


def main() -> None:
    init_state()
    inject_custom_css()

    collection = get_chroma_collection()
    stats = get_collection_stats(collection)
    section = render_sidebar(stats)
    actions = render_header()

    if actions["upload"]:
        st.session_state.pending_nav = "Documents"
        st.rerun()
    if actions["clear"]:
        st.session_state.messages = []
        st.success("Chat cleared.")
        st.rerun()
    if actions["ingest"]:
        ingest_all_known_pdfs()
        stats = get_collection_stats(collection)

    if section == "App overview":
        question = render_overview(stats, st.session_state.messages)
        if question:
            answer_question(question)
        recent_docs, recent_convos = st.columns(2)
        with recent_docs:
            render_document_table(stats.get("documents", [])[:6], title="Recent documents")
        with recent_convos:
            st.markdown('<div class="section-card"><div class="section-title">Recent conversations</div>', unsafe_allow_html=True)
            user_questions = [m["content"] for m in st.session_state.messages if m.get("role") == "user"][-6:]
            if user_questions:
                for item in reversed(user_questions):
                    st.caption(item)
            else:
                st.caption("No questions asked yet.")
            st.markdown("</div>", unsafe_allow_html=True)
    elif section == "Chat / Answer":
        render_chat_screen(stats)
    elif section == "Documents":
        render_documents_screen(stats)
    elif section == "Ingestion status":
        render_ingestion_status(stats)
    elif section == "Models":
        render_models_screen()
    elif section == "Example questions":
        render_examples_screen()
    elif section == "Settings / Debug":
        render_settings_screen(stats)


if __name__ == "__main__":
    main()
