from __future__ import annotations

import base64
import html
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import streamlit as st

from rag_utils import (
    CHAT_MODEL,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBEDDING_MODEL,
    UPLOADED_DOCS_DIR,
    build_token_usage_summary,
    delete_document_by_hash,
    generate_answer,
    get_chroma_collection,
    get_collection_stats,
    get_document_hash,
    ingest_folder,
    ingest_pdf,
    reset_vector_db,
    retrieve_context_with_usage,
    rerank_chunks_with_usage,
    rewrite_query_result,
    ensure_project_dirs,
)
from ui_components import (
    get_status_card_icon_svg,
    inject_custom_css,
    load_pdf_document_detail_icon_data_uri,
    load_pdf_viewer_control_icon_data_uri,
    render_chat_message,
    render_document_table,
    render_empty_state,
    render_error_state,
    render_header,
    render_ingestion_status_cards,
    render_metric_card,
    render_overview,
    render_sidebar,
    render_upload_badges,
    load_header_action_icon_data_uri,
    load_upload_icon_data_uri,
)

NAV_SECTIONS = {
    "App overview",
    "Chat / Answer",
    "Documents",
    "Ingestion status",
    "Models",
    "Example questions",
    "Settings / Debug",
}

CLIENT_MODAL_RENDERED_PREVIEW_LIMIT = 4

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
        "ingestion_active": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "pending_nav" in st.session_state:
        st.session_state.nav_section = st.session_state.pop("pending_nav")


def add_ingestion_event(message: str) -> None:
    st.session_state.ingestion_events.append(message)
    st.session_state.ingestion_events = st.session_state.ingestion_events[-20:]


def invalidate_collection_stats_cache() -> None:
    st.session_state.pop("collection_stats_cache", None)


def get_cached_collection_stats(collection, *, force_refresh: bool = False) -> dict[str, Any]:
    try:
        chunk_count = int(collection.count())
    except Exception:
        chunk_count = -1

    cache = st.session_state.get("collection_stats_cache")
    if not force_refresh and cache and cache.get("chunk_count") == chunk_count:
        return cache.get("stats", {})

    stats = get_collection_stats(collection)
    st.session_state.collection_stats_cache = {
        "chunk_count": int(stats.get("total_chunks", chunk_count) or 0),
        "stats": stats,
    }
    return stats


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
        invalidate_collection_stats_cache()
    except Exception as exc:
        invalidate_collection_stats_cache()
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


def infer_ingestion_stage(message: str) -> str | None:
    message_text = message.lower()
    if "extracting" in message_text:
        return "extract"
    if "chunking" in message_text:
        return "chunk"
    if "embedding" in message_text:
        return "embed"
    if "indexed" in message_text or "sync" in message_text:
        return "sync"
    return None


def set_ingestion_progress_run(
    *,
    active: bool,
    active_stage: str | None = None,
    active_file: str = "",
    file_index: int = 0,
    total_files: int = 0,
    results: list[dict[str, Any]] | None = None,
) -> None:
    st.session_state.ingestion_progress_run = {
        "active": active,
        "active_stage": active_stage,
        "active_file": active_file,
        "file_index": file_index,
        "total_files": total_files,
        "results": list(results or []),
    }


def clear_ingestion_progress_run() -> None:
    st.session_state.pop("ingestion_progress_run", None)


def get_upload_notice_level(results: list[dict[str, Any]]) -> str:
    indexed = len([result for result in results if str(result.get("status", "")).lower() == "indexed"])
    skipped = len([result for result in results if str(result.get("status", "")).lower() == "skipped"])
    failed = len([result for result in results if str(result.get("status", "")).lower() == "failed"])
    if failed:
        return "error"
    if skipped and not indexed:
        return "info"
    if skipped:
        return "warning"
    if indexed:
        return "success"
    return "info"


def upload_and_ingest_files(uploaded_files, progress_placeholder=None) -> list[dict[str, Any]]:
    if st.session_state.get("ingestion_active"):
        return []

    st.session_state.ingestion_active = True

    def status(message: str) -> None:
        add_ingestion_event(message)
        active_stage = infer_ingestion_stage(message)
        if active_stage:
            set_ingestion_progress_run(
                active=True,
                active_stage=active_stage,
                active_file=active_file,
                file_index=file_index,
                total_files=total_files,
                results=results,
            )
        update_ingestion_progress_placeholder(progress_placeholder)

    results: list[dict[str, Any]] = []
    try:
        collection = get_chroma_collection()
        saved_paths = save_uploaded_files(uploaded_files)
        total_files = len(saved_paths)
        set_ingestion_progress_run(active=True, active_stage="extract", total_files=total_files, results=results)
        st.session_state.pop("ingestion_progress_notice", None)
        st.session_state.pop("ingestion_progress_notice_level", None)
        update_ingestion_progress_placeholder(progress_placeholder)
        for file_index, saved_path in enumerate(saved_paths, start=1):
            active_file = saved_path.name
            set_ingestion_progress_run(
                active=True,
                active_stage="extract",
                active_file=active_file,
                file_index=file_index,
                total_files=total_files,
                results=results,
            )
            update_ingestion_progress_placeholder(progress_placeholder)
            try:
                result = ingest_pdf(saved_path, collection=collection, status_callback=status)
            except Exception as exc:
                result = {
                    "filename": saved_path.name,
                    "document_hash": "",
                    "status": "failed",
                    "reason": str(exc),
                    "pages": 0,
                    "chunks": 0,
                }
            results.append(result)
            status_text = str(result.get("status", "") or "").lower()
            if status_text == "skipped":
                add_ingestion_event(f"Skipped duplicate upload {saved_path.name}: already indexed.")
            elif status_text == "failed":
                reason = str(result.get("reason", "Ingestion failed") or "Ingestion failed")
                add_ingestion_event(f"Failed to index {saved_path.name}: {reason}.")
            elif status_text == "indexed":
                add_ingestion_event(f"Indexed uploaded PDF {saved_path.name}.")
            if status_text == "indexed":
                set_ingestion_progress_run(
                    active=True,
                    active_stage="sync",
                    active_file=active_file,
                    file_index=file_index,
                    total_files=total_files,
                    results=results,
                )
                update_ingestion_progress_placeholder(progress_placeholder)

        st.session_state.last_ingestion_results = results
        invalidate_collection_stats_cache()
        st.session_state.ingestion_progress_notice = format_upload_ingestion_notice(results)
        st.session_state.ingestion_progress_notice_level = get_upload_notice_level(results)
        clear_ingestion_progress_run()
        update_ingestion_progress_placeholder(progress_placeholder, results=results)
    finally:
        st.session_state.ingestion_active = False
    return results


def format_upload_ingestion_notice(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No files were processed."
    indexed = len([result for result in results if str(result.get("status", "")).lower() == "indexed"])
    skipped = len([result for result in results if str(result.get("status", "")).lower() == "skipped"])
    failed = len([result for result in results if str(result.get("status", "")).lower() == "failed"])

    parts = [f"Uploaded and indexed {indexed} file(s)."]
    if skipped:
        parts.append(f"Skipped {skipped} duplicate file(s).")
    if failed:
        parts.append(f"{failed} file(s) failed to index.")
    return " ".join(parts)


def format_file_size(size_bytes: Any) -> str:
    try:
        size = int(size_bytes or 0)
    except (TypeError, ValueError):
        size = 0
    if size <= 0:
        return "Unknown"
    size_mb = size / (1024 * 1024)
    if size_mb >= 1:
        return f"{size_mb:.1f} MB"
    return f"{max(1, round(size / 1024)):,} KB"


def format_ingested_timestamp(timestamp: Any) -> str:
    if not timestamp:
        return "Not available"
    timestamp_text = str(timestamp)
    try:
        return datetime.fromisoformat(timestamp_text).strftime("%b %d, %Y %I:%M %p")
    except Exception:
        return timestamp_text


def resolve_source_pdf_path(document: dict[str, Any]) -> Path | None:
    filename = str(document.get("filename", "") or "")
    document_hash = str(document.get("document_hash", "") or "")
    search_dirs = [Path(DOCS_DIR), Path(UPLOADED_DOCS_DIR)]
    filename_matches = [directory / filename for directory in search_dirs if filename and (directory / filename).exists()]

    if document_hash:
        for candidate in filename_matches:
            try:
                if get_document_hash(candidate) == document_hash:
                    return candidate
            except Exception:
                continue
        for directory in search_dirs:
            for candidate in sorted(directory.glob("*.pdf")):
                try:
                    if get_document_hash(candidate) == document_hash:
                        return candidate
                except Exception:
                    continue

    return filename_matches[0] if filename_matches else None


def get_document_location_label(document: dict[str, Any]) -> str:
    pdf_path = resolve_source_pdf_path(document)
    if not pdf_path:
        return "Source unavailable"

    try:
        resolved_path = pdf_path.resolve()
        uploaded_root = Path(UPLOADED_DOCS_DIR).resolve()
        docs_root = Path(DOCS_DIR).resolve()
        if resolved_path.is_relative_to(uploaded_root):
            return "uploaded_docs/"
        if resolved_path.is_relative_to(docs_root):
            return "docs/"
    except Exception:
        pass

    return "Source unavailable"


def find_uploaded_pdfs_by_hash(document_hash: str) -> list[Path]:
    if not document_hash:
        return []

    matches: list[Path] = []
    uploaded_dir = Path(UPLOADED_DOCS_DIR)
    if not uploaded_dir.exists():
        return matches

    for candidate in sorted(uploaded_dir.glob("*.pdf")):
        try:
            if get_document_hash(candidate) == document_hash:
                matches.append(candidate)
        except Exception:
            continue
    return matches


def find_document_by_hash(stats: dict[str, Any], target: str) -> dict[str, Any] | None:
    target = unquote(target or "").strip()
    if not target:
        return None
    for document in stats.get("documents", []):
        document_hash = str(document.get("document_hash", "") or "").strip()
        filename = str(document.get("filename", "") or "").strip()
        if target in {document_hash, filename}:
            return document
    return None


def clear_document_query_params(*names: str, rerun: bool = False) -> None:
    for name in names:
        try:
            if name in st.query_params:
                del st.query_params[name]
        except Exception:
            continue
    if rerun:
        st.rerun()


def delete_document_from_knowledge_base(document: dict[str, Any], collection) -> dict[str, Any]:
    filename = str(document.get("filename", "") or "document")
    document_hash = str(document.get("document_hash", "") or "").strip()
    if not document_hash:
        return {
            "filename": filename,
            "document_hash": "",
            "deleted_chunks": 0,
            "deleted_files": [],
            "source_file_deleted": False,
            "status": "failed",
            "reason": "Missing document hash.",
        }

    deleted_chunks = delete_document_by_hash(collection, document_hash)
    deleted_files: list[str] = []
    for pdf_path in find_uploaded_pdfs_by_hash(document_hash):
        try:
            pdf_path.unlink()
            deleted_files.append(str(pdf_path))
        except Exception:
            continue

    source_file_deleted = bool(deleted_files)
    source_copy = "removed uploaded source" if source_file_deleted else "kept source file"
    chunk_copy = f"{deleted_chunks:,} indexed chunk" + ("" if deleted_chunks == 1 else "s")
    add_ingestion_event(f"Deleted {filename}: {source_copy} and {chunk_copy}.")
    invalidate_collection_stats_cache()

    return {
        "filename": filename,
        "document_hash": document_hash,
        "deleted_chunks": deleted_chunks,
        "deleted_files": deleted_files,
        "source_file_deleted": source_file_deleted,
        "status": "deleted",
        "reason": "Deleted uploaded source and indexed chunks." if source_file_deleted else "Deleted indexed chunks; source file was retained.",
    }


def get_pdf_modal_document(stats: dict[str, Any]) -> dict[str, Any] | None:
    selected = get_query_param("view_doc")
    if not selected:
        return None

    selected = unquote(selected).strip()
    for document in stats.get("documents", []):
        document_hash = str(document.get("document_hash", "") or "").strip()
        filename = str(document.get("filename", "") or "").strip()
        if selected in {document_hash, filename}:
            return document
    return None


def pdf_modal_id(document: dict[str, Any]) -> str:
    filename = str(document.get("filename", "") or "document")
    document_hash = str(document.get("document_hash", "") or "").strip()
    target = quote(document_hash or filename, safe="")
    return f"pdf-modal-{target}"


def get_query_param(name: str) -> str:
    try:
        selected = st.query_params[name]
    except Exception:
        selected = st.query_params.get(name, "")
    if isinstance(selected, list):
        selected = selected[0] if selected else ""
    return str(selected or "").strip()


def clear_modal_query_params(*, rerun: bool = False) -> None:
    for name in ("view_doc", "from_section", "reingest_doc", "section"):
        try:
            if name in st.query_params:
                del st.query_params[name]
        except Exception:
            continue
    if rerun:
        st.rerun()


def consume_navigation_query_param() -> None:
    query_section = get_query_param("section")
    if query_section in NAV_SECTIONS:
        st.session_state.nav_section = query_section
        clear_modal_query_params(rerun=True)


def apply_modal_source_section() -> None:
    if not get_query_param("view_doc"):
        return
    source_section = get_query_param("from_section")
    if source_section in NAV_SECTIONS:
        st.session_state.nav_section = source_section


def scroll_page_to_top() -> None:
    st.iframe(
        """
<script>
(() => {
  const parentWindow = window.parent;
  const parentDocument = parentWindow.document;
  const scrollTop = () => {
    parentWindow.scrollTo({ top: 0, left: 0, behavior: "auto" });
    parentDocument.documentElement.scrollTop = 0;
    parentDocument.body.scrollTop = 0;
    parentDocument.querySelectorAll([
      "[data-testid='stMain']",
      "[data-testid='stAppViewContainer']",
      "[data-testid='stMainBlockContainer']",
      "section.main",
      "main"
    ].join(",")).forEach((element) => {
      if (typeof element.scrollTo === "function") {
        element.scrollTo({ top: 0, left: 0, behavior: "auto" });
      } else {
        element.scrollTop = 0;
      }
    });
  };
  parentWindow.requestAnimationFrame(scrollTop);
  [50, 150, 350, 700, 1200].forEach((delay) => parentWindow.setTimeout(scrollTop, delay));
})();
</script>
""",
        height=1,
        width=1,
    )


def handle_pdf_reingest_action(stats: dict[str, Any]) -> None:
    target = get_query_param("reingest_doc")
    if not target:
        return
    source_section = get_query_param("from_section")

    document = find_document_by_hash(stats, target)
    if not document:
        clear_document_query_params("reingest_doc", "selected_doc", rerun=True)
        return

    pdf_path = resolve_source_pdf_path(document)
    if not pdf_path:
        st.session_state.pdf_modal_notice = "Source PDF not found in docs/ or uploaded_docs/."
        st.session_state.document_action_notice = ("error", "Source PDF not found in docs/ or uploaded_docs/.")
    else:
        collection = get_chroma_collection()
        old_hash = str(document.get("document_hash", "") or "").strip()
        purged_chunks = delete_document_by_hash(collection, old_hash)
        result = ingest_pdf(pdf_path, collection=collection, force=True)
        st.session_state.last_ingestion_results = [result]
        add_ingestion_event(f"Re-ingested {pdf_path.name}: purged {purged_chunks:,} old chunks before indexing.")
        st.session_state.pdf_modal_notice = f"Re-ingested {pdf_path.name}."
        st.session_state.document_action_notice = ("success", f"Re-ingested {pdf_path.name}.")
        invalidate_collection_stats_cache()

    try:
        if "reingest_doc" in st.query_params:
            del st.query_params["reingest_doc"]
        if source_section == "Documents":
            if pdf_path:
                st.query_params["selected_doc"] = result.get("document_hash") or target
            if "view_doc" in st.query_params:
                del st.query_params["view_doc"]
            if "from_section" in st.query_params:
                del st.query_params["from_section"]
        else:
            st.query_params["view_doc"] = result.get("document_hash") if pdf_path else target
            if source_section in NAV_SECTIONS:
                st.query_params["from_section"] = source_section
    except Exception:
        clear_document_query_params("reingest_doc")
        if source_section == "Documents":
            st.query_params["selected_doc"] = result.get("document_hash") if pdf_path else target
        else:
            st.query_params["view_doc"] = result.get("document_hash") if pdf_path else target
            if source_section in NAV_SECTIONS:
                st.query_params["from_section"] = source_section
    st.rerun()


def confirm_delete_document(document: dict[str, Any], collection) -> None:
    document_hash = str(document.get("document_hash", "") or "").strip()
    result = delete_document_from_knowledge_base(document, collection)
    if result.get("status") == "deleted":
        filename = result.get("filename", "Document")
        chunks = int(result.get("deleted_chunks", 0) or 0)
        if result.get("source_file_deleted"):
            message = f"Deleted {filename}: removed uploaded source and {chunks:,} indexed chunks."
        else:
            message = f"Deleted {filename}: purged {chunks:,} indexed chunks and retained the source file."
        st.session_state.document_action_notice = ("success", message)
    else:
        st.session_state.document_action_notice = ("error", result.get("reason", "Document could not be deleted."))

    clear_document_query_params("delete_doc", "view_doc", "from_section")
    try:
        if get_query_param("selected_doc") == document_hash:
            del st.query_params["selected_doc"]
    except Exception:
        pass
    st.rerun()


def delete_confirmation_markup(document: dict[str, Any]) -> str:
    filename = str(document.get("filename", "") or "document")
    pages = int(document.get("pages") or 0)
    chunks = int(document.get("chunks") or 0)
    document_hash = str(document.get("document_hash", "") or "")
    short_hash = document_hash[:12] if document_hash else "n/a"
    location = get_document_location_label(document)
    uploaded_matches = find_uploaded_pdfs_by_hash(document_hash)
    source_is_uploaded = bool(uploaded_matches)

    if source_is_uploaded:
        body = (
            f"This will remove {filename} from uploaded_docs/ and delete all indexed ChromaDB chunks "
            "for this SHA-256 hash. The assistant will no longer retrieve it."
        )
        source_title = "Remove source PDF"
        source_body = "Delete the file from uploaded_docs/."
    else:
        body = (
            f"This will delete all indexed ChromaDB chunks for {filename}. The source PDF is not in "
            "uploaded_docs/, so the source file will be kept. The assistant will no longer retrieve it."
        )
        source_title = "Source PDF retained"
        source_body = f"Keep the source file in {location}."

    page_label = f"{pages:,} page" + ("" if pages == 1 else "s")
    chunk_label = f"{chunks:,} chunk" + ("" if chunks == 1 else "s")
    return f"""
<div class="delete-confirm-panel">
  <div class="delete-confirm-title">Delete document?</div>
  <div class="delete-confirm-copy">{html.escape(body)}</div>
  <div class="delete-summary-card">
    <div class="selected-pdf-mark">PDF</div>
    <div>
      <div class="selected-doc-name">{html.escape(filename)}</div>
      <div class="selected-doc-size">{html.escape(page_label)} &bull; {html.escape(chunk_label)} &bull; {html.escape(short_hash)}</div>
    </div>
  </div>
  <div class="delete-check-row"><span class="delete-check-dot">&#10003;</span><div><strong>{html.escape(source_title)}</strong><br>{html.escape(source_body)}</div></div>
  <div class="delete-check-row"><span class="delete-check-dot">&#10003;</span><div><strong>Purge vector chunks</strong><br>Delete all chunks in ChromaDB for this document hash.</div></div>
  <div class="delete-check-row"><span class="delete-check-dot info">i</span><div><strong>Chat history is not deleted</strong><br>Your conversations remain intact.</div></div>
  <div class="delete-warning">This cannot be undone from the app.</div>
</div>
"""


def render_delete_confirmation_controls(document: dict[str, Any], collection, key_prefix: str) -> None:
    cancel_col, delete_col = st.columns([1, 1.35])
    with cancel_col:
        if st.button("Cancel", key=f"{key_prefix}_cancel_delete_doc", use_container_width=True):
            clear_document_query_params("delete_doc", rerun=True)
    with delete_col:
        if st.button(
            "Delete document",
            key=f"{key_prefix}_confirm_delete_doc",
            type="primary",
            use_container_width=True,
        ):
            confirm_delete_document(document, collection)


def render_delete_confirmation_inline(document: dict[str, Any], collection, key_prefix: str = "inline") -> None:
    st.markdown(delete_confirmation_markup(document), unsafe_allow_html=True)
    render_delete_confirmation_controls(document, collection, key_prefix)


def handle_document_delete_action(stats: dict[str, Any]) -> None:
    target = get_query_param("delete_doc")
    if not target:
        return

    document = find_document_by_hash(stats, target)
    if not document:
        st.session_state.document_action_notice = ("error", "Document is no longer available.")
        clear_document_query_params("delete_doc", "selected_doc", rerun=True)
        return

    dialog = getattr(st, "dialog", None)
    if not callable(dialog):
        return

    collection = get_chroma_collection()

    @dialog("Delete document?")
    def _delete_document_dialog() -> None:
        st.markdown(delete_confirmation_markup(document), unsafe_allow_html=True)
        render_delete_confirmation_controls(document, collection, "dialog")

    _delete_document_dialog()


def pdf_data_uri(pdf_path: Path) -> str:
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def quiet_pymupdf_console_messages(fitz_module: Any) -> None:
    tools = getattr(fitz_module, "TOOLS", None)
    if not tools:
        return
    for method_name in ("mupdf_display_errors", "mupdf_display_warnings"):
        method = getattr(tools, method_name, None)
        if callable(method):
            try:
                method(False)
            except Exception:
                continue


@st.cache_data(show_spinner=False)
def render_pdf_page_images(
    pdf_path_text: str,
    file_size: int,
    modified_ns: int,
    page_scale: float = 1.4,
    thumb_scale: float = 0.22,
) -> list[dict[str, Any]]:
    _ = (file_size, modified_ns)
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PyMuPDF is not available") from exc

    quiet_pymupdf_console_messages(fitz)

    pages: list[dict[str, Any]] = []
    document = fitz.open(pdf_path_text)
    try:
        for page_index, page in enumerate(document, start=1):
            page_pixmap = page.get_pixmap(matrix=fitz.Matrix(page_scale, page_scale), alpha=False)
            thumb_pixmap = page.get_pixmap(matrix=fitz.Matrix(thumb_scale, thumb_scale), alpha=False)
            page_data = base64.b64encode(page_pixmap.tobytes("png")).decode("ascii")
            thumb_data = base64.b64encode(thumb_pixmap.tobytes("png")).decode("ascii")
            pages.append(
                {
                    "page": page_index,
                    "image_uri": f"data:image/png;base64,{page_data}",
                    "thumb_uri": f"data:image/png;base64,{thumb_data}",
                    "width": page_pixmap.width,
                    "height": page_pixmap.height,
                }
            )
    finally:
        document.close()
    return pages


def get_rendered_pdf_pages(pdf_path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        stat = pdf_path.stat()
        pages = render_pdf_page_images(str(pdf_path.resolve()), stat.st_size, stat.st_mtime_ns)
    except Exception:
        return [], "PDF preview rendering failed. Use Open full PDF to view the document."
    return pages, ""


def render_pdf_viewer_script(viewer_id: str, total_pages: int) -> None:
    script = f"""
<script>
(function() {{
  const viewerId = {viewer_id!r};
  const totalPages = {int(max(total_pages, 1))};
  const parentWindow = window.parent;
  const parentDocument = parentWindow.document;
  parentWindow.__ragPdfViewerRegistry = parentWindow.__ragPdfViewerRegistry || {{}};
  parentWindow.__ragPdfViewerRegistry[viewerId] = function() {{
  const root = parentDocument.querySelector('[data-pdf-viewer-id="' + viewerId + '"]');
  if (!root || root.dataset.viewerReady === "true") return;
  root.dataset.viewerReady = "true";

  const scroller = root.querySelector("[data-pdf-scroll]");
  const images = Array.from(root.querySelectorAll("[data-pdf-page]"));
  const thumbs = Array.from(root.querySelectorAll("[data-pdf-thumb]"));
  const pageLabel = root.querySelector("[data-pdf-page-label]");
  const zoomLabel = root.querySelector("[data-pdf-zoom-label]");
  const zoomFocus = root.querySelector("[data-pdf-zoom-focus]");
  const zoomOut = root.querySelector("[data-pdf-zoom-out]");
  const zoomIn = root.querySelector("[data-pdf-zoom-in]");
  const pagePrev = root.querySelector("[data-pdf-page-prev]");
  const pageNext = root.querySelector("[data-pdf-page-next]");
  if (!scroller || !images.length) return;

  let activePage = 1;
  let zoom = 100;
  let focusMode = false;

  function clamp(value, min, max) {{
    return Math.max(min, Math.min(max, value));
  }}

  function setActivePage(page) {{
    activePage = clamp(Number(page) || 1, 1, totalPages);
    if (pageLabel) pageLabel.textContent = "Page " + activePage + " of " + totalPages;
    if (pagePrev) pagePrev.disabled = activePage <= 1;
    if (pageNext) pageNext.disabled = activePage >= totalPages;
    thumbs.forEach((thumb) => {{
      const isActive = Number(thumb.dataset.pdfThumb) === activePage;
      thumb.classList.toggle("is-active", isActive);
      thumb.setAttribute("aria-current", isActive ? "page" : "false");
    }});
  }}

  function detectActivePage() {{
    const scrollerRect = scroller.getBoundingClientRect();
    const center = scrollerRect.top + scrollerRect.height / 2;
    let closest = activePage;
    let closestDistance = Number.POSITIVE_INFINITY;
    images.forEach((image) => {{
      const rect = image.getBoundingClientRect();
      const distance = Math.abs(rect.top + rect.height / 2 - center);
      if (distance < closestDistance) {{
        closestDistance = distance;
        closest = Number(image.dataset.pdfPage) || 1;
      }}
    }});
    setActivePage(closest);
  }}

  function scrollToPage(page, behavior) {{
    const target = root.querySelector('[data-pdf-page="' + page + '"]');
    if (!target) return;
    const targetRect = target.getBoundingClientRect();
    const scrollerRect = scroller.getBoundingClientRect();
    const nextTop = scroller.scrollTop + targetRect.top - scrollerRect.top - 12;
    scroller.scrollTo({{ top: Math.max(0, nextTop), behavior: behavior || "smooth" }});
    setActivePage(page);
  }}

  function captureBaseWidths() {{
    images.forEach((image) => {{
      if (Number(image.dataset.pdfBaseWidth || 0) > 0) return;
      const rect = image.getBoundingClientRect();
      const baseWidth = rect.width || image.clientWidth || image.naturalWidth || 1;
      image.dataset.pdfBaseWidth = String(baseWidth);
    }});
  }}

  function setFocusMode(enabled) {{
    focusMode = Boolean(enabled);
    root.classList.toggle("is-focus-zoom", focusMode);
    if (zoomFocus) {{
      zoomFocus.classList.toggle("is-active", focusMode);
      zoomFocus.setAttribute("aria-pressed", focusMode ? "true" : "false");
    }}
  }}

  function setZoom(nextZoom) {{
    captureBaseWidths();
    zoom = clamp(nextZoom, 50, 200);
    images.forEach((image) => {{
      const baseWidth = Number(image.dataset.pdfBaseWidth || 0) || image.getBoundingClientRect().width || 1;
      image.style.setProperty("width", Math.max(1, Math.round(baseWidth * zoom / 100)) + "px", "important");
      image.style.setProperty("max-width", "none", "important");
    }});
    if (zoomLabel) zoomLabel.textContent = zoom + "%";
    if (zoomOut) zoomOut.disabled = zoom <= 50;
    if (zoomIn) zoomIn.disabled = zoom >= 200;
    root.classList.toggle("is-zoomed-in", zoom > 100);
  }}

  function applyZoom(nextZoom) {{
    setZoom(nextZoom);
    window.setTimeout(function() {{ scrollToPage(activePage, "auto"); detectActivePage(); }}, 40);
  }}

  thumbs.forEach((thumb) => {{
    thumb.addEventListener("click", function(event) {{
      event.preventDefault();
      scrollToPage(Number(thumb.dataset.pdfThumb) || 1, "auto");
    }});
  }});

  if (zoomOut) {{
    zoomOut.addEventListener("click", function(event) {{
      event.preventDefault();
      applyZoom(zoom - 10);
    }});
  }}
  if (zoomIn) {{
    zoomIn.addEventListener("click", function(event) {{
      event.preventDefault();
      applyZoom(zoom + 10);
    }});
  }}
  if (zoomFocus) {{
    zoomFocus.addEventListener("click", function(event) {{
      event.preventDefault();
      setFocusMode(!focusMode);
    }});
  }}
  if (pagePrev) {{
    pagePrev.addEventListener("click", function(event) {{
      event.preventDefault();
      scrollToPage(activePage - 1, "auto");
    }});
  }}
  if (pageNext) {{
    pageNext.addEventListener("click", function(event) {{
      event.preventDefault();
      scrollToPage(activePage + 1, "auto");
    }});
  }}

  scroller.addEventListener("click", function(event) {{
    if (!focusMode) return;
    const pageImage = event.target && event.target.closest ? event.target.closest("[data-pdf-page]") : null;
    if (!pageImage || !scroller.contains(pageImage)) return;
    event.preventDefault();

    const nextZoom = clamp(zoom + 25, 50, 200);
    if (nextZoom === zoom) return;

    const imageRect = pageImage.getBoundingClientRect();
    const scrollerRect = scroller.getBoundingClientRect();
    const ratioX = (event.clientX - imageRect.left) / Math.max(1, imageRect.width);
    const ratioY = (event.clientY - imageRect.top) / Math.max(1, imageRect.height);
    const viewportX = event.clientX - scrollerRect.left;
    const viewportY = event.clientY - scrollerRect.top;
    setActivePage(Number(pageImage.dataset.pdfPage) || activePage);
    setZoom(nextZoom);

    window.setTimeout(function() {{
      const nextImageRect = pageImage.getBoundingClientRect();
      const nextScrollerRect = scroller.getBoundingClientRect();
      const nextLeft = scroller.scrollLeft + nextImageRect.left - nextScrollerRect.left + nextImageRect.width * ratioX - viewportX;
      const nextTop = scroller.scrollTop + nextImageRect.top - nextScrollerRect.top + nextImageRect.height * ratioY - viewportY;
      scroller.scrollTo({{ left: Math.max(0, nextLeft), top: Math.max(0, nextTop), behavior: "auto" }});
      window.setTimeout(detectActivePage, 40);
    }}, 140);
  }});

  parentDocument.addEventListener("keydown", function(event) {{
    if (event.key === "Escape" && focusMode) {{
      setFocusMode(false);
    }}
  }});

  let ticking = false;
  scroller.addEventListener("scroll", function() {{
    if (ticking) return;
    ticking = true;
    parentWindow.requestAnimationFrame(function() {{
      detectActivePage();
      ticking = false;
    }});
  }}, {{ passive: true }});

  parentWindow.requestAnimationFrame(function() {{
    captureBaseWidths();
    applyZoom(100);
    setFocusMode(false);
    detectActivePage();
  }});
  }};

  const root = parentDocument.querySelector('[data-pdf-viewer-id="' + viewerId + '"]');
  if (root && !root.classList.contains("is-hidden")) {{
    parentWindow.__ragPdfViewerRegistry[viewerId]();
  }}
}})();
</script>
"""
    st.iframe(script, height=1, width=1)


def render_pdf_modal_shell(
    document: dict[str, Any],
    pdf_path: Path | None,
    *,
    hidden: bool = False,
    source_section: str | None = None,
    notice: str | None = None,
    render_pages: bool = True,
) -> None:
    filename = str(document.get("filename", "") or "Untitled document")
    extension = filename.rsplit(".", 1)[-1].upper() if "." in filename else "PDF"
    pages = int(document.get("pages") or 0)
    chunks = int(document.get("chunks") or 0)
    status = str(document.get("status", "Indexed") or "Indexed").title()
    document_hash = str(document.get("document_hash", "") or "n/a")
    short_hash = document_hash[:12] if document_hash != "n/a" else "n/a"
    file_size = format_file_size(document.get("file_size") or (pdf_path.stat().st_size if pdf_path else 0))
    last_ingested = format_ingested_timestamp(document.get("last_ingested"))
    chunking_strategy = str(document.get("chunking_strategy", "semantic") or "semantic").title()
    page_label = f"{pages:,} page" if pages == 1 else f"{pages:,} pages"
    chunk_label = f"{chunks:,} chunk" if chunks == 1 else f"{chunks:,} chunks"
    pdf_uri = pdf_data_uri(pdf_path) if pdf_path else ""
    escaped_pdf_uri = html.escape(pdf_uri, quote=True)
    if pdf_path and render_pages:
        rendered_pages, render_warning = get_rendered_pdf_pages(pdf_path)
    elif pdf_path:
        rendered_pages, render_warning = [], ""
    else:
        rendered_pages, render_warning = [], ""
    total_preview_pages = len(rendered_pages) or max(pages, 1)
    if rendered_pages:
        thumb_html = "".join(
            '<div class="pdf-thumb-wrap">'
            f'<button type="button" class="pdf-thumb{" is-active" if page["page"] == 1 else ""}" '
            f'data-pdf-thumb="{page["page"]}" aria-label="Show page {page["page"]}" '
            f'aria-current="{"page" if page["page"] == 1 else "false"}">'
            f'<img src="{html.escape(page["thumb_uri"], quote=True)}" alt="Page {page["page"]} thumbnail" loading="lazy" />'
            '</button>'
            f'<span class="pdf-thumb-page">{page["page"]}</span>'
            '</div>'
            for page in rendered_pages
        )
        page_html = "".join(
            f'<img class="pdf-page-image" data-pdf-page="{page["page"]}" '
            f'src="{html.escape(page["image_uri"], quote=True)}" '
            f'alt="Page {page["page"]}" loading="lazy" '
            f'width="{page["width"]}" height="{page["height"]}" />'
            for page in rendered_pages
        )
        preview_html = f'<div class="pdf-page-scroll" data-pdf-scroll><div class="pdf-page-stack">{page_html}</div></div>'
    else:
        thumb_html = "".join(
            '<div class="pdf-thumb-wrap">'
            f'<div class="pdf-thumb{" is-active" if page == 1 else ""}"></div>'
            f'<span class="pdf-thumb-page">{page}</span>'
            '</div>'
            for page in range(1, max(1, min(total_preview_pages, 4)) + 1)
        )
        if pdf_path:
            preview_html = (
                f'<div class="pdf-preview-fallback"><div class="pdf-modal-note">{html.escape(render_warning)}</div>'
                f'<embed class="pdf-preview-iframe" src="{escaped_pdf_uri}#toolbar=0&navpanes=0&page=1" type="application/pdf" /></div>'
            )
        else:
            preview_html = '<div class="pdf-missing-source">Source PDF not found in docs/ or uploaded_docs/.</div>'
    source_section = source_section or get_query_param("from_section")
    if source_section not in NAV_SECTIONS:
        source_section = str(st.session_state.get("nav_section", "App overview"))
    close_href = f"?section={quote(source_section, safe='')}"
    open_pdf_html = (
        f'<a class="pdf-modal-action primary" href="{escaped_pdf_uri}" target="_blank" download="{html.escape(filename)}">Open full PDF</a>'
        if pdf_path
        else '<span class="pdf-modal-action primary is-disabled">Open full PDF</span>'
    )
    if notice is None:
        notice = st.session_state.pop("pdf_modal_notice", "")
    notice_html = f'<div class="pdf-modal-note">{html.escape(notice)}</div>' if notice else ""
    detail_icon_filenames = {
        "Document hash": "pdf_document_hash_icon.png",
        "Chunking strategy": "pdf_chunking_strategy_icon.png",
        "Source collection": "pdf_source_collection_icon.png",
        "File type": "pdf_file_type_icon.png",
        "File size": "pdf_file_size_icon.png",
        "Pages": "pdf_pages_icon.png",
        "Chunks": "pdf_chunks_icon.png",
        "Indexed": "pdf_indexed_status_icon.png",
        "Last ingested": "pdf_last_ingested_icon.png",
    }
    details = [
        ("Document hash", short_hash),
        ("Chunking strategy", chunking_strategy),
        ("Source collection", COLLECTION_NAME),
        ("File type", extension),
        ("File size", file_size),
        ("Pages", f"{pages:,}"),
        ("Chunks", f"{chunks:,}"),
        ("Indexed", "Yes" if "index" in status.lower() else status),
        ("Last ingested", last_ingested),
    ]
    detail_rows = []
    for label, value in details:
        icon_uri = load_pdf_document_detail_icon_data_uri(detail_icon_filenames[label])
        icon_html = (
            f'<img class="pdf-modal-icon pdf-detail-icon" src="{html.escape(icon_uri, quote=True)}" '
            f'alt="" aria-hidden="true" loading="lazy" />'
            if icon_uri
            else '<span class="pdf-modal-icon pdf-detail-icon is-empty" aria-hidden="true"></span>'
        )
        detail_rows.append(
            f'<div class="pdf-detail-row">{icon_html}<div class="pdf-detail-label">{html.escape(label)}</div>'
            f'<div class="pdf-detail-value{" is-yes" if label == "Indexed" else ""}">{html.escape(str(value))}</div></div>'
        )
    detail_html = "".join(detail_rows)
    modal_id = pdf_modal_id(document)
    viewer_id = f"pdf-viewer-{document_hash[:12] if document_hash != 'n/a' else quote(filename, safe='')}"
    viewer_icon_filenames = {
        "zoom_focus": "pdf_zoom_focus_icon.png",
        "zoom_out": "pdf_zoom_out_icon.png",
        "zoom_in": "pdf_zoom_in_icon.png",
        "page_prev": "pdf_page_prev_icon.png",
        "page_next": "pdf_page_next_icon.png",
    }
    viewer_icons = {
        key: load_pdf_viewer_control_icon_data_uri(filename)
        for key, filename in viewer_icon_filenames.items()
    }
    zoom_focus_html = (
        f'<img class="pdf-control-icon" src="{html.escape(viewer_icons["zoom_focus"], quote=True)}" alt="" aria-hidden="true" loading="lazy" />'
        if viewer_icons["zoom_focus"]
        else '<span class="pdf-control-fallback" aria-hidden="true">Z</span>'
    )
    zoom_out_html = (
        f'<img class="pdf-control-icon" src="{html.escape(viewer_icons["zoom_out"], quote=True)}" alt="" aria-hidden="true" loading="lazy" />'
        if viewer_icons["zoom_out"]
        else '<span class="pdf-control-fallback" aria-hidden="true">-</span>'
    )
    zoom_in_html = (
        f'<img class="pdf-control-icon" src="{html.escape(viewer_icons["zoom_in"], quote=True)}" alt="" aria-hidden="true" loading="lazy" />'
        if viewer_icons["zoom_in"]
        else '<span class="pdf-control-fallback" aria-hidden="true">+</span>'
    )
    page_prev_html = (
        f'<img class="pdf-control-icon" src="{html.escape(viewer_icons["page_prev"], quote=True)}" alt="" aria-hidden="true" loading="lazy" />'
        if viewer_icons["page_prev"]
        else '<span class="pdf-control-fallback" aria-hidden="true">&lt;</span>'
    )
    page_next_html = (
        f'<img class="pdf-control-icon" src="{html.escape(viewer_icons["page_next"], quote=True)}" alt="" aria-hidden="true" loading="lazy" />'
        if viewer_icons["page_next"]
        else '<span class="pdf-control-fallback" aria-hidden="true">&gt;</span>'
    )
    preview_info_uri = load_pdf_document_detail_icon_data_uri("pdf_preview_info_icon.png")
    preview_info_icon = (
        f'<img class="pdf-modal-icon pdf-preview-info-icon" src="{html.escape(preview_info_uri, quote=True)}" '
        'alt="" aria-hidden="true" loading="lazy" />'
        if preview_info_uri
        else '<span class="pdf-modal-icon pdf-preview-info-icon is-empty" aria-hidden="true"></span>'
    )
    preview_note_html = (
        f'<div class="pdf-modal-note pdf-preview-note">{preview_info_icon}'
        '<span>Preview the original source PDF used for indexing and retrieval.</span></div>'
    )
    overlay_class = "pdf-modal-overlay is-hidden" if hidden else "pdf-modal-overlay"
    close_control_html = (
        '<a class="pdf-modal-close" href="#pdf-modal-closed" data-pdf-modal-close aria-label="Close">&times;</a>'
        if hidden
        else f'<a class="pdf-modal-close" href="{close_href}" target="_self" aria-label="Close">&times;</a>'
    )
    modal_html = (
        f'<div id="{html.escape(modal_id, quote=True)}" class="{overlay_class}" '
        f'data-pdf-client-modal="{"true" if hidden else "false"}" '
        f'data-pdf-viewer-id="{html.escape(viewer_id, quote=True)}" aria-hidden="{"true" if hidden else "false"}">'
        '<div class="pdf-modal-dialog"><div class="pdf-modal-shell">'
        '<div class="pdf-modal-header"><div class="pdf-modal-heading">'
        '<div class="pdf-modal-badge">PDF</div>'
        f'<div class="pdf-modal-title" title="{html.escape(filename)}">{html.escape(filename)}</div>'
        f'</div>{close_control_html}</div>'
        '<div class="pdf-modal-pills">'
        f'<span class="pdf-modal-pill">{html.escape(extension)}</span>'
        f'<span class="pdf-modal-pill">{html.escape(file_size)}</span>'
        f'<span class="pdf-modal-pill">{html.escape(page_label)}</span>'
        f'<span class="pdf-modal-pill">{html.escape(chunk_label)}</span>'
        f'<span class="pdf-modal-pill is-indexed">{html.escape(status)}</span>'
        f'<span class="pdf-modal-pill">Last ingested {html.escape(last_ingested)}</span>'
        '</div><div class="pdf-modal-layout">'
        '<div class="pdf-modal-preview"><div class="pdf-modal-section-title">Document preview</div>'
        '<div class="pdf-preview-stage">'
        f'<div class="pdf-thumb-rail">{thumb_html}</div><div><div class="pdf-frame-shell">{preview_html}</div>'
        '<div class="pdf-preview-footer"><span class="pdf-page-nav">'
        f'<button type="button" class="pdf-page-nav-button" data-pdf-page-prev aria-label="Previous page">{page_prev_html}</button>'
        f'<span data-pdf-page-label>Page 1 of {total_preview_pages:,}</span>'
        f'<button type="button" class="pdf-page-nav-button" data-pdf-page-next aria-label="Next page">{page_next_html}</button>'
        '</span>'
        '<span class="pdf-preview-controls" aria-label="PDF zoom controls">'
        f'<button type="button" class="pdf-zoom-button pdf-zoom-focus-button" data-pdf-zoom-focus aria-label="Zoom focus" aria-pressed="false">{zoom_focus_html}</button>'
        f'<button type="button" class="pdf-zoom-button" data-pdf-zoom-out aria-label="Zoom out">{zoom_out_html}</button>'
        '<span class="pdf-zoom-label" data-pdf-zoom-label>100%</span>'
        f'<button type="button" class="pdf-zoom-button" data-pdf-zoom-in aria-label="Zoom in">{zoom_in_html}</button>'
        '</span></div>'
        '</div></div></div>'
        '<div class="pdf-modal-details"><div class="pdf-details-title">Document details</div>'
        f'{detail_html}{preview_note_html}{notice_html}'
        f'<div class="pdf-modal-actions-title">Actions</div>{open_pdf_html}</div>'
        '</div></div></div></div>'
    )
    st.markdown(modal_html, unsafe_allow_html=True)
    if rendered_pages and not hidden:
        render_client_pdf_modal_controller()


def render_pdf_preview_dialog(document: dict[str, Any]) -> None:
    pdf_path = resolve_source_pdf_path(document)
    render_pdf_modal_shell(document, pdf_path)


def render_client_pdf_modal_controller() -> None:
    st.iframe(
        """
<script>
(() => {
  const parentWindow = window.parent;
  const parentDocument = parentWindow.document;
  const VERSION = "2026-06-27.2";
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  parentWindow.__ragPdfModalLastTrigger = parentWindow.__ragPdfModalLastTrigger || null;
  const VISIBLE_MODAL_SELECTOR = '.pdf-modal-overlay[data-pdf-viewer-id]:not(.is-hidden), .pdf-modal-overlay[data-pdf-viewer-id].is-hidden:target';

  const getViewerRoot = (element) => (
    element && element.closest ? element.closest(".pdf-modal-overlay[data-pdf-viewer-id]") : null
  );
  const getOpenModals = () => Array.from(
    parentDocument.querySelectorAll(
      '.pdf-modal-overlay[data-pdf-client-modal="true"]:not(.is-hidden), .pdf-modal-overlay[data-pdf-client-modal="true"].is-hidden:target'
    )
  );
  const getParts = (root) => ({
    scroller: root.querySelector("[data-pdf-scroll]"),
    images: Array.from(root.querySelectorAll("[data-pdf-page]")),
    thumbs: Array.from(root.querySelectorAll("[data-pdf-thumb]")),
    pageLabel: root.querySelector("[data-pdf-page-label]"),
    zoomLabel: root.querySelector("[data-pdf-zoom-label]"),
    zoomFocus: root.querySelector("[data-pdf-zoom-focus]"),
    zoomOut: root.querySelector("[data-pdf-zoom-out]"),
    zoomIn: root.querySelector("[data-pdf-zoom-in]"),
    pagePrev: root.querySelector("[data-pdf-page-prev]"),
    pageNext: root.querySelector("[data-pdf-page-next]"),
  });

  const captureBaseWidths = (root) => {
    const { images } = getParts(root);
    images.forEach((image) => {
      const rect = image.getBoundingClientRect();
      const measured = rect.width || image.clientWidth || image.naturalWidth || Number(image.getAttribute("width")) || 1;
      if (!image.dataset.pdfBaseWidth || Number(image.dataset.pdfBaseWidth) <= 0) {
        image.dataset.pdfBaseWidth = String(measured);
      }
    });
  };

  const setActivePage = (root, page) => {
    const state = root && root.__ragPdfViewerState;
    if (!state) return;
    const { thumbs, pageLabel, pagePrev, pageNext } = getParts(root);
    state.activePage = clamp(Number(page) || 1, 1, state.totalPages);
    if (pageLabel) pageLabel.textContent = "Page " + state.activePage + " of " + state.totalPages;
    if (pagePrev) pagePrev.disabled = state.activePage <= 1;
    if (pageNext) pageNext.disabled = state.activePage >= state.totalPages;
    thumbs.forEach((thumb) => {
      const isActive = Number(thumb.dataset.pdfThumb) === state.activePage;
      thumb.classList.toggle("is-active", isActive);
      thumb.setAttribute("aria-current", isActive ? "page" : "false");
    });
  };

  const detectActivePage = (root) => {
    const state = ensureViewer(root);
    const { scroller, images } = getParts(root);
    if (!state || !scroller || !images.length) return;
    const scrollerRect = scroller.getBoundingClientRect();
    const center = scrollerRect.top + scrollerRect.height / 2;
    let closest = state.activePage;
    let closestDistance = Number.POSITIVE_INFINITY;
    images.forEach((image) => {
      const rect = image.getBoundingClientRect();
      const distance = Math.abs(rect.top + rect.height / 2 - center);
      if (distance < closestDistance) {
        closestDistance = distance;
        closest = Number(image.dataset.pdfPage) || 1;
      }
    });
    setActivePage(root, closest);
  };

  const scrollToPage = (root, page, behavior = "smooth") => {
    const state = ensureViewer(root);
    const { scroller } = getParts(root);
    if (!state || !scroller) return;
    const targetPage = clamp(Number(page) || 1, 1, state.totalPages);
    const target = root.querySelector('[data-pdf-page="' + targetPage + '"]');
    if (!target) return;
    const targetRect = target.getBoundingClientRect();
    const scrollerRect = scroller.getBoundingClientRect();
    const nextTop = scroller.scrollTop + targetRect.top - scrollerRect.top - 12;
    scroller.scrollTo({ top: Math.max(0, nextTop), behavior });
    setActivePage(root, targetPage);
    parentWindow.setTimeout(() => detectActivePage(root), behavior === "smooth" ? 180 : 40);
  };

  const setFocusMode = (root, enabled) => {
    const state = root && root.__ragPdfViewerState;
    if (!state) return;
    const { zoomFocus } = getParts(root);
    state.focusMode = Boolean(enabled);
    root.classList.toggle("is-focus-zoom", state.focusMode);
    if (zoomFocus) {
      zoomFocus.classList.toggle("is-active", state.focusMode);
      zoomFocus.setAttribute("aria-pressed", state.focusMode ? "true" : "false");
    }
  };

  const setZoom = (root, nextZoom) => {
    const state = root && root.__ragPdfViewerState;
    if (!state) return;
    const { images, zoomLabel, zoomOut, zoomIn } = getParts(root);
    captureBaseWidths(root);
    state.zoom = clamp(Number(nextZoom) || 100, 50, 200);
    images.forEach((image) => {
      const baseWidth = Number(image.dataset.pdfBaseWidth || 0) || image.getBoundingClientRect().width || 1;
      image.style.setProperty("width", Math.max(1, Math.round(baseWidth * state.zoom / 100)) + "px", "important");
      image.style.setProperty("max-width", "none", "important");
    });
    if (zoomLabel) zoomLabel.textContent = state.zoom + "%";
    if (zoomOut) zoomOut.disabled = state.zoom <= 50;
    if (zoomIn) zoomIn.disabled = state.zoom >= 200;
    root.classList.toggle("is-zoomed-in", state.zoom > 100);
  };

  const applyZoom = (root, nextZoom) => {
    const state = ensureViewer(root);
    if (!state) return;
    const activePage = state.activePage;
    setZoom(root, nextZoom);
    parentWindow.setTimeout(() => {
      scrollToPage(root, activePage, "auto");
      detectActivePage(root);
    }, 40);
  };

  function ensureViewer(root) {
    if (!root) return null;
    const { scroller, images } = getParts(root);
    if (!scroller || !images.length) return null;
    const state = root.__ragPdfViewerState || {
      activePage: 1,
      zoom: 100,
      focusMode: false,
      scroller: null,
      scrollTicking: false,
      focusClickBound: false,
    };
    state.totalPages = Math.max(1, images.length);
    root.__ragPdfViewerState = state;
    root.dataset.viewerReady = "true";

    if (state.scroller !== scroller) {
      state.scroller = scroller;
      scroller.addEventListener("scroll", () => {
        const currentState = root.__ragPdfViewerState;
        if (!currentState || currentState.scrollTicking) return;
        currentState.scrollTicking = true;
        parentWindow.requestAnimationFrame(() => {
          detectActivePage(root);
          currentState.scrollTicking = false;
        });
      }, { passive: true });
    }

    if (!state.focusClickBound) {
      state.focusClickBound = true;
      scroller.addEventListener("click", (event) => {
        const currentState = ensureViewer(root);
        if (!currentState || !currentState.focusMode) return;
        const pageImage = event.target && event.target.closest ? event.target.closest("[data-pdf-page]") : null;
        if (!pageImage || !scroller.contains(pageImage)) return;
        event.preventDefault();
        const nextZoom = clamp(currentState.zoom + 25, 50, 200);
        if (nextZoom === currentState.zoom) return;
        setActivePage(root, Number(pageImage.dataset.pdfPage) || currentState.activePage);
        setZoom(root, nextZoom);
        parentWindow.setTimeout(() => detectActivePage(root), 80);
      });
    }

    captureBaseWidths(root);
    setActivePage(root, state.activePage);
    setZoom(root, state.zoom);
    setFocusMode(root, state.focusMode);
    return state;
  }

  const resetViewer = (root) => {
    const state = ensureViewer(root);
    if (!state) return;
    state.activePage = 1;
    state.zoom = 100;
    state.focusMode = false;
    setFocusMode(root, false);
    setZoom(root, 100);
    scrollToPage(root, 1, "auto");
    parentWindow.setTimeout(() => detectActivePage(root), 40);
  };

  const initVisibleViewers = () => {
    parentDocument.querySelectorAll(VISIBLE_MODAL_SELECTOR).forEach((modal) => {
      ensureViewer(modal);
      parentWindow.requestAnimationFrame(() => detectActivePage(modal));
    });
  };

  const closeModal = (modal, restoreFocus = true) => {
    if (!modal) return;
    modal.classList.add("is-hidden");
    modal.setAttribute("aria-hidden", "true");
    if (modal.matches(":target")) {
      parentWindow.history.replaceState(
        null,
        "",
        parentWindow.location.pathname + parentWindow.location.search + "#pdf-modal-closed"
      );
    }
    if (!getOpenModals().length) {
      parentDocument.body.classList.remove("pdf-modal-open");
    }
    const lastTrigger = parentWindow.__ragPdfModalLastTrigger;
    if (restoreFocus && lastTrigger && typeof lastTrigger.focus === "function") {
      lastTrigger.focus({ preventScroll: true });
    }
  };

  const openModal = (modal, trigger) => {
    if (!modal) return false;
    getOpenModals().forEach((openModalElement) => {
      if (openModalElement !== modal) closeModal(openModalElement, false);
    });
    parentWindow.__ragPdfModalLastTrigger = trigger || null;
    modal.classList.remove("is-hidden");
    modal.setAttribute("aria-hidden", "false");
    parentDocument.body.classList.add("pdf-modal-open");
    parentWindow.requestAnimationFrame(() => {
      resetViewer(modal);
    });
    const closeButton = modal.querySelector("[data-pdf-modal-close], .pdf-modal-close");
    if (closeButton && typeof closeButton.focus === "function") {
      parentWindow.setTimeout(() => closeButton.focus({ preventScroll: true }), 30);
    }
    return true;
  };

  const openHashTargetModal = () => {
    const hash = parentWindow.location.hash ? parentWindow.location.hash.slice(1) : "";
    if (!hash || hash === "pdf-modal-closed") return;
    const modal = parentDocument.getElementById(decodeURIComponent(hash));
    if (!modal || !modal.matches(".pdf-modal-overlay[data-pdf-viewer-id]")) return;
    if (modal.dataset.pdfClientModal === "true") {
      openModal(modal, null);
      return;
    }
    ensureViewer(modal);
    parentWindow.requestAnimationFrame(() => detectActivePage(modal));
  };

  const handleClick = (event) => {
    const trigger = event.target.closest("[data-pdf-modal-target]");
    if (trigger) {
      const targetId = trigger.getAttribute("data-pdf-modal-target");
      const modal = targetId ? parentDocument.getElementById(targetId) : null;
      if (modal && openModal(modal, trigger)) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      const fallback = trigger.getAttribute("data-pdf-modal-fallback");
      if (fallback) {
        event.preventDefault();
        parentWindow.location.href = fallback;
      }
      return;
    }

    const thumb = event.target.closest("[data-pdf-thumb]");
    if (thumb) {
      const root = getViewerRoot(thumb);
      if (root) {
        event.preventDefault();
        event.stopPropagation();
        scrollToPage(root, Number(thumb.dataset.pdfThumb) || 1, "auto");
      }
      return;
    }

    const pagePrev = event.target.closest("[data-pdf-page-prev]");
    if (pagePrev) {
      const root = getViewerRoot(pagePrev);
      const state = ensureViewer(root);
      if (state) {
        event.preventDefault();
        event.stopPropagation();
        scrollToPage(root, state.activePage - 1, "auto");
      }
      return;
    }

    const pageNext = event.target.closest("[data-pdf-page-next]");
    if (pageNext) {
      const root = getViewerRoot(pageNext);
      const state = ensureViewer(root);
      if (state) {
        event.preventDefault();
        event.stopPropagation();
        scrollToPage(root, state.activePage + 1, "auto");
      }
      return;
    }

    const zoomOut = event.target.closest("[data-pdf-zoom-out]");
    if (zoomOut) {
      const root = getViewerRoot(zoomOut);
      const state = ensureViewer(root);
      if (state) {
        event.preventDefault();
        event.stopPropagation();
        applyZoom(root, state.zoom - 10);
      }
      return;
    }

    const zoomIn = event.target.closest("[data-pdf-zoom-in]");
    if (zoomIn) {
      const root = getViewerRoot(zoomIn);
      const state = ensureViewer(root);
      if (state) {
        event.preventDefault();
        event.stopPropagation();
        applyZoom(root, state.zoom + 10);
      }
      return;
    }

    const zoomFocus = event.target.closest("[data-pdf-zoom-focus]");
    if (zoomFocus) {
      const root = getViewerRoot(zoomFocus);
      const state = ensureViewer(root);
      if (state) {
        event.preventDefault();
        event.stopPropagation();
        setFocusMode(root, !state.focusMode);
      }
      return;
    }

    const closeButton = event.target.closest("[data-pdf-modal-close]");
    if (closeButton) {
      const modal = closeButton.closest(".pdf-modal-overlay");
      event.preventDefault();
      event.stopPropagation();
      closeModal(modal);
      return;
    }

    const openModalOverlay = event.target.classList && event.target.classList.contains("pdf-modal-overlay")
      ? event.target
      : null;
    if (
      openModalOverlay &&
      openModalOverlay.dataset.pdfClientModal === "true" &&
      !openModalOverlay.classList.contains("is-hidden")
    ) {
      event.preventDefault();
      closeModal(openModalOverlay);
    }
  };

  const handleKeydown = (event) => {
    if (event.key !== "Escape") return;
    const openModals = getOpenModals();
    if (!openModals.length) return;
    event.preventDefault();
    closeModal(openModals[openModals.length - 1]);
  };

  if (parentWindow.__ragPdfModalHandlers) {
    parentDocument.removeEventListener("click", parentWindow.__ragPdfModalHandlers.click, true);
    parentDocument.removeEventListener("keydown", parentWindow.__ragPdfModalHandlers.keydown);
    parentWindow.removeEventListener("hashchange", parentWindow.__ragPdfModalHandlers.hashchange);
  }
  parentWindow.__ragPdfModalHandlers = {
    click: handleClick,
    keydown: handleKeydown,
    hashchange: openHashTargetModal,
  };
  parentDocument.addEventListener("click", handleClick, true);
  parentDocument.addEventListener("keydown", handleKeydown);
  parentWindow.addEventListener("hashchange", openHashTargetModal);
  parentDocument.documentElement.dataset.pdfModalControllerVersion = VERSION;

  parentWindow.__ragPdfModalController = {
    ensureViewer,
    resetViewer,
    detectActivePage,
    scrollToPage,
    setZoom,
    setActivePage,
    closeModal,
    openModal,
  };

  parentWindow.requestAnimationFrame(initVisibleViewers);
  parentWindow.setTimeout(initVisibleViewers, 60);
  parentWindow.requestAnimationFrame(openHashTargetModal);
})();
</script>
""",
        height=1,
        width=1,
    )


def render_client_pdf_modals(documents: list[dict[str, Any]], source_section: str) -> None:
    if not documents or get_query_param("view_doc"):
        return

    render_pages = len(documents) <= CLIENT_MODAL_RENDERED_PREVIEW_LIMIT
    seen: set[str] = set()
    for document in documents:
        modal_id = pdf_modal_id(document)
        if modal_id in seen:
            continue
        seen.add(modal_id)
        pdf_path = resolve_source_pdf_path(document)
        render_pdf_modal_shell(
            document,
            pdf_path,
            hidden=True,
            source_section=source_section,
            notice="",
            render_pages=render_pages,
        )
    render_client_pdf_modal_controller()


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


def format_storage_estimate(value_mb: Any) -> str:
    try:
        size_mb = float(value_mb or 0)
    except (TypeError, ValueError):
        size_mb = 0.0
    if size_mb <= 0:
        return "0 KB"
    if size_mb < 1:
        return f"{max(1, round(size_mb * 1024)):,} KB"
    return f"{size_mb:.1f} MB"


def render_documents_section_heading() -> None:
    st.markdown(
        """
<div class="documents-section-head">
  <div class="documents-section-title">Documents</div>
  <div class="documents-section-rule"></div>
  <div class="documents-circuit" aria-hidden="true"></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_documents_upload_card(progress_placeholder=None) -> None:
    upload_icon = load_upload_icon_data_uri("upload_cloud_document_icon.png")
    pdf_icon = load_upload_icon_data_uri("pdf_only_icon.png")
    header_upload_icon = load_header_action_icon_data_uri("upload.png")
    upload_reset = int(st.session_state.get("documents_upload_reset", 0) or 0)
    ingestion_active = bool(st.session_state.get("ingestion_active"))
    with st.container(key="documents_upload_card"):
        st.markdown(
            f"""
<div class="documents-card-title">Upload PDFs</div>
<style>
.st-key-documents_upload_submit button::before {{
  content: "";
  width: 20px;
  height: 20px;
  flex: 0 0 20px;
  background: currentColor;
  -webkit-mask: url("{header_upload_icon}") center / contain no-repeat;
  mask: url("{header_upload_icon}") center / contain no-repeat;
}}
</style>
""",
            unsafe_allow_html=True,
        )
        st.session_state.pop("documents_upload_notice", None)
        with st.container(key="documents_upload_zone"):
            with st.container(key="documents_upload_input_layer"):
                uploaded = st.file_uploader(
                    "Drag PDFs here or browse files",
                    type=["pdf"],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                    key=f"documents_upload_input_{upload_reset}",
                )
            if uploaded:
                total_size = sum(int(getattr(file, "size", 0) or 0) for file in uploaded)
                first_file = uploaded[0]
                extra_count = len(uploaded) - 1
                name = html.escape(str(getattr(first_file, "name", "Selected PDF") or "Selected PDF"))
                size = html.escape(format_file_size(total_size))
                suffix = f" + {extra_count} more" if extra_count else ""
                pending_upload_html = f"""
<div class="documents-upload-pending">
  <div class="documents-upload-file">
    <img src="{pdf_icon}" alt="" aria-hidden="true" />
    <div>
      <div class="documents-upload-file-name" title="{name}{html.escape(suffix)}">{name}{html.escape(suffix)}</div>
      <div class="documents-upload-file-size">{size}</div>
    </div>
  </div>
</div>
"""
            else:
                pending_upload_html = ""
            selected_class = " has-file" if uploaded else ""
            st.markdown(
                f"""
<div class="documents-upload-visual{selected_class}">
  <img class="documents-upload-cloud" src="{upload_icon}" alt="" aria-hidden="true" />
  {pending_upload_html}
  <div class="documents-upload-helper">Drag PDFs here or browse files</div>
</div>
""",
                unsafe_allow_html=True,
            )
            if uploaded:
                action_col, cancel_col = st.columns([1, 0.18], gap="small")
                with action_col:
                    if st.button(
                        "Upload PDFs",
                        key="documents_upload_submit",
                        use_container_width=True,
                        disabled=ingestion_active,
                    ):
                        if not uploaded:
                            st.warning("No PDF selected. Please choose a PDF first.")
                        else:
                            results = upload_and_ingest_files(uploaded, progress_placeholder=progress_placeholder)
                            st.session_state.documents_upload_reset = upload_reset + 1
                            st.rerun()
                with cancel_col:
                    if st.button(
                        "×",
                        key="documents_upload_cancel",
                        help="Cancel upload",
                        use_container_width=True,
                        disabled=ingestion_active,
                    ):
                        st.session_state.documents_upload_reset = upload_reset + 1
                        st.rerun()
        render_upload_badges()


def pipeline_state(step_name: str, events: list[str], results: list[dict[str, Any]]) -> str:
    if results:
        latest_status = str(results[-1].get("status", "")).lower()
        if latest_status in {"indexed", "skipped"}:
            return "complete"
        if latest_status == "failed":
            return "pending"
    event_text = " ".join(events).lower()
    if step_name.lower() in event_text:
        return "active"
    return "pending"


def summarize_ingestion_results(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "indexed": len([result for result in results if str(result.get("status", "")).lower() == "indexed"]),
        "skipped": len([result for result in results if str(result.get("status", "")).lower() == "skipped"]),
        "failed": len([result for result in results if str(result.get("status", "")).lower() == "failed"]),
        "pages": sum(int(result.get("pages") or 0) for result in results if str(result.get("status", "")).lower() == "indexed"),
        "chunks": sum(int(result.get("chunks") or 0) for result in results if str(result.get("status", "")).lower() == "indexed"),
    }


def build_ingestion_progress_rows(
    events: list[str],
    results: list[dict[str, Any]],
    active_stage: str | None = None,
) -> list[tuple[str, str, str]]:
    if active_stage:
        stage_order = {"extract": 0, "chunk": 1, "embed": 2, "sync": 3}
        active_index = stage_order.get(active_stage)
        if active_index is not None:
            labels = ["Extract pages", "Semantic chunking", "Embedding chunks", "ChromaDB sync"]
            active_values = ["In progress", "In progress", "In progress", "Syncing"]
            live_rows: list[tuple[str, str, str]] = []
            for index, label in enumerate(labels):
                if index < active_index:
                    live_rows.append((label, "complete", "Completed"))
                elif index == active_index:
                    live_rows.append((label, "active", active_values[index]))
                else:
                    live_rows.append((label, "pending", "Pending"))
            return live_rows

    summary = summarize_ingestion_results(results)
    if summary["indexed"]:
        return [
            ("Extract pages", "complete", f'{summary["pages"]:,} pages'),
            ("Semantic chunking", "complete", f'{summary["chunks"]:,} chunks'),
            ("Embedding chunks", "complete", "Completed"),
            ("ChromaDB sync", "complete", "Completed"),
        ]
    if summary["skipped"] and not summary["indexed"] and not summary["failed"]:
        return [
            ("Extract pages", "pending", "Duplicate"),
            ("Semantic chunking", "pending", "Skipped"),
            ("Embedding chunks", "pending", "Skipped"),
            ("ChromaDB sync", "pending", "Already indexed"),
        ]
    if summary["failed"] and not summary["indexed"]:
        return [
            ("Extract pages", "pending", "Failed"),
            ("Semantic chunking", "pending", "Failed"),
            ("Embedding chunks", "pending", "Failed"),
            ("ChromaDB sync", "pending", "Failed"),
        ]

    latest = results[-1] if results else {}
    indexed = str(latest.get("status", "")).lower() == "indexed"
    pages = int(latest.get("pages") or 0)
    chunks = int(latest.get("chunks") or 0)

    return [
        ("Extract pages", "complete" if pages else pipeline_state("extracting", events, results), f"{pages:,} pages" if pages else "Idle"),
        ("Semantic chunking", "complete" if chunks else pipeline_state("chunking", events, results), f"{chunks:,} chunks" if chunks else "Idle"),
        ("Embedding chunks", "complete" if indexed else pipeline_state("embedding", events, results), "Completed" if indexed else "Pending"),
        ("ChromaDB sync", "complete" if indexed else pipeline_state("indexed", events, results), "Completed" if indexed else "Pending"),
    ]


def render_ingestion_progress_card_content(
    events: list[str],
    results: list[dict[str, Any]],
    active_stage: str | None = None,
    notice: str = "",
    notice_level: str = "success",
) -> None:
    rows = build_ingestion_progress_rows(events, results, active_stage=active_stage)
    row_html = []
    for label, state, value in rows:
        state_class = {"complete": "is-complete", "active": "is-active"}.get(state, "")
        dot = "&#10003;" if state == "complete" else ""
        row_html.append(
            f'<div class="pipeline-row {state_class}"><span class="pipeline-dot">{dot}</span>'
            f'<span>{html.escape(label)}</span><span class="pipeline-value">{html.escape(value)}</span></div>'
        )
    notice_html = (
        f'<div class="ingestion-progress-notice is-{html.escape(notice_level)}">{html.escape(notice)}</div>'
        if notice
        else ""
    )

    st.markdown(
        f"""
<div class="documents-progress-card">
  <div class="documents-card-title">Ingestion progress</div>
  <div class="progress-pipeline">{''.join(row_html)}</div>
  {notice_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_ingestion_progress_card(active_stage: str | None = None, results: list[dict[str, Any]] | None = None) -> None:
    events = [str(event) for event in st.session_state.ingestion_events[-8:]]
    progress_run = st.session_state.get("ingestion_progress_run") or {}
    is_active = bool(progress_run.get("active"))
    current_active_stage = active_stage
    current_results = list(st.session_state.last_ingestion_results or []) if results is None else list(results)
    if is_active:
        current_active_stage = str(progress_run.get("active_stage") or active_stage or "")
        current_results = list(progress_run.get("results") or [])
    notice = "" if is_active else str(st.session_state.get("ingestion_progress_notice", "") or "")
    notice_level = str(st.session_state.get("ingestion_progress_notice_level", "success") or "success")
    render_ingestion_progress_card_content(
        events,
        current_results,
        active_stage=current_active_stage,
        notice=notice,
        notice_level=notice_level,
    )


def update_ingestion_progress_placeholder(
    placeholder,
    active_stage: str | None = None,
    results: list[dict[str, Any]] | None = None,
) -> None:
    if placeholder is None:
        return
    with placeholder.container():
        render_ingestion_progress_card(active_stage=active_stage, results=results)


def render_documents_metric_cards(stats: dict[str, Any]) -> None:
    total_documents = int(stats.get("total_documents", 0) or 0)
    total_chunks = int(stats.get("total_chunks", 0) or 0)
    storage = format_storage_estimate(stats.get("storage_estimate_mb", 0))
    cards = [
        ("Total Documents", f"{total_documents:,}", "Deduped by SHA-256", "warm", "document"),
        ("Total Chunks", f"{total_chunks:,}", "Stored in ChromaDB", "cool", "layers"),
        ("Storage Estimate", storage, "Source PDFs only", "gold", "database"),
    ]
    card_html = []
    for label, value, helper, tone, icon_name in cards:
        icon_svg = get_status_card_icon_svg(icon_name)
        card_html.append(
            f"""
<div class="documents-metric-card is-{tone}">
  <div class="documents-metric-icon">{icon_svg}</div>
  <div>
    <div class="documents-metric-label">{html.escape(label)}</div>
    <div class="documents-metric-value">{html.escape(value)}</div>
    <div class="documents-metric-helper">{html.escape(helper)}</div>
  </div>
</div>
"""
        )
    st.markdown(f'<div class="documents-metric-grid">{"".join(card_html)}</div>', unsafe_allow_html=True)


def get_selected_document(stats: dict[str, Any]) -> dict[str, Any] | None:
    selected = get_query_param("selected_doc")
    if selected:
        document = find_document_by_hash(stats, selected)
        if document:
            return document
    documents = stats.get("documents", [])
    return documents[0] if documents else None


def render_selected_document_panel(document: dict[str, Any] | None) -> None:
    if not document:
        st.markdown(
            """
<div class="selected-document-card">
  <div class="selected-document-header"><div class="documents-card-title">Selected document</div></div>
  <div class="selected-preview-empty">No indexed documents yet.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    filename = str(document.get("filename", "") or "document")
    file_size = format_file_size(document.get("file_size"))
    pages = int(document.get("pages") or 0)
    chunks = int(document.get("chunks") or 0)
    status = str(document.get("status", "Indexed") or "Indexed").title()
    last_ingested = format_ingested_timestamp(document.get("last_ingested"))
    document_hash = str(document.get("document_hash", "") or "")
    short_hash = document_hash[:12] if document_hash else "n/a"
    chunking = str(document.get("chunking_strategy", "Semantic") or "Semantic").title()
    location = get_document_location_label(document)
    target = quote(document_hash or filename, safe="")
    delete_href = f"?delete_doc={target}&selected_doc={target}"
    pdf_path = resolve_source_pdf_path(document)
    preview_html = '<div class="selected-preview-empty">Source PDF unavailable.</div>'
    if pdf_path:
        try:
            stat = pdf_path.stat()
            pages_rendered = render_pdf_page_images(str(pdf_path), stat.st_size, stat.st_mtime_ns, thumb_scale=0.42)
            if pages_rendered:
                preview_html = f'<img src="{pages_rendered[0]["thumb_uri"]}" alt="First page preview of {html.escape(filename)}" />'
        except Exception:
            preview_html = '<div class="selected-preview-empty">Preview unavailable.</div>'

    metadata_rows = [
        ("folder", "Location", location),
        ("page", "Pages", f"{pages:,}"),
        ("stack", "Chunks", f"{chunks:,} ({chunking})"),
        ("check", "Status", status),
        ("date", "Last ingested", last_ingested),
        ("hash", "Hash (SHA-256)", short_hash),
        ("split", "Chunking", chunking),
        ("box", "Vector store", "ChromaDB"),
    ]
    row_html = "".join(
        f'<div class="selected-meta-row"><span class="selected-meta-icon">{html.escape(icon[:1].upper())}</span>'
        f'<span>{html.escape(label)}</span><span class="selected-meta-value" title="{html.escape(value)}">{html.escape(value)}</span></div>'
        for icon, label, value in metadata_rows
    )

    st.markdown(
        f"""
<div class="selected-document-card">
  <div class="selected-document-header">
    <div class="documents-card-title">Selected document</div>
    <a class="selected-close" href="?" title="Clear selected document">&times;</a>
  </div>
  <div class="selected-doc-identity">
    <div class="selected-pdf-mark">PDF</div>
    <div>
      <div class="selected-doc-name">{html.escape(filename)}</div>
      <div class="selected-doc-size">{html.escape(file_size)}</div>
    </div>
  </div>
  <div>{row_html}</div>
  <div class="selected-preview">
    <div class="selected-preview-head"><span>Page preview</span><span>1 / {max(1, pages):,}</span></div>
    {preview_html}
  </div>
  <a class="selected-delete" href="{delete_href}">{_trash_svg_inline()}<span>Delete document</span></a>
  <div class="selected-delete-copy">Removes the source PDF and indexed chunks.</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _trash_svg_inline() -> str:
    return (
        '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true">'
        '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 14h10l1-14M9 7V4h6v3" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def render_documents_screen(stats: dict[str, Any]) -> None:
    render_documents_section_heading()

    notice = st.session_state.pop("document_action_notice", None)
    if notice:
        level, message = notice
        if level == "success":
            st.success(message)
        elif level == "error":
            st.error(message)
        else:
            st.info(message)

    delete_target = get_query_param("delete_doc")
    dialog_available = callable(getattr(st, "dialog", None))
    delete_document = find_document_by_hash(stats, delete_target) if delete_target else None
    if delete_document and not dialog_available:
        render_delete_confirmation_inline(delete_document, get_chroma_collection())

    documents = stats.get("documents", [])
    main_col, selected_col = st.columns([3.15, 1], gap="large")
    with main_col:
        upload_col, status_col = st.columns([1.12, 1.08], gap="medium")
        with status_col:
            progress_placeholder = st.empty()
            update_ingestion_progress_placeholder(progress_placeholder)
        with upload_col:
            render_documents_upload_card(progress_placeholder=progress_placeholder)

        render_documents_metric_cards(stats)
        render_document_table(
            documents,
            title="Document library",
            source_section="Documents",
            enable_delete=True,
            info_copy="Deleting a document removes its uploaded PDF and all ChromaDB chunks for that SHA-256 hash.",
        )

    with selected_col:
        render_selected_document_panel(get_selected_document(stats))

    render_client_pdf_modals(documents, "Documents")


def render_ingestion_status(stats: dict[str, Any]) -> None:
    st.markdown('<div class="ingestion-status-title">Ingestion status</div>', unsafe_allow_html=True)
    render_ingestion_status_cards(stats)
    documents = stats.get("documents", [])
    render_document_table(documents, source_section="Ingestion status")
    render_client_pdf_modals(documents, "Ingestion status")


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


def render_collection_stats_panel(stats: dict[str, Any]) -> None:
    documents = stats.get("documents", []) or []
    total_documents = int(stats.get("total_documents", 0) or 0)
    total_chunks = int(stats.get("total_chunks", 0) or 0)
    storage_mb = float(stats.get("storage_estimate_mb", 0) or 0)
    max_chunks = max((int(document.get("chunks", 0) or 0) for document in documents), default=0)

    document_rows = []
    for document in documents[:5]:
        filename = str(document.get("filename", "") or "Unknown document")
        pages = int(document.get("pages", 0) or 0)
        chunks = int(document.get("chunks", 0) or 0)
        status = str(document.get("status", "Indexed") or "Indexed").title()
        last_ingested = format_ingested_timestamp(document.get("last_ingested"))
        file_size = format_file_size(document.get("file_size"))
        details = f"{pages:,} pages &middot; {chunks:,} chunks"
        if file_size != "Unknown":
            details = f"{details} &middot; {html.escape(file_size)}"
        document_rows.append(
            f"""
      <div class="collection-doc-row">
        <div>
          <div class="collection-doc-name" title="{html.escape(filename)}">{html.escape(filename)}</div>
          <div class="collection-doc-meta">{details}</div>
        </div>
        <span class="collection-mini-pill">{html.escape(status)}</span>
        <div class="collection-doc-meta">{html.escape(str(document.get("chunking_strategy", "semantic")).title())}</div>
        <div class="collection-doc-date">{html.escape(last_ingested)}</div>
      </div>
"""
        )

    if document_rows:
        documents_html = f'<div class="collection-doc-list">{"".join(document_rows)}</div>'
    else:
        documents_html = '<div class="collection-empty">No indexed documents yet. Upload and ingest PDFs to populate collection stats.</div>'

    bar_rows = []
    for document in documents[:5]:
        filename = str(document.get("filename", "") or "Unknown document")
        chunks = int(document.get("chunks", 0) or 0)
        width = int((chunks / max_chunks) * 100) if max_chunks else 0
        bar_rows.append(
            f"""
      <div class="collection-bar-row">
        <div class="collection-bar-label" title="{html.escape(filename)}">{html.escape(filename)}</div>
        <div class="collection-bar-track"><span class="collection-bar-fill" style="width: {width}%"></span></div>
        <div class="collection-bar-value">{chunks:,}</div>
      </div>
"""
        )
    bars_html = "".join(bar_rows) if bar_rows else '<div class="collection-empty">Chunk distribution will appear after ingestion.</div>'

    st.markdown(
        f"""
<div class="collection-stats-card">
  <div class="collection-stats-header">
    <div>
      <div class="collection-stats-title">Collection stats</div>
      <div class="collection-stats-copy">Current local ChromaDB index health and document coverage.</div>
    </div>
    <div class="collection-stats-badge">ChromaDB</div>
  </div>
  <div class="collection-stats-grid">
    <div class="collection-stat-tile">
      <div class="collection-stat-label">Documents indexed</div>
      <div class="collection-stat-value">{total_documents:,}</div>
      <div class="collection-stat-helper">Deduped source PDFs</div>
    </div>
    <div class="collection-stat-tile">
      <div class="collection-stat-label">Chunks stored</div>
      <div class="collection-stat-value">{total_chunks:,}</div>
      <div class="collection-stat-helper">Retrieval-ready passages</div>
    </div>
    <div class="collection-stat-tile">
      <div class="collection-stat-label">Storage estimate</div>
      <div class="collection-stat-value">{storage_mb:.2f} MB</div>
      <div class="collection-stat-helper">Source PDF footprint</div>
    </div>
  </div>
  <div class="collection-stats-section-title">Indexed documents</div>
  {documents_html}
  <div class="collection-stats-section-title">Chunks by document</div>
  <div class="collection-bars">{bars_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_settings_screen(stats: dict[str, Any]) -> None:
    st.markdown('<div class="section-title">Settings / Debug</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="section-card">
  <div class="section-title">Environment</div>
  <p>Secrets are loaded server-side from .env or environment variables. API keys are never rendered in the UI.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    render_collection_stats_panel(stats)
    with st.expander("Raw stats payload", expanded=False):
        st.json(stats, expanded=1)

    st.warning("Resetting the vector database deletes all indexed chunks. Source PDFs are not deleted.")
    confirmed = st.checkbox("I understand this will delete the local ChromaDB collection.")
    if st.button("Reset vector DB", disabled=not confirmed):
        reset_vector_db()
        invalidate_collection_stats_cache()
        st.session_state.messages = []
        st.success("Vector database reset.")
        st.rerun()

    if st.button("Delete uploaded PDFs", disabled=not confirmed):
        shutil.rmtree(UPLOADED_DOCS_DIR, ignore_errors=True)
        Path(UPLOADED_DOCS_DIR).mkdir(parents=True, exist_ok=True)
        invalidate_collection_stats_cache()
        st.success("uploaded_docs/ cleared.")
        st.rerun()


def main() -> None:
    init_state()
    inject_custom_css()

    collection = get_chroma_collection()
    stats = get_cached_collection_stats(collection)
    handle_pdf_reingest_action(stats)
    handle_document_delete_action(stats)
    consume_navigation_query_param()
    apply_modal_source_section()
    section = render_sidebar(stats)
    previous_section = st.session_state.get("_rendered_nav_section")
    section_changed = previous_section != section
    st.session_state["_rendered_nav_section"] = section
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
        stats = get_cached_collection_stats(collection, force_refresh=True)

    if section == "App overview":
        question = render_overview(stats, st.session_state.messages)
        if question:
            answer_question(question)
        recent_docs, recent_convos = st.columns(2)
        with recent_docs:
            recent_documents = stats.get("documents", [])[:6]
            render_document_table(recent_documents, title="Recent documents", source_section="App overview")
            render_client_pdf_modals(recent_documents, "App overview")
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

    selected_document = get_pdf_modal_document(stats)
    if selected_document:
        render_pdf_preview_dialog(selected_document)
    elif get_query_param("view_doc"):
        clear_modal_query_params(rerun=True)
    elif section_changed or section == "Settings / Debug":
        scroll_page_to_top()


if __name__ == "__main__":
    main()
