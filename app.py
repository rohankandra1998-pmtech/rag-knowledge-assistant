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


def get_query_param(name: str) -> str:
    try:
        selected = st.query_params[name]
    except Exception:
        selected = st.query_params.get(name, "")
    if isinstance(selected, list):
        selected = selected[0] if selected else ""
    return str(selected or "").strip()


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

    document = next(
        (
            item
            for item in stats.get("documents", [])
            if unquote(target) in {str(item.get("document_hash", "")), str(item.get("filename", ""))}
        ),
        None,
    )
    if not document:
        st.query_params.clear()
        return

    pdf_path = resolve_source_pdf_path(document)
    if not pdf_path:
        st.session_state.pdf_modal_notice = "Source PDF not found in docs/ or uploaded_docs/."
    else:
        collection = get_chroma_collection()
        result = ingest_pdf(pdf_path, collection=collection, force=True)
        st.session_state.last_ingestion_results = [result]
        add_ingestion_event(f"Re-ingested {pdf_path.name}.")
        st.session_state.pdf_modal_notice = f"Re-ingested {pdf_path.name}."

    try:
        st.query_params["view_doc"] = target
        if source_section in NAV_SECTIONS:
            st.query_params["from_section"] = source_section
        del st.query_params["reingest_doc"]
    except Exception:
        st.query_params.clear()
        st.query_params["view_doc"] = target
        if source_section in NAV_SECTIONS:
            st.query_params["from_section"] = source_section
    st.rerun()


def pdf_data_uri(pdf_path: Path) -> str:
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


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
  const root = window.parent.document.querySelector('[data-pdf-viewer-id="' + viewerId + '"]');
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

  window.parent.document.addEventListener("keydown", function(event) {{
    if (event.key === "Escape" && focusMode) {{
      setFocusMode(false);
    }}
  }});

  let ticking = false;
  scroller.addEventListener("scroll", function() {{
    if (ticking) return;
    ticking = true;
    window.parent.requestAnimationFrame(function() {{
      detectActivePage();
      ticking = false;
    }});
  }}, {{ passive: true }});

  window.parent.requestAnimationFrame(function() {{
    captureBaseWidths();
    applyZoom(100);
    setFocusMode(false);
    detectActivePage();
  }});
}})();
</script>
"""
    st.iframe(script, height=1, width=1)


def render_pdf_modal_shell(document: dict[str, Any], pdf_path: Path | None) -> None:
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
    rendered_pages, render_warning = get_rendered_pdf_pages(pdf_path) if pdf_path else ([], "")
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
    source_section = get_query_param("from_section")
    if source_section not in NAV_SECTIONS:
        source_section = str(st.session_state.get("nav_section", "App overview"))
    close_href = f"?section={quote(source_section, safe='')}"
    open_pdf_html = (
        f'<a class="pdf-modal-action primary" href="{escaped_pdf_uri}" target="_blank" download="{html.escape(filename)}">Open full PDF</a>'
        if pdf_path
        else '<span class="pdf-modal-action primary is-disabled">Open full PDF</span>'
    )
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
    modal_html = (
        f'<div class="pdf-modal-overlay" data-pdf-viewer-id="{html.escape(viewer_id, quote=True)}">'
        '<div class="pdf-modal-dialog"><div class="pdf-modal-shell">'
        '<div class="pdf-modal-header"><div class="pdf-modal-heading">'
        '<div class="pdf-modal-badge">PDF</div>'
        f'<div class="pdf-modal-title" title="{html.escape(filename)}">{html.escape(filename)}</div>'
        f'</div><a class="pdf-modal-close" href="{close_href}" target="_self" aria-label="Close">&times;</a></div>'
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
    if rendered_pages:
        render_pdf_viewer_script(viewer_id, total_preview_pages)


def render_pdf_preview_dialog(document: dict[str, Any]) -> None:
    pdf_path = resolve_source_pdf_path(document)
    render_pdf_modal_shell(document, pdf_path)


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

    render_document_table(stats.get("documents", []), source_section="Documents")


def render_ingestion_status(stats: dict[str, Any]) -> None:
    st.markdown('<div class="ingestion-status-title">Ingestion status</div>', unsafe_allow_html=True)
    render_ingestion_status_cards(stats)
    render_document_table(stats.get("documents", []), source_section="Ingestion status")


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
  <p>Secrets are loaded server-side from .env or environment variables. API keys are never rendered in the UI.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("**Collection stats**")
    st.json(stats, expanded=1)

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
    handle_pdf_reingest_action(stats)
    query_section = get_query_param("section") or get_query_param("from_section")
    if query_section in NAV_SECTIONS:
        st.session_state.nav_section = query_section
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
        stats = get_collection_stats(collection)

    if section == "App overview":
        question = render_overview(stats, st.session_state.messages)
        if question:
            answer_question(question)
        recent_docs, recent_convos = st.columns(2)
        with recent_docs:
            render_document_table(stats.get("documents", [])[:6], title="Recent documents", source_section="App overview")
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
    elif section_changed or section == "Settings / Debug":
        scroll_page_to_top()


if __name__ == "__main__":
    main()
