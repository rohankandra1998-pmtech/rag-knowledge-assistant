from __future__ import annotations

import base64
import html
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from rag_utils import CHAT_MODEL, EMBEDDING_MODEL


PALETTE = {
    "terracotta": "#C8472C",
    "brown": "#412A1E",
    "gold": "#F8DE3C",
    "off_white": "#FEFEFE",
    "sky": "#58ACF4",
    "blue": "#105EDD",
    "navy": "#0B3075",
}

APP_ICON_PATH = Path(__file__).parent / "assets" / "rag-app-icon-tight.png"
SIDEBAR_ICON_DIR = Path(__file__).parent / "assets" / "sidebar-icons"
SIDEBAR_NAV_ITEMS = [
    {"label": "App overview", "icon": "App_Overview_Icon.png"},
    {"label": "Chat / Answer", "icon": "Chat_Answer_Icon.png"},
    {"label": "Documents", "icon": "Documents_Icon.png"},
    {"label": "Ingestion status", "icon": "Ingestion_Status_Icon.png"},
    {"label": "Models", "icon": "Models_Icon.png"},
    {"label": "Example questions", "icon": "Example_Questions_Icon.png"},
    {"label": "Settings / Debug", "icon": "Settings_Debug_Icon.png"},
]


@st.cache_data(show_spinner=False)
def _load_app_icon_data_uri() -> str:
    encoded = base64.b64encode(APP_ICON_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_data(show_spinner=False)
def _load_sidebar_icon_data_uri(filename: str) -> str:
    icon_path = SIDEBAR_ICON_DIR / filename
    encoded = base64.b64encode(icon_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _format_sidebar_nav_label(label: str) -> str:
    icon_by_label = {item["label"]: item["icon"] for item in SIDEBAR_NAV_ITEMS}
    icon_uri = _load_sidebar_icon_data_uri(icon_by_label[label])
    return f"![]({icon_uri}) {label}"


def inject_custom_css() -> None:
    st.markdown(
        """
<style>
:root {
  --terracotta: #C8472C;
  --brown: #412A1E;
  --gold: #F8DE3C;
  --off-white: #FEFEFE;
  --sky: #58ACF4;
  --blue: #105EDD;
  --navy: #0B3075;
  --ink: #111936;
  --muted: #64708A;
  --line: #DFE7F3;
  --card: rgba(255,255,255,0.94);
  --shadow: 0 18px 45px rgba(11, 48, 117, 0.10);
}

html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
  background:
    radial-gradient(circle at 85% 10%, rgba(88, 172, 244, 0.12), transparent 26%),
    linear-gradient(180deg, #FEFEFE 0%, #F6F9FD 100%);
  color: var(--ink);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.35rem 1.75rem 3rem; max-width: 1440px; }

[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(65,42,30,0.98) 0%, #24170F 100%);
  border-right: 1px solid rgba(248, 222, 60, 0.16);
}
[data-testid="stSidebar"] * { color: #fff; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: rgba(255,255,255,0.78); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stRadio p { color: #fff !important; }

.sidebar-logo {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  padding: 1rem 0.55rem 1.3rem;
}
.sidebar-logo .app-logo-img {
  width: 64px;
  height: 64px;
  flex: 0 0 64px;
  display: block;
  object-fit: contain;
}
.sidebar-title {
  font-size: 1.03rem;
  font-weight: 800;
  line-height: 1.25;
}
.sidebar-subtitle {
  font-size: 0.78rem;
  color: rgba(255,255,255,0.68);
}

[data-testid="stSidebar"] [role="radiogroup"] {
  display: flex;
  flex-direction: column;
  gap: 0.52rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [role="radio"] {
  border-radius: 10px;
  padding: 0.62rem 0.78rem;
  min-height: 50px;
  align-items: center;
  width: 100%;
  transition: background 140ms ease, box-shadow 140ms ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child,
[data-testid="stSidebar"] [role="radio"] > div:first-child {
  display: none !important;
  width: 0 !important;
  min-width: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
  display: none !important;
}
[data-testid="stSidebar"] [role="radio"] [data-testid="stMarkdownContainer"] p {
  display: flex;
  align-items: center;
  line-height: 1.1;
}
[data-testid="stSidebar"] [role="radio"] [data-testid="stMarkdownContainer"] p {
  gap: 14px;
  margin: 0;
}
[data-testid="stSidebar"] [role="radio"] img {
  width: 32px;
  height: 32px;
  object-fit: contain;
  flex: 0 0 32px;
  display: inline-block;
  background: transparent !important;
  vertical-align: middle;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]:checked),
[data-testid="stSidebar"] [aria-checked="true"] {
  background: linear-gradient(135deg, #C8472C, #E05B36);
  box-shadow: 0 12px 25px rgba(200,71,44,0.28);
}

.side-card {
  margin: 0.8rem 0.25rem;
  padding: 0.85rem;
  border: 1px solid rgba(255,255,255,0.16);
  border-radius: 12px;
  background: rgba(255,255,255,0.055);
}
.side-card-title {
  font-weight: 800;
  margin-bottom: 0.5rem;
}
.side-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  font-size: 0.82rem;
  color: rgba(255,255,255,0.76);
  padding: 0.16rem 0;
}
.side-pill {
  color: #0B3075 !important;
  background: #FEFEFE;
  border-radius: 999px;
  padding: 0.12rem 0.5rem;
  font-size: 0.72rem;
  font-weight: 800;
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  margin: 0.35rem 0 1.35rem;
}
.app-title {
  color: var(--navy);
  font-size: 2rem;
  line-height: 1.05;
  font-weight: 900;
  margin: 0;
}
.app-subtitle {
  color: #405072;
  font-size: 1rem;
  margin-top: 0.35rem;
}
.st-key-header_actions {
  justify-content: flex-end;
  padding-top: 0.35rem;
}
.st-key-header_actions [data-testid="stButton"] {
  width: auto !important;
  flex: 0 0 auto;
}
.st-key-header_actions div.stButton > button {
  height: 2.9rem;
  min-height: 2.9rem;
  padding: 0 1rem;
  border-radius: 9px;
  font-size: 0.92rem;
  font-weight: 800;
  white-space: nowrap;
  box-shadow: 0 12px 28px rgba(11,48,117,0.08);
}
.st-key-header_actions div.stButton > button[kind="primary"] {
  box-shadow: 0 14px 28px rgba(16,94,221,0.18);
}
.st-key-header_actions div.stButton > button p {
  white-space: nowrap;
}

.section-card, .hero-card, .metric-card, .answer-card, .source-card, .debug-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: var(--shadow);
}
.hero-card {
  position: relative;
  overflow: hidden;
  padding: 2.4rem 2rem;
  min-height: 250px;
  background:
    linear-gradient(90deg, rgba(254,254,254,0.96), rgba(241,247,255,0.92)),
    repeating-linear-gradient(0deg, transparent 0, transparent 23px, rgba(88,172,244,0.08) 24px),
    repeating-linear-gradient(90deg, transparent 0, transparent 23px, rgba(88,172,244,0.08) 24px);
}
.hero-card:after {
  content: "";
  position: absolute;
  width: 320px;
  height: 320px;
  right: -70px;
  top: -30px;
  background:
    radial-gradient(circle at 40% 35%, rgba(16,94,221,0.28), transparent 4px),
    radial-gradient(circle at 70% 50%, rgba(200,71,44,0.35), transparent 4px),
    radial-gradient(circle at 55% 72%, rgba(248,222,60,0.45), transparent 5px),
    linear-gradient(135deg, rgba(88,172,244,0.10), rgba(16,94,221,0.04));
  border: 1px solid rgba(16,94,221,0.10);
  border-radius: 50%;
  opacity: 0.78;
}
.hero-title {
  position: relative;
  z-index: 1;
  color: var(--navy);
  font-size: 2.65rem;
  line-height: 1.08;
  font-weight: 900;
  margin: 0 0 0.55rem;
}
.hero-copy {
  position: relative;
  z-index: 1;
  color: #405072;
  font-size: 1.05rem;
  max-width: 600px;
}

.metric-card {
  padding: 1.1rem;
  min-height: 128px;
}
.metric-label { color: #405072; font-size: 0.86rem; font-weight: 700; }
.metric-value { color: var(--navy); font-size: 1.95rem; font-weight: 900; margin-top: 0.15rem; }
.metric-delta { font-size: 0.78rem; margin-top: 0.3rem; font-weight: 800; }
.delta-warm { color: var(--terracotta); }
.delta-cool { color: var(--blue); }
.delta-gold { color: #A87400; }

.section-title {
  color: var(--navy);
  font-size: 1.1rem;
  font-weight: 900;
  margin-bottom: 0.65rem;
}
.section-card {
  padding: 1rem;
  margin: 1rem 0;
}
.workflow {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1rem;
}
.workflow-step {
  padding: 1rem;
  border: 1px solid #E4ECF7;
  border-radius: 14px;
  background: #fff;
}
.step-index {
  display: inline-flex;
  width: 24px;
  height: 24px;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: var(--navy);
  color: #fff;
  font-size: 0.75rem;
  font-weight: 900;
  margin-bottom: 0.5rem;
}

.chat-user {
  margin-left: auto;
  max-width: 72%;
  background: #ECF4FF;
  border: 1px solid #CFE1FB;
  color: var(--navy);
  border-radius: 16px 16px 4px 16px;
  padding: 0.9rem 1rem;
  font-weight: 700;
}
.answer-card {
  padding: 1rem 1.1rem;
  margin: 0.75rem 0 1rem;
}
.assistant-label {
  color: var(--navy);
  font-weight: 900;
  margin-bottom: 0.55rem;
}
.answer-body {
  color: #1E2A4A;
  line-height: 1.68;
  white-space: pre-wrap;
}
.citation-pill {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  border: 1px solid #BBD6FF;
  background: #E9F3FF;
  color: var(--blue);
  padding: 0.05rem 0.42rem;
  font-size: 0.78rem;
  font-weight: 800;
}

.source-card {
  padding: 0.9rem;
  margin: 0.55rem 0;
}
.source-title { color: var(--navy); font-weight: 900; }
.source-meta { color: #64708A; font-size: 0.82rem; margin-top: 0.2rem; }
.score-bar {
  height: 7px;
  background: #E7EEF8;
  border-radius: 999px;
  overflow: hidden;
  margin-top: 0.55rem;
}
.score-bar > span {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #58ACF4, #105EDD);
}

.doc-table-card {
  margin: 1rem 0;
  overflow: hidden;
  border: 1px solid rgba(16, 94, 221, 0.16);
  border-radius: 18px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,251,255,0.96)),
    var(--card);
  box-shadow: 0 18px 46px rgba(11, 48, 117, 0.12);
}
.doc-table-header {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: flex-start;
  padding: 1rem 1.1rem 0.85rem;
  border-bottom: 1px solid #E3ECF8;
  background:
    radial-gradient(circle at 92% -20%, rgba(88,172,244,0.20), transparent 34%),
    linear-gradient(90deg, #FFFFFF, #F6FAFF);
}
.doc-table-title {
  color: var(--navy);
  font-size: 1.08rem;
  font-weight: 900;
}
.doc-table-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 0.45rem;
}
.doc-summary-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  border: 1px solid #D7E6FA;
  border-radius: 999px;
  background: #FFFFFF;
  color: #405072;
  padding: 0.2rem 0.52rem;
  font-size: 0.74rem;
  font-weight: 800;
}
.doc-summary-pill strong { color: var(--navy); }
.doc-table-actions {
  display: flex;
  gap: 0.42rem;
  flex-shrink: 0;
}
.doc-icon-btn {
  border: 1px solid #CFE1FB;
  border-radius: 10px;
  background: #FFFFFF;
  color: var(--navy);
  min-width: 2.15rem;
  height: 2.15rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.74rem;
  font-weight: 900;
  box-shadow: 0 8px 20px rgba(16,94,221,0.08);
}
.doc-table-scroll {
  overflow-x: auto;
}
.doc-table-grid {
  min-width: 980px;
}
.doc-table-head,
.doc-table-row {
  display: grid;
  grid-template-columns: minmax(290px, 1.75fr) 80px minmax(145px, 0.8fr) 112px minmax(170px, 0.95fr) 118px 130px;
  align-items: center;
}
.doc-table-head {
  padding: 0 1rem;
  min-height: 48px;
  background: #F7FAFE;
  color: #62708B;
  font-size: 0.76rem;
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: 0.02em;
}
.doc-table-row {
  min-height: 74px;
  padding: 0 1rem;
  border-top: 1px solid #E8EFF8;
  background: rgba(255,255,255,0.84);
}
.doc-table-row:nth-child(even) {
  background: rgba(248,251,255,0.86);
}
.doc-table-row:hover {
  background: #F2F8FF;
}
.doc-cell {
  min-width: 0;
  padding: 0.55rem 0.65rem 0.55rem 0;
  color: #17233F;
}
.doc-main {
  display: flex;
  align-items: center;
  gap: 0.72rem;
  min-width: 0;
}
.doc-file-icon {
  width: 42px;
  height: 48px;
  flex: 0 0 42px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  position: relative;
  border: 1px solid rgba(200,71,44,0.36);
  border-radius: 10px;
  background: linear-gradient(180deg, #FFF7F4, #FFECE6);
  color: var(--terracotta);
  font-size: 0.68rem;
  font-weight: 950;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.72);
}
.doc-file-icon:after {
  content: "";
  position: absolute;
  right: 6px;
  top: 6px;
  width: 9px;
  height: 9px;
  border-top: 2px solid rgba(200,71,44,0.62);
  border-right: 2px solid rgba(200,71,44,0.62);
}
.doc-file-text {
  min-width: 0;
}
.doc-file-name {
  overflow: hidden;
  color: var(--navy);
  font-size: 0.9rem;
  font-weight: 900;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.doc-file-meta {
  margin-top: 0.18rem;
  color: #71809A;
  font-size: 0.76rem;
  font-weight: 700;
}
.doc-num {
  color: #1D2947;
  font-size: 0.94rem;
  font-weight: 900;
}
.chunk-cell {
  display: flex;
  flex-direction: column;
  gap: 0.32rem;
}
.chunk-meter {
  width: min(100%, 112px);
  height: 7px;
  overflow: hidden;
  border-radius: 999px;
  background: #E4EDF9;
}
.chunk-meter span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #58ACF4, #105EDD);
}
.status-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: fit-content;
  max-width: 100%;
  border-radius: 999px;
  padding: 0.25rem 0.62rem;
  font-size: 0.74rem;
  font-weight: 900;
  white-space: nowrap;
}
.status-indexed {
  border: 1px solid rgba(40, 143, 71, 0.24);
  background: #EAF8EF;
  color: #1D7F3B;
}
.status-failed {
  border: 1px solid rgba(200,71,44,0.25);
  background: #FFF0EC;
  color: var(--terracotta);
}
.status-skipped,
.status-pending {
  border: 1px solid rgba(248,222,60,0.45);
  background: #FFFBE6;
  color: #8A6500;
}
.hash-chip {
  display: inline-block;
  max-width: 98px;
  overflow: hidden;
  border: 1px solid #D7E6FA;
  border-radius: 9px;
  background: #FFFFFF;
  color: #405072;
  padding: 0.24rem 0.45rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.72rem;
  font-weight: 800;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.doc-row-actions {
  display: flex;
  gap: 0.36rem;
}
.tiny-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 42px;
  height: 30px;
  border: 1px solid #CFE1FB;
  border-radius: 9px;
  background: #FFFFFF;
  color: var(--blue);
  font-size: 0.72rem;
  font-weight: 900;
  white-space: nowrap;
}
.tiny-action.alt {
  color: var(--navy);
}
.doc-info-strip {
  margin: 0 1rem 1rem;
  border: 1px solid #D7E6FA;
  border-radius: 14px;
  background: linear-gradient(90deg, #EFF7FF, #FFFFFF);
  color: #405072;
  padding: 0.78rem 0.9rem;
  font-size: 0.84rem;
  font-weight: 750;
}

.debug-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.8rem;
}
.debug-card {
  padding: 0.9rem;
  min-height: 118px;
}
.debug-label { font-weight: 900; color: var(--navy); font-size: 0.86rem; }
.debug-value { color: #405072; font-size: 0.86rem; margin-top: 0.5rem; }

.upload-zone {
  border: 1.5px dashed #B7D1F8;
  background: linear-gradient(180deg, #FFFFFF, #F6FAFF);
  border-radius: 16px;
  padding: 2rem 1rem;
  text-align: center;
}
.upload-glyph {
  width: 72px;
  height: 72px;
  display: inline-flex;
  justify-content: center;
  align-items: center;
  border-radius: 24px;
  border: 1px solid #BBD6FF;
  background: #ECF4FF;
  color: var(--blue);
  font-size: 1.35rem;
  font-weight: 900;
  margin-bottom: 0.85rem;
}
.empty-state, .error-state {
  border-radius: 16px;
  padding: 1rem;
  border: 1px solid #DFE7F3;
  background: #fff;
}
.error-state {
  border-color: rgba(200,71,44,0.35);
  background: rgba(200,71,44,0.06);
}

div.stButton > button {
  border-radius: 10px;
  border: 1px solid #CFE1FB;
  font-weight: 800;
  min-height: 2.65rem;
}
div.stButton > button[kind="primary"] {
  background: var(--blue);
  color: #fff;
  border-color: var(--blue);
}
[data-testid="stFileUploader"] {
  border-radius: 16px;
}

@media (max-width: 980px) {
  .app-header { flex-direction: column; }
  .st-key-header_actions {
    justify-content: flex-start;
    padding-top: 0;
  }
  .hero-title { font-size: 2.05rem; }
  .workflow, .debug-grid { grid-template-columns: 1fr 1fr; }
  .chat-user { max-width: 92%; }
  .doc-table-header { flex-direction: column; }
}
@media (max-width: 620px) {
  .workflow, .debug-grid { grid-template-columns: 1fr; }
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_sidebar(stats: dict[str, Any]) -> str:
    with st.sidebar:
        app_icon_uri = _load_app_icon_data_uri()
        st.markdown(
            f"""
<div class="sidebar-logo">
  <img class="app-logo-img" src="{app_icon_uri}" alt="RAG Knowledge Assistant logo" />
  <div>
    <div class="sidebar-title">RAG Knowledge<br/>Assistant</div>
    <div class="sidebar-subtitle">Grounded document Q&A</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        nav_items = [item["label"] for item in SIDEBAR_NAV_ITEMS]
        section = st.radio(
            "Navigation",
            nav_items,
            key="nav_section",
            format_func=_format_sidebar_nav_label,
            label_visibility="collapsed",
        )

        st.markdown(
            f"""
<div class="side-card">
  <div class="side-card-title">Models</div>
  <div class="side-row"><span>LLM</span><span class="side-pill">{CHAT_MODEL}</span></div>
  <div class="side-row"><span>Embeddings</span><span class="side-pill">3-small</span></div>
</div>
<div class="side-card">
  <div class="side-card-title">Ingestion status</div>
  <div class="side-row"><span>Documents indexed</span><strong>{stats.get("total_documents", 0)}</strong></div>
  <div class="side-row"><span>Chunks stored</span><strong>{stats.get("total_chunks", 0):,}</strong></div>
</div>
""",
            unsafe_allow_html=True,
        )

        recent_docs = stats.get("documents", [])[:3]
        if recent_docs:
            doc_rows = "".join(
                f'<div class="side-row"><span>{html.escape(doc["filename"][:26])}</span><span>{doc["chunks"]:,}</span></div>'
                for doc in recent_docs
            )
            st.markdown(
                f'<div class="side-card"><div class="side-card-title">Recent documents</div>{doc_rows}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            """
<div class="side-card">
  <div class="side-card-title">Example questions</div>
  <div class="side-row"><span>Summarize a report</span></div>
  <div class="side-row"><span>Compare policies</span></div>
  <div class="side-row"><span>Find risks</span></div>
</div>
<div class="side-card">
  <div class="side-row"><strong>Admin</strong><span class="side-pill">Local</span></div>
  <div class="side-row"><span>portfolio-ready build</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

    return section


def render_header() -> dict[str, bool]:
    left, right = st.columns([1.2, 1], gap="large", vertical_alignment="top")
    with left:
        st.markdown(
            """
<div class="app-header">
  <div>
    <h1 class="app-title">RAG Knowledge Assistant</h1>
    <div class="app-subtitle">Conversational document Q&amp;A with citations</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        with st.container(
            key="header_actions",
            horizontal=True,
            horizontal_alignment="right",
            vertical_alignment="top",
            gap="small",
        ):
            upload_clicked = st.button(
                "Upload PDFs",
                key="header_upload",
                icon=":material/upload:",
                width=156,
            )
            ingest_clicked = st.button(
                "Ingest",
                key="header_ingest",
                type="primary",
                icon=":material/play_arrow:",
                width=120,
            )
            clear_clicked = st.button(
                "Clear chat",
                key="header_clear",
                icon=":material/delete:",
                width=140,
            )
    return {"upload": upload_clicked, "ingest": ingest_clicked, "clear": clear_clicked}


def render_metric_card(label: str, value: str, delta: str, tone: str = "cool") -> None:
    tone_class = {"warm": "delta-warm", "gold": "delta-gold"}.get(tone, "delta-cool")
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-label">{html.escape(label)}</div>
  <div class="metric-value">{html.escape(value)}</div>
  <div class="metric-delta {tone_class}">{html.escape(delta)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_overview(stats: dict[str, Any], recent_messages: list[dict[str, Any]]) -> str | None:
    st.markdown(
        """
<div class="hero-card">
  <h2 class="hero-title">Turn documents into answers.</h2>
  <div class="hero-copy">Ask questions across your documents and get grounded answers with clear citations.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    ask_col, button_col = st.columns([5, 1])
    question = ask_col.text_input(
        "Ask anything across your indexed PDFs",
        placeholder="Ask a question about your documents...",
        label_visibility="collapsed",
        key="overview_question",
    )
    submit = button_col.button("Ask", key="overview_ask", type="primary", use_container_width=True)

    chips = [
        "What are the key risks mentioned in the report?",
        "Summarize the employee onboarding process.",
        "Compare Q1 and Q2 performance metrics.",
    ]
    chip_cols = st.columns(3)
    chip_question = None
    for col, chip in zip(chip_cols, chips):
        if col.button(chip, key=f"overview_chip_{chip}", use_container_width=True):
            chip_question = chip

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Documents indexed", str(stats.get("total_documents", 0)), "Persistent ChromaDB", "warm")
    with metric_cols[1]:
        render_metric_card("Chunks stored", f'{stats.get("total_chunks", 0):,}', "Semantic retrieval-ready")
    with metric_cols[2]:
        confidence = "Ready" if stats.get("total_chunks", 0) else "Needs docs"
        render_metric_card("Answer relevance", confidence, "Reranking enabled", "gold")
    with metric_cols[3]:
        questions = len([m for m in recent_messages if m.get("role") == "user"])
        render_metric_card("Questions asked", str(questions), "This session")

    st.markdown(
        """
<div class="section-card">
  <div class="section-title">How it works</div>
  <div class="workflow">
    <div class="workflow-step"><span class="step-index">1</span><strong>Upload</strong><br/><span>Add PDFs to the knowledge base.</span></div>
    <div class="workflow-step"><span class="step-index">2</span><strong>Chunk &amp; Embed</strong><br/><span>Split documents into semantic chunks.</span></div>
    <div class="workflow-step"><span class="step-index">3</span><strong>Retrieve</strong><br/><span>Find the most relevant evidence.</span></div>
    <div class="workflow-step"><span class="step-index">4</span><strong>Answer</strong><br/><span>Generate responses with citations.</span></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    return chip_question or (question if submit and question.strip() else None)


def _format_answer_html(answer: str) -> str:
    escaped = html.escape(answer)
    pattern = re.compile(r"(\[source:.*?\])", flags=re.IGNORECASE)
    escaped = pattern.sub(r'<span class="citation-pill">\1</span>', escaped)
    return escaped


def render_chat_message(message: dict[str, Any]) -> None:
    role = message.get("role")
    content = message.get("content", "")
    if role == "user":
        st.markdown(f'<div class="chat-user">{html.escape(content)}</div>', unsafe_allow_html=True)
        return

    st.markdown(
        f"""
<div class="answer-card">
  <div class="assistant-label">RAG Knowledge Assistant</div>
  <div class="answer-body">{_format_answer_html(content)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if message.get("sources"):
        render_sources_panel(message["sources"], title="Sources used")
    if message.get("debug"):
        render_debug_panel(message["debug"])


def render_sources_panel(sources: list[dict[str, Any]], title: str = "Sources used") -> None:
    with st.expander(f"{title} ({len(sources)})", expanded=True):
        for source in sources:
            similarity = source.get("similarity")
            rerank = source.get("rerank_score")
            score = rerank if rerank is not None else similarity
            score_pct = int(max(0, min(1, float(score or 0))) * 100)
            snippet = (source.get("text") or "").replace("\n", " ")
            if len(snippet) > 260:
                snippet = snippet[:260] + "..."
            st.markdown(
                f"""
<div class="source-card">
  <div class="source-title">{html.escape(str(source.get("source", "Unknown source")))}</div>
  <div class="source-meta">Page {html.escape(str(source.get("page_number", "?")))} · Chunk {html.escape(str(source.get("chunk_id", "")))}</div>
  <div class="source-meta">Similarity: {similarity if similarity is not None else "n/a"} · Rerank: {rerank if rerank is not None else "n/a"}</div>
  <div class="score-bar"><span style="width:{score_pct}%"></span></div>
  <p>{html.escape(snippet)}</p>
</div>
""",
                unsafe_allow_html=True,
            )


def render_debug_panel(debug: dict[str, Any]) -> None:
    with st.expander("Behind the scenes", expanded=False):
        token_usage = debug.get("token_usage", {}) or {}
        token_total = token_usage.get("total", {}) or {}
        answer_total = debug.get("answer_total_tokens") or token_total.get("total_tokens", 0)
        st.markdown(
            f"""
<div class="debug-grid">
  <div class="debug-card"><div class="debug-label">Original query</div><div class="debug-value">{html.escape(debug.get("original_query", ""))}</div></div>
  <div class="debug-card"><div class="debug-label">Rewritten query</div><div class="debug-value">{html.escape(debug.get("rewritten_query", ""))}</div></div>
  <div class="debug-card"><div class="debug-label">Model used</div><div class="debug-value">{html.escape(debug.get("model", CHAT_MODEL))}</div></div>
  <div class="debug-card"><div class="debug-label">Response time</div><div class="debug-value">{debug.get("response_time", 0):.2f}s</div></div>
  <div class="debug-card"><div class="debug-label">Total tokens used</div><div class="debug-value">{int(token_total.get("total_tokens", 0) or 0):,}</div></div>
  <div class="debug-card"><div class="debug-label">Answer tokens</div><div class="debug-value">{int(answer_total or 0):,}</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("**Pipeline:** Query rewrite -> Retrieve top 10 -> Rerank top 5 -> Generate answer")
        steps = token_usage.get("steps", []) or []
        if steps:
            st.markdown("**Token usage by step**")
            st.dataframe(
                [
                    {
                        "Task": step.get("task", ""),
                        "Input tokens": int(step.get("prompt_tokens", 0) or 0),
                        "Output tokens": int(step.get("completion_tokens", 0) or 0),
                        "Total tokens": int(step.get("total_tokens", 0) or 0),
                    }
                    for step in steps
                ],
                hide_index=True,
            )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top retrieved chunks**")
            for chunk in debug.get("retrieved_chunks", [])[:10]:
                st.caption(
                    f'{chunk.get("source")} p.{chunk.get("page_number")} · '
                    f'similarity {chunk.get("similarity")}'
                )
        with col2:
            st.markdown("**Reranked chunks sent to model**")
            for chunk in debug.get("reranked_chunks", [])[:5]:
                st.caption(
                    f'{chunk.get("source")} p.{chunk.get("page_number")} · '
                    f'rerank {chunk.get("rerank_score")}'
                )
        st.caption(
            f'Prompt estimate: {debug.get("prompt_tokens_estimate", 0):,} tokens · '
            f'Completion estimate: {debug.get("completion_tokens_estimate", 0):,} tokens'
        )


def _format_ingested_timestamp(timestamp: Any) -> str:
    if not timestamp:
        return ""
    timestamp_text = str(timestamp)
    try:
        return datetime.fromisoformat(timestamp_text).strftime("%b %d, %Y %I:%M %p")
    except Exception:
        return timestamp_text


def _status_class(status: str) -> str:
    normalized = status.lower()
    if "index" in normalized:
        return "status-indexed"
    if "fail" in normalized or "error" in normalized:
        return "status-failed"
    if "skip" in normalized:
        return "status-skipped"
    return "status-pending"


def render_document_table(documents: list[dict[str, Any]], title: str = "Indexed documents") -> None:
    if not documents:
        render_empty_state("No indexed documents yet", "Upload PDFs or place files in docs/, then run ingestion.")
        return

    total_documents = len(documents)
    total_chunks = sum(int(doc.get("chunks") or 0) for doc in documents)
    max_chunks = max((int(doc.get("chunks") or 0) for doc in documents), default=0)
    formatted_timestamps = [_format_ingested_timestamp(doc.get("last_ingested")) for doc in documents]
    last_updated = next((timestamp for timestamp in formatted_timestamps if timestamp), "Not ingested yet")

    row_html = []
    for doc in documents:
        filename = str(doc.get("filename", "") or "Untitled document")
        extension = filename.rsplit(".", 1)[-1].upper() if "." in filename else "PDF"
        pages = int(doc.get("pages") or 0)
        chunks = int(doc.get("chunks") or 0)
        status = str(doc.get("status", "Indexed") or "Indexed").title()
        timestamp = _format_ingested_timestamp(doc.get("last_ingested")) or "Not available"
        document_hash = str(doc.get("document_hash", "") or "")
        short_hash = document_hash[:12] if document_hash else "n/a"
        density_width = 0 if max_chunks == 0 else max(8, int((chunks / max_chunks) * 100))
        status_class = _status_class(status)

        row_html.append(
            '<div class="doc-table-row">'
            '<div class="doc-cell">'
            '<div class="doc-main">'
            f'<div class="doc-file-icon">{html.escape(extension[:4])}</div>'
            '<div class="doc-file-text">'
            f'<div class="doc-file-name" title="{html.escape(filename)}">{html.escape(filename)}</div>'
            f'<div class="doc-file-meta">{html.escape(extension)} source document</div>'
            '</div>'
            '</div>'
            '</div>'
            f'<div class="doc-cell doc-num">{pages:,}</div>'
            '<div class="doc-cell chunk-cell">'
            f'<span class="doc-num">{chunks:,}</span>'
            f'<div class="chunk-meter" aria-label="Chunk density"><span style="width:{density_width}%"></span></div>'
            '</div>'
            f'<div class="doc-cell"><span class="status-pill {status_class}">{html.escape(status)}</span></div>'
            f'<div class="doc-cell">{html.escape(timestamp)}</div>'
            f'<div class="doc-cell"><span class="hash-chip" title="{html.escape(document_hash)}">{html.escape(short_hash)}</span></div>'
            '<div class="doc-cell">'
            '<div class="doc-row-actions">'
            '<span class="tiny-action">View</span>'
            '<span class="tiny-action alt">Sync</span>'
            '</div>'
            '</div>'
            '</div>'
        )

    rows_markup = "".join(row_html)
    table_markup = f"""
<div class="doc-table-card">
  <div class="doc-table-header">
    <div>
      <div class="doc-table-title">{html.escape(title)}</div>
      <div class="doc-table-summary">
        <span class="doc-summary-pill"><strong>{total_documents:,}</strong> documents</span>
        <span class="doc-summary-pill"><strong>{total_chunks:,}</strong> chunks</span>
        <span class="doc-summary-pill">Updated <strong>{html.escape(last_updated)}</strong></span>
      </div>
    </div>
    <div class="doc-table-actions" aria-label="Document table actions">
      <span class="doc-icon-btn" title="Refresh">Refresh</span>
      <span class="doc-icon-btn" title="Info">Info</span>
    </div>
  </div>
  <div class="doc-table-scroll">
    <div class="doc-table-grid">
      <div class="doc-table-head">
        <div class="doc-cell">Document</div>
        <div class="doc-cell">Pages</div>
        <div class="doc-cell">Chunks</div>
        <div class="doc-cell">Status</div>
        <div class="doc-cell">Last ingested</div>
        <div class="doc-cell">Hash</div>
        <div class="doc-cell">Actions</div>
      </div>
{rows_markup}
    </div>
  </div>
  <div class="doc-info-strip">All documents are chunked semantically and stored as high-quality embeddings for accurate retrieval.</div>
</div>
"""
    st.html(table_markup)


def render_empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="empty-state">
  <strong>{html.escape(title)}</strong>
  <p>{html.escape(body)}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_error_state(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="error-state">
  <strong>{html.escape(title)}</strong>
  <p>{html.escape(body)}</p>
</div>
""",
        unsafe_allow_html=True,
    )
