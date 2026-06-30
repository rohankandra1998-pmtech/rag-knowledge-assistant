from __future__ import annotations

import base64
import html
import re
from functools import lru_cache
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import streamlit as st
from PIL import Image, ImageFilter

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
HEADER_ICON_DIR = Path(__file__).parent / "assets" / "header-icons"
INDEXED_DOCS_ICON_DIR = Path(__file__).parent / "assets" / "indexed-documents"
INDEXED_DOCS_OPTIMIZED_ICON_DIR = Path(__file__).parent / "assets" / "indexed-documents-optimized"
PDF_MODAL_ICON_DIR = Path(__file__).parent / "assets" / "pdf-modal-icons"
UPLOAD_ICON_DIR = Path(__file__).parent / "assets" / "upload-icons"
CHAT_UI_ICON_DIR = Path(__file__).parent / "assets" / "chat-ui"
SIDEBAR_NAV_ITEMS = [
    {"label": "Chat / Answer", "icon": "Chat_Answer_Icon.png"},
    {"label": "Documents", "icon": "Documents_Icon.png"},
    {"label": "Ingestion status", "icon": "Ingestion_Status_Icon.png"},
    {"label": "Models", "icon": "Models_Icon.png"},
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


@st.cache_data(show_spinner=False)
def _load_header_icon_data_uri(filename: str) -> str:
    icon_path = HEADER_ICON_DIR / filename
    source = Image.open(icon_path).convert("RGBA")
    alpha = Image.new("L", source.size, 0)
    pixels = source.load()
    alpha_pixels = alpha.load()
    for y in range(source.height):
        for x in range(source.width):
            red, green, blue, _ = pixels[x, y]
            luminance = int((red * 299 + green * 587 + blue * 114) / 1000)
            alpha_pixels[x, y] = 255 if luminance < 160 else 0

    bounds = alpha.getbbox()
    if bounds:
        alpha = alpha.crop(bounds)
    side = max(alpha.size)
    padding = max(6, int(side * 0.08))
    normalized = Image.new("L", (side + padding * 2, side + padding * 2), 0)
    offset = ((normalized.width - alpha.width) // 2, (normalized.height - alpha.height) // 2)
    normalized.paste(alpha, offset)
    normalized = normalized.resize((96, 96), Image.Resampling.LANCZOS)

    icon = Image.new("RGBA", normalized.size, (255, 255, 255, 0))
    icon.putalpha(normalized)
    output = BytesIO()
    icon.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@lru_cache(maxsize=None)
def _load_png_data_uri_fast(path_text: str) -> str:
    try:
        encoded = base64.b64encode(Path(path_text).read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/png;base64,{encoded}"


@st.cache_data(show_spinner=False)
def _load_indexed_docs_icon_data_uri(filename: str) -> str:
    optimized_path = INDEXED_DOCS_OPTIMIZED_ICON_DIR / filename
    if optimized_path.exists():
        return _load_png_data_uri_fast(str(optimized_path))

    icon_path = INDEXED_DOCS_ICON_DIR / filename
    try:
        source = Image.open(icon_path).convert("RGBA")
    except OSError:
        return ""

    alpha = Image.new("L", source.size, 0)
    source_pixels = source.load()
    alpha_pixels = alpha.load()
    for y in range(source.height):
        for x in range(source.width):
            red, green, blue, original_alpha = source_pixels[x, y]
            luminance = int((red * 299 + green * 587 + blue * 114) / 1000)
            saturation = max(red, green, blue) - min(red, green, blue)
            is_artwork = original_alpha > 0 and (saturation > 34 or luminance < 150)
            alpha_pixels[x, y] = 255 if is_artwork else 0

    if filename == "pdf-file-icon.png":
        alpha = alpha.filter(ImageFilter.MaxFilter(9))

    bounds = alpha.getbbox()
    if not bounds:
        return ""

    source = source.crop(bounds)
    alpha = alpha.crop(bounds)
    source.putalpha(alpha)

    output = BytesIO()
    source.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_data(show_spinner=False)
def load_upload_icon_data_uri(filename: str) -> str:
    return _load_png_data_uri_fast(str(UPLOAD_ICON_DIR / filename))


def _load_chat_ui_icon_data_uri(filename: str) -> str:
    return _load_png_data_uri_fast(str(CHAT_UI_ICON_DIR / filename))


def _load_chat_empty_state_asset_data_uri(filename: str) -> str:
    return _load_png_data_uri_fast(str(CHAT_UI_ICON_DIR / "empty-state" / filename))


def _build_evidence_empty_icon_svg() -> str:
    return """
<svg viewBox="0 0 120 90" role="img" aria-label="Evidence document search" focusable="false">
  <g fill="none" stroke-linecap="round" stroke-linejoin="round">
    <path d="M43 12h30l18 18v42c0 6.1-4.9 11-11 11H43c-6.1 0-11-4.9-11-11V23c0-6.1 4.9-11 11-11Z" stroke="#105EDD" stroke-width="6.4"/>
    <path d="M73 13v17h17" stroke="#105EDD" stroke-width="6.4"/>
    <circle cx="58" cy="52" r="11.5" stroke="#105EDD" stroke-width="6.4"/>
    <path d="m67 61 12 12" stroke="#105EDD" stroke-width="6.4"/>
    <path d="M20 45c7.2-2.1 10.9-5.9 13-13 2.1 7.1 5.8 10.9 13 13-7.2 2.1-10.9 5.9-13 13-2.1-7.1-5.8-10.9-13-13Z" stroke="#F8B400" stroke-width="4.8"/>
    <path d="M91 63c4.8-1.4 7.2-3.8 8.7-8.6 1.4 4.8 3.9 7.2 8.6 8.6-4.8 1.4-7.2 3.9-8.6 8.6-1.5-4.8-3.9-7.2-8.7-8.6Z" stroke="#F8B400" stroke-width="4.4"/>
  </g>
  <circle cx="26" cy="29" r="4" fill="#F8B400"/>
  <circle cx="101" cy="47" r="3.6" fill="#F8B400"/>
</svg>
"""


def load_header_action_icon_data_uri(filename: str) -> str:
    return _load_header_icon_data_uri(filename)


def render_upload_badges() -> None:
    badges = [
        ("pdf_only_icon.png", "PDF only", "is-red"),
        ("persistent_upload_shield_icon.png", "Persistent upload", "is-blue"),
        ("duplicate_safe_shield_icon.png", "Duplicate-safe", "is-green"),
    ]
    badge_html = []
    for filename, label, tone in badges:
        icon_uri = load_upload_icon_data_uri(filename)
        icon_html = f'<img src="{icon_uri}" alt="" aria-hidden="true" loading="lazy" />' if icon_uri else ""
        badge_html.append(
            f'<span class="documents-badge {tone}">{icon_html}<span>{html.escape(label)}</span></span>'
        )
    st.markdown(f'<div class="documents-badges">{"".join(badge_html)}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_pdf_modal_icon_data_uri(folder: str, filename: str) -> str:
    icon_path = PDF_MODAL_ICON_DIR / folder / filename
    try:
        source = Image.open(icon_path).convert("RGBA")
    except OSError:
        return ""

    pixels = source.load()
    for y in range(source.height):
        for x in range(source.width):
            red, green, blue, alpha = pixels[x, y]
            is_near_white = red > 242 and green > 242 and blue > 242 and (max(red, green, blue) - min(red, green, blue)) < 12
            if is_near_white:
                pixels[x, y] = (red, green, blue, 0)
            elif alpha > 0:
                pixels[x, y] = (red, green, blue, alpha)

    bounds = source.getbbox()
    if bounds:
        source = source.crop(bounds)

    output = BytesIO()
    source.save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_pdf_viewer_control_icon_data_uri(filename: str) -> str:
    return _load_pdf_modal_icon_data_uri("viewer-controls", filename)


@st.cache_data(show_spinner=False)
def load_pdf_document_detail_icon_data_uri(filename: str) -> str:
    return _load_pdf_modal_icon_data_uri("document-details", filename)


def _format_sidebar_nav_label(label: str) -> str:
    icon_by_label = {item["label"]: item["icon"] for item in SIDEBAR_NAV_ITEMS}
    icon_uri = _load_sidebar_icon_data_uri(icon_by_label[label])
    return f"![]({icon_uri}) {label}"


def inject_custom_css() -> None:
    upload_icon_uri = _load_header_icon_data_uri("upload.png")
    ingest_icon_uri = _load_header_icon_data_uri("ingest.png")
    clear_icon_uri = _load_header_icon_data_uri("clear-chat.png")
    pipeline_arrow_icon_uri = _load_chat_ui_icon_data_uri("pipeline-arrow.png")
    pipeline_clock_icon_uri = _load_chat_ui_icon_data_uri("pipeline-clock.png")
    composer_send_icon_uri = _load_chat_ui_icon_data_uri("composer-send.png")
    sources_used_icon_uri = _load_chat_ui_icon_data_uri("sources-used.png")
    behind_scenes_icon_uri = _load_chat_ui_icon_data_uri("behind-the-scenes.png")
    st.markdown(
        f"""
<style>
:root {{
  --pipeline-arrow-icon: url("{pipeline_arrow_icon_uri}");
  --pipeline-clock-icon: url("{pipeline_clock_icon_uri}");
  --composer-send-icon: url("{composer_send_icon_uri}");
  --answer-sources-icon: url("{sources_used_icon_uri}");
  --answer-debug-icon: url("{behind_scenes_icon_uri}");
}}
.st-key-header_actions .st-key-header_upload [data-testid^="stBaseButton"]::before {{
  content: "";
  width: 22px;
  height: 22px;
  flex: 0 0 22px;
  background: currentColor;
  -webkit-mask: url("{upload_icon_uri}") center / contain no-repeat;
  mask: url("{upload_icon_uri}") center / contain no-repeat;
}}
.st-key-header_actions .st-key-header_ingest [data-testid^="stBaseButton"]::before {{
  content: "";
  width: 22px;
  height: 22px;
  flex: 0 0 22px;
  background: currentColor;
  -webkit-mask: url("{ingest_icon_uri}") center / contain no-repeat;
  mask: url("{ingest_icon_uri}") center / contain no-repeat;
}}
.st-key-header_actions .st-key-header_clear [data-testid^="stBaseButton"]::before {{
  content: "";
  width: 22px;
  height: 22px;
  flex: 0 0 22px;
  background: currentColor;
  -webkit-mask: url("{clear_icon_uri}") center / contain no-repeat;
  mask: url("{clear_icon_uri}") center / contain no-repeat;
}}
</style>
""",
        unsafe_allow_html=True,
    )
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

.stale-element {
  opacity: 1 !important;
  filter: none !important;
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
  padding: 0.85rem 0.55rem 1.3rem;
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

.st-key-app_header_shell {
  margin: 0.35rem 0 1.35rem;
}
.st-key-app_header_shell [data-testid="stHorizontalBlock"] {
  align-items: flex-start;
}
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  margin: 0;
}
.app-title-block {
  min-width: 0;
}
.app-title {
  color: var(--navy);
  font-size: clamp(2.15rem, 2.7vw, 3rem);
  line-height: 1.05;
  font-weight: 900;
  margin: 0;
  white-space: nowrap;
}
.app-subtitle {
  color: #405072;
  font-size: 1rem;
  margin-top: 0.35rem;
}
.st-key-header_actions {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: nowrap;
  padding-top: 1.55rem;
  width: 100%;
}
.st-key-header_actions [data-testid="stButton"] {
  flex: 0 0 auto;
  width: auto !important;
}
.st-key-header_actions div.stButton > button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.55rem;
  height: 2.75rem;
  min-height: 2.75rem;
  padding: 0 0.8rem;
  border-radius: 10px;
  font-size: 0.92rem;
  font-weight: 800;
  white-space: nowrap;
  box-shadow: 0 12px 28px rgba(11,48,117,0.08);
  transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease, border-color 140ms ease;
}
.st-key-header_actions div.stButton > button:hover {
  transform: translateY(-1px);
}
.st-key-header_actions .st-key-header_upload [data-testid^="stBaseButton"],
.st-key-header_actions .st-key-header_upload button,
.st-key-header_actions [data-testid="stElementContainer"]:nth-of-type(1) [data-testid^="stBaseButton"] {
  background: #FFFFFF !important;
  border-color: #CFE1FB !important;
  color: var(--navy) !important;
}
.st-key-header_actions .st-key-header_upload [data-testid^="stBaseButton"]:hover,
.st-key-header_actions .st-key-header_upload button:hover,
.st-key-header_actions [data-testid="stElementContainer"]:nth-of-type(1) [data-testid^="stBaseButton"]:hover {
  background: #F6FAFF !important;
  border-color: #BBD6FF !important;
  color: var(--navy) !important;
}
.st-key-header_actions .st-key-header_ingest [data-testid^="stBaseButton"],
.st-key-header_actions .st-key-header_ingest button,
.st-key-header_actions [data-testid="stElementContainer"]:nth-of-type(2) [data-testid^="stBaseButton"] {
  background: var(--blue) !important;
  border-color: var(--blue) !important;
  color: #FFFFFF !important;
  box-shadow: 0 14px 28px rgba(16,94,221,0.18);
}
.st-key-header_actions .st-key-header_ingest [data-testid^="stBaseButton"]:hover,
.st-key-header_actions .st-key-header_ingest button:hover,
.st-key-header_actions [data-testid="stElementContainer"]:nth-of-type(2) [data-testid^="stBaseButton"]:hover {
  background: #0B4FC7 !important;
  border-color: #0B4FC7 !important;
  color: #FFFFFF !important;
  box-shadow: 0 16px 32px rgba(16,94,221,0.22);
}
.st-key-header_actions .st-key-header_clear [data-testid^="stBaseButton"],
.st-key-header_actions .st-key-header_clear button,
.st-key-header_actions [data-testid="stElementContainer"]:nth-of-type(3) [data-testid^="stBaseButton"] {
  background: var(--terracotta) !important;
  border-color: var(--terracotta) !important;
  color: #FFFFFF !important;
  box-shadow: 0 14px 28px rgba(200,71,44,0.18);
}
.st-key-header_actions .st-key-header_clear [data-testid^="stBaseButton"]:hover,
.st-key-header_actions .st-key-header_clear button:hover,
.st-key-header_actions [data-testid="stElementContainer"]:nth-of-type(3) [data-testid^="stBaseButton"]:hover {
  background: #A93A24 !important;
  border-color: #A93A24 !important;
  color: #FFFFFF !important;
  box-shadow: 0 16px 32px rgba(200,71,44,0.22);
}
.st-key-header_actions div.stButton > button p {
  white-space: nowrap;
  margin: 0;
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

.ingestion-status-title {
  color: var(--navy);
  font-size: 1.45rem;
  font-weight: 900;
  line-height: 1.12;
  margin: 0 0 1rem;
}
.ingestion-status-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1.45rem;
  margin: 0.2rem 0 1rem;
}
.ingestion-status-card {
  --accent: var(--blue);
  --helper: var(--blue);
  position: relative;
  min-height: 156px;
  overflow: hidden;
  display: flex;
  align-items: center;
  gap: 1.15rem;
  padding: 1.35rem 1.55rem;
  border: 1px solid #DFE7F3;
  border-left: 3px solid var(--accent);
  border-radius: 18px;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,251,255,0.95)),
    #FFFFFF;
  box-shadow: 0 18px 45px rgba(11, 48, 117, 0.10);
}
.ingestion-status-card.is-warm {
  --accent: #FF3B16;
  --helper: #FF3B16;
  --badge-bg: #FFF0EA;
  --badge-border: rgba(255, 59, 22, 0.18);
  --icon-stroke: #FF3B16;
}
.ingestion-status-card.is-cool {
  --accent: #105EDD;
  --helper: #105EDD;
  --badge-bg: #EAF3FF;
  --badge-border: rgba(16, 94, 221, 0.18);
  --icon-stroke: #105EDD;
}
.ingestion-status-card.is-gold {
  --accent: #F5B400;
  --helper: #A87400;
  --badge-bg: #FFF7DF;
  --badge-border: rgba(245, 180, 0, 0.24);
  --icon-stroke: #F5B400;
}
.ingestion-status-icon {
  width: 70px;
  height: 70px;
  flex: 0 0 70px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  border: 1px solid var(--badge-border);
  background: var(--badge-bg);
  box-shadow: 0 10px 22px rgba(11, 48, 117, 0.06);
}
.ingestion-status-icon svg {
  width: 38px;
  height: 38px;
  display: block;
  color: var(--icon-stroke);
  stroke: currentColor;
}
.ingestion-status-copy {
  min-width: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.ingestion-status-label {
  color: var(--navy);
  font-size: 0.95rem;
  line-height: 1.18;
  font-weight: 800;
}
.ingestion-status-value {
  color: #020A34;
  font-size: 2.5rem;
  line-height: 1.02;
  font-weight: 900;
  margin-top: 0.62rem;
  letter-spacing: 0;
  white-space: nowrap;
  word-break: keep-all;
  overflow-wrap: normal;
}
.ingestion-status-value.is-text {
  font-size: 2.1rem;
}
.ingestion-status-helper {
  color: var(--helper);
  font-size: 0.86rem;
  font-weight: 800;
  margin-top: 1.18rem;
}

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
}
.chat-answer-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-bottom: 0.55rem;
}
.chat-assistant-time {
  color: #5F6D83;
  font-size: 0.72rem;
  font-weight: 750;
  white-space: nowrap;
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

.chat-section-head {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 0.1rem 0 0.75rem;
}
.chat-section-head .section-title {
  margin: 0;
  white-space: nowrap;
}
.chat-section-rule {
  height: 1px;
  flex: 1;
  background: linear-gradient(90deg, rgba(11,48,117,0.22), rgba(11,48,117,0));
}
.st-key-chat_canvas_card {
  min-height: 610px;
  padding: 1rem 1.05rem 0.9rem;
  border: 1px solid rgba(16, 94, 221, 0.14);
  border-radius: 16px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,251,255,0.94)),
    #FFFFFF;
  box-shadow: 0 18px 46px rgba(11, 48, 117, 0.08);
}
.st-key-chat_canvas_card:has(.chat-empty-state) {
  min-height: 535px;
  padding-bottom: 0.45rem;
}
.chat-thread {
  display: flex;
  flex-direction: column;
  gap: 0.7rem;
}
.chat-user-row {
  display: flex;
  justify-content: flex-end;
  margin: 0.25rem 0 0.1rem;
}
.chat-user-bubble {
  display: flex;
  align-items: center;
  gap: 0.62rem;
  width: fit-content;
  max-width: 74%;
  min-height: 45px;
  padding: 0.62rem 0.78rem 0.62rem 1rem;
  border: 1px solid rgba(245, 180, 0, 0.68);
  border-radius: 11px;
  background: linear-gradient(180deg, #FFFDF7, #FFF9E8);
  color: #1E2435;
  font-size: 0.92rem;
  font-weight: 800;
  box-shadow: 0 10px 24px rgba(245, 180, 0, 0.08);
}
.chat-user-time {
  color: #5F6D83;
  font-size: 0.7rem;
  font-weight: 700;
  white-space: nowrap;
}
.chat-user-avatar {
  position: relative;
  width: 26px;
  height: 26px;
  flex: 0 0 26px;
  border-radius: 999px;
  background: #24170F;
}
.chat-user-avatar::before,
.chat-user-avatar::after {
  content: "";
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  border-radius: 999px;
  background: #FFFFFF;
}
.chat-user-avatar::before {
  top: 5px;
  width: 7px;
  height: 7px;
}
.chat-user-avatar::after {
  bottom: 5px;
  width: 14px;
  height: 8px;
}
.chat-assistant-row {
  display: grid;
  grid-template-columns: 42px minmax(0, 1fr);
  gap: 0.45rem;
  align-items: start;
  max-width: 83%;
  margin: 0.1rem 0 0.35rem;
}
.chat-assistant-row.is-selected .chat-answer-card {
  border-color: rgba(16, 94, 221, 0.58);
  background: linear-gradient(180deg, rgba(255,255,255,0.99), rgba(247,251,255,0.98));
  box-shadow: 0 0 0 2px rgba(16,94,221,0.12), 0 18px 42px rgba(16, 94, 221, 0.16);
}
.chat-bot-avatar {
  width: 42px;
  height: 42px;
  margin-top: 0;
  border-radius: 12px;
  border: 1px solid rgba(207, 225, 251, 0.72);
  background: rgba(255,255,255,0.42);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 10px 20px rgba(16, 94, 221, 0.08);
}
.chat-bot-avatar img {
  width: 40px;
  height: 40px;
  display: block;
  object-fit: contain;
}
.chat-answer-card {
  padding: 0.92rem 1rem 0.86rem;
  border: 1px solid #DCE7F5;
  border-radius: 12px;
  background: rgba(255,255,255,0.98);
  box-shadow: 0 10px 28px rgba(11, 48, 117, 0.06);
}
.chat-assistant-row:has(+ [class*="st-key-answer_footer_"]) .chat-answer-card {
  border-bottom-color: transparent;
  border-radius: 12px 12px 0 0;
  box-shadow: 0 8px 20px rgba(11, 48, 117, 0.045);
}
.chat-answer-card .answer-body {
  font-size: 0.9rem;
  line-height: 1.52;
}
.chat-source-pill {
  display: inline-flex;
  max-width: 100%;
  margin-top: 0.52rem;
  padding: 0.16rem 0.52rem;
  border: 1px solid #BBD6FF;
  border-radius: 7px;
  background: #EAF3FF;
  color: var(--blue);
  font-size: 0.75rem;
  font-weight: 900;
  line-height: 1.35;
  overflow-wrap: anywhere;
  white-space: normal;
}
.chat-answer-divider {
  height: 1px;
  margin-top: 0.62rem;
  background: #E5EDF7;
}
[class*="st-key-answer_footer_"] {
  max-width: calc(83% - 42px);
  margin-left: 42px;
  margin-top: -0.36rem;
  margin-bottom: 0.95rem;
  padding: 0.52rem 0.85rem 0.56rem;
  border: 1px solid #DCE7F5;
  border-top-color: #E5EDF7;
  border-radius: 0 0 12px 12px;
  background: rgba(255,255,255,0.98);
  box-shadow: 0 12px 24px rgba(11, 48, 117, 0.045);
}
[class*="st-key-answer_footer_"] button {
  min-height: 38px !important;
  height: 38px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  border-radius: 8px !important;
  border-color: #DCE7F5 !important;
  background: #FFFFFF !important;
  color: var(--navy) !important;
  box-shadow: none !important;
  font-size: 0.8rem !important;
  font-weight: 850 !important;
  line-height: 1 !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  transition: background 160ms ease, border-color 160ms ease, box-shadow 160ms ease, color 160ms ease, transform 160ms ease;
}
[class*="st-key-answer_footer_"] button p {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin: 0;
  line-height: 1;
  white-space: nowrap;
}
[class*="st-key-answer_footer_"] button [data-testid="stMarkdownContainer"] {
  height: 100%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
[class*="st-key-answer_sources_button_"] button p::before,
[class*="st-key-answer_debug_button_"] button p::before {
  content: "";
  width: 17px;
  height: 17px;
  flex: 0 0 17px;
  display: inline-block;
  margin-right: 0.44rem;
  transform: translateY(0);
  background-position: center;
  background-repeat: no-repeat;
  background-size: contain;
}
[class*="st-key-answer_sources_button_"] button p::before {
  background-image: var(--answer-sources-icon);
}
[class*="st-key-answer_debug_button_"] button p::before {
  background-image: var(--answer-debug-icon);
}
[class*="st-key-answer_footer_"] button:hover {
  border-color: #105EDD !important;
  color: #105EDD !important;
  background: #F7FBFF !important;
  box-shadow: 0 8px 16px rgba(16,94,221,0.10) !important;
  transform: translateY(-1px);
}
[class*="st-key-answer_footer_"] button:active {
  transform: translateY(0);
}
[class*="st-key-answer_footer_"] button:focus-visible {
  outline: 3px solid rgba(88, 172, 244, 0.30) !important;
  outline-offset: 2px !important;
}
[class*="st-key-answer_sources_button_active_"] button,
[class*="st-key-answer_debug_button_active_"] button {
  border-color: #105EDD !important;
  background: #105EDD !important;
  color: #FFFFFF !important;
  box-shadow: 0 10px 20px rgba(16,94,221,0.18) !important;
}
[class*="st-key-answer_sources_button_active_"] button p::before,
[class*="st-key-answer_debug_button_active_"] button p::before {
  filter: brightness(0) invert(1);
}
[class*="st-key-answer_sources_button_active_"] button:hover,
[class*="st-key-answer_debug_button_active_"] button:hover {
  border-color: #0C4EC2 !important;
  background: #0C4EC2 !important;
  color: #FFFFFF !important;
  box-shadow: 0 12px 22px rgba(16,94,221,0.22) !important;
}
.chat-footer-status {
  min-height: 38px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  color: #68758E;
  font-size: 0.74rem;
  font-weight: 800;
}
.chat-footer-status::before {
  content: "";
  width: 8px;
  height: 8px;
  margin-right: 0.38rem;
  border-radius: 999px;
  background: #105EDD;
  box-shadow: 0 0 0 3px rgba(16,94,221,0.12);
}
.chat-pipeline-strip {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  flex-wrap: wrap;
  margin: 0.55rem 0 0;
  padding: 0.55rem 0.72rem;
  border: 1px solid #DCE7F5;
  border-radius: 12px;
  background: #FFFFFF;
}
.pipeline-step {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  color: var(--navy);
  font-size: 0.76rem;
  font-weight: 800;
}
.pipeline-step:not(:last-of-type)::after {
  content: "→";
  color: #98A8C0;
  margin-left: 0.25rem;
}
.pipeline-check {
  width: 18px;
  height: 18px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: #13A73F;
  color: #FFFFFF;
  font-size: 0.72rem;
  font-weight: 900;
}
.pipeline-time {
  margin-left: auto;
  display: inline-flex;
  align-items: center;
  gap: 0.34rem;
  color: var(--navy);
  font-size: 0.78rem;
  font-weight: 900;
}
.pipeline-step {
  color: #7B879D;
}
.pipeline-step.is-complete {
  color: var(--navy);
}
.pipeline-step.is-active {
  color: var(--blue);
}
.pipeline-step.is-failed {
  color: var(--terracotta);
}
.pipeline-step::after {
  content: none !important;
}
.pipeline-check {
  border: 1px solid #C8D5E8;
  background: #F4F7FB;
  color: transparent;
}
.pipeline-check.is-complete {
  border-color: #13A73F;
  background: #13A73F;
  color: #FFFFFF;
}
.pipeline-check.is-active {
  position: relative;
  border-color: var(--blue);
  background: var(--blue);
  box-shadow: 0 0 0 4px rgba(16, 94, 221, 0.12);
}
.pipeline-check.is-active::after {
  content: "";
  width: 8px;
  height: 8px;
  border-radius: 999px;
  border: 2px solid rgba(255,255,255,0.45);
  border-top-color: #FFFFFF;
  animation: pipeline-spin 850ms linear infinite;
}
.pipeline-check.is-failed {
  border-color: var(--terracotta);
  background: var(--terracotta);
  color: #FFFFFF;
}
.pipeline-time.is-loading {
  color: var(--blue);
}
.pipeline-time.is-failed {
  color: var(--terracotta);
}
.pipeline-arrow {
  width: 16px;
  height: 12px;
  flex: 0 0 16px;
  background: #8DA2C0;
  -webkit-mask: var(--pipeline-arrow-icon) center / contain no-repeat;
  mask: var(--pipeline-arrow-icon) center / contain no-repeat;
}
.pipeline-clock {
  width: 15px;
  height: 15px;
  flex: 0 0 15px;
  background: currentColor;
  -webkit-mask: var(--pipeline-clock-icon) center / contain no-repeat;
  mask: var(--pipeline-clock-icon) center / contain no-repeat;
}
@keyframes pipeline-spin {
  to { transform: rotate(360deg); }
}
.st-key-chat_composer_card {
  margin-top: 0.55rem;
  padding: 0.55rem;
  border: 1px solid #DCE7F5;
  border-radius: 16px;
  background: #FFFFFF;
  box-shadow: 0 12px 28px rgba(11, 48, 117, 0.06);
}
.st-key-chat_composer_card [data-testid="stForm"] {
  border: 0;
  padding: 0;
}
.st-key-chat_composer_card [data-testid="stHorizontalBlock"] {
  align-items: center;
}
.st-key-chat_composer_card input {
  border: 0 !important;
  box-shadow: none !important;
  color: var(--navy) !important;
  font-weight: 700;
}
.st-key-chat_composer_card [data-testid="stTextInputRootElement"] {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.st-key-chat_composer_card button {
  position: relative;
  width: 46px !important;
  height: 46px !important;
  min-height: 46px !important;
  padding: 0 !important;
  border-radius: 999px !important;
  border-color: transparent !important;
  background: transparent !important;
  color: #FFFFFF !important;
  box-shadow: 0 12px 22px rgba(16, 94, 221, 0.24) !important;
  cursor: pointer;
  transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease, border-color 160ms ease, outline-color 160ms ease;
}
.st-key-chat_composer_card button p {
  font-size: 0;
  width: 0;
  min-width: 0;
  margin: 0;
}
.st-key-chat_composer_card button::before {
  content: "";
  width: 46px;
  min-width: 46px;
  height: 46px;
  flex: 0 0 46px;
  display: block;
  background: center / contain no-repeat var(--composer-send-icon);
  transition: filter 160ms ease;
}
.st-key-chat_composer_card button:hover {
  background: #081A52 !important;
  border-color: #081A52 !important;
  box-shadow: 0 16px 28px rgba(16, 94, 221, 0.30) !important;
  transform: translateY(-1px);
}
.st-key-chat_composer_card button:hover::before {
  filter: brightness(0.94) saturate(1.08);
}
.st-key-chat_composer_card button:active {
  transform: translateY(0);
  box-shadow: 0 8px 16px rgba(16, 94, 221, 0.20) !important;
}
.st-key-chat_composer_card button:active::before {
  filter: brightness(0.9) saturate(1.05);
}
.st-key-chat_composer_card button:focus-visible {
  outline: 3px solid rgba(88, 172, 244, 0.36) !important;
  outline-offset: 3px !important;
}
.chat-empty-state {
  min-height: 420px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  gap: 0.7rem;
  color: var(--navy);
  padding: 1.05rem 0.75rem 0.35rem;
  overflow: hidden;
}
.chat-empty-graphic {
  position: relative;
  width: min(100%, 940px);
  margin: 0 auto 0.35rem;
}
.chat-empty-flow-image {
  display: block;
  width: min(100%, 900px);
  height: auto;
  object-fit: contain;
  margin: 0 auto;
  filter: drop-shadow(0 22px 42px rgba(11, 48, 117, 0.10));
}
.chat-empty-title {
  font-size: 1.25rem;
  font-weight: 900;
  line-height: 1.2;
}
.chat-empty-copy {
  max-width: 520px;
  color: #53637F;
  font-size: 0.94rem;
  line-height: 1.55;
}
@media (max-width: 920px) {
  .chat-empty-state {
    min-height: 445px;
  }
  .chat-empty-graphic {
    width: 100%;
  }
  .chat-empty-flow-image {
    width: min(100%, 720px);
  }
}
@media (max-width: 640px) {
  .chat-empty-state {
    min-height: 460px;
    padding-inline: 0.25rem;
  }
  .st-key-chat_canvas_card:has(.chat-empty-state) {
    min-height: 480px;
  }
  .chat-empty-flow-image {
    width: 100%;
  }
}
.st-key-chat_evidence_panel_shell {
  padding: 0.95rem;
  border: 1px solid rgba(16, 94, 221, 0.14);
  border-radius: 16px;
  background: rgba(255,255,255,0.98);
  box-shadow: inset 4px 0 0 #105EDD, 0 18px 46px rgba(11, 48, 117, 0.08);
}
.evidence-header-row {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.45rem;
  margin-bottom: 0.65rem;
}
.evidence-header {
  color: var(--navy);
  font-size: 1rem;
  font-weight: 900;
  line-height: 1.15;
  margin-bottom: 0;
  white-space: nowrap;
}
.evidence-selected-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.28rem;
  max-width: 100%;
  padding: 0.3rem 0.52rem;
  border: 1px solid #CFE1FB;
  border-radius: 999px;
  background: #F3F8FF;
  color: #105EDD;
  font-size: 0.72rem;
  font-weight: 850;
}
.evidence-selected-pill-icon {
  width: 14px;
  height: 14px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 14px;
}
.evidence-selected-pill-icon img {
  width: 14px;
  height: 14px;
  display: block;
  object-fit: contain;
}
.evidence-selected-preview {
  margin: 0.45rem 0 0.62rem;
  padding: 0.72rem;
  border: 1px solid #D6E5F8;
  border-radius: 9px;
  background: linear-gradient(180deg, #F8FBFF, #FFFFFF);
  box-shadow: 0 10px 24px rgba(11, 48, 117, 0.05);
}
.evidence-selected-preview-title {
  margin-bottom: 0.48rem;
  color: var(--navy);
  font-size: 0.78rem;
  font-weight: 900;
}
.evidence-selected-preview-body {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr);
  gap: 0.55rem;
  align-items: center;
}
.evidence-selected-preview-icon {
  width: 34px;
  height: 34px;
  border-radius: 9px;
  border: 1px solid #CFE1FB;
  background: #FFFFFF;
  box-shadow: 0 8px 14px rgba(16,94,221,0.08);
}
.evidence-selected-preview-icon img {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: contain;
}
.evidence-selected-preview-text {
  min-width: 0;
  color: #1D2E52;
  font-size: 0.76rem;
  line-height: 1.38;
}
.evidence-opened-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.22rem;
  max-width: 100%;
  padding: 0.34rem 0.5rem;
  border: 1px solid #CFE1FB;
  border-radius: 7px;
  background: #EDF5FF;
  color: var(--navy);
  font-size: 0.78rem;
}
.evidence-opened-pill strong {
  color: var(--blue);
}
.evidence-section-title {
  margin: 0.78rem 0 0.45rem;
  color: var(--navy);
  font-size: 0.86rem;
  font-weight: 900;
}
.evidence-section-title.compact {
  margin-top: 0.7rem;
}
.evidence-source-card {
  margin: 0.45rem 0 0.65rem;
  overflow: hidden;
  border: 1px solid #DCE7F5;
  border-radius: 10px;
  background: #FFFFFF;
  box-shadow: 0 8px 20px rgba(11, 48, 117, 0.05);
}
.evidence-source-card[open] {
  border-color: #C7DCF5;
  box-shadow: 0 12px 24px rgba(11, 48, 117, 0.07);
}
.evidence-source-summary {
  list-style: none;
  display: grid;
  grid-template-columns: 35px minmax(0, 1fr) auto;
  gap: 0.65rem;
  align-items: center;
  padding: 0.7rem 0.72rem;
  cursor: pointer;
  transition: background 160ms ease;
}
.evidence-source-summary::-webkit-details-marker {
  display: none;
}
.evidence-source-summary::marker {
  content: "";
}
.evidence-source-summary:hover {
  background: #F7FAFF;
}
.evidence-source-top {
  display: flex;
  gap: 0.65rem;
  align-items: flex-start;
}
.evidence-pdf-badge {
  width: 35px;
  height: 40px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 35px;
  border-radius: 7px;
  background: linear-gradient(180deg, #E52D18, #B81F12);
  color: #FFFFFF;
  font-size: 0.68rem;
  font-weight: 900;
  box-shadow: 0 8px 16px rgba(229,45,24,0.18);
}
.evidence-source-title-wrap {
  min-width: 0;
}
.evidence-source-name {
  color: var(--navy);
  font-size: 0.82rem;
  font-weight: 900;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.evidence-source-meta {
  color: #425275;
  font-size: 0.74rem;
  margin-top: 0.12rem;
}
.evidence-source-compact-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem;
  margin-top: 0.34rem;
}
.evidence-compact-score {
  display: inline-flex;
  align-items: center;
  gap: 0.22rem;
  min-height: 21px;
  padding: 0.12rem 0.38rem;
  border: 1px solid #DCE7F5;
  border-radius: 999px;
  background: #F8FBFF;
  color: #425275;
  font-size: 0.68rem;
  font-weight: 800;
}
.evidence-compact-score strong {
  color: var(--navy);
  font-size: 0.68rem;
}
.evidence-source-chevron {
  width: 24px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #DCE7F5;
  border-radius: 999px;
  background: #FFFFFF;
  color: var(--blue);
  font-size: 0.9rem;
  font-weight: 900;
  transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
}
.evidence-source-card[open] .evidence-source-chevron {
  transform: rotate(90deg);
  border-color: #BBD6F5;
  background: #EEF6FF;
}
.evidence-source-expanded {
  padding: 0.62rem 0.72rem 0.72rem;
  border-top: 1px solid #E8EEF7;
  background: #FFFFFF;
}
.evidence-score-row {
  display: grid;
  grid-template-columns: 70px 42px minmax(0, 1fr);
  align-items: center;
  gap: 0.45rem;
  margin-top: 0.52rem;
  color: #20345B;
  font-size: 0.74rem;
}
.evidence-score-row strong {
  color: var(--navy);
  font-size: 0.73rem;
}
.evidence-score-track {
  height: 7px;
  overflow: hidden;
  border-radius: 999px;
  background: #D9E3F2;
}
.evidence-score-track span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #58ACF4, #105EDD);
}
.evidence-snippet {
  margin-top: 0.62rem;
  padding: 0.58rem;
  border: 1px solid #E2EAF5;
  border-radius: 7px;
  background: #F8FAFD;
  color: #1D2E52;
  font-size: 0.78rem;
  line-height: 1.42;
}
.evidence-source-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.45rem;
  margin-top: 0.5rem;
}
.evidence-action-link {
  min-height: 34px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #DCE7F5;
  border-radius: 7px;
  background: #FFFFFF;
  color: var(--blue) !important;
  font-size: 0.76rem;
  font-weight: 900;
  text-decoration: none !important;
}
.evidence-action-link.is-primary {
  background: #F6FAFF;
}
.evidence-debug-card {
  overflow: hidden;
  border: 1px solid #DCE7F5;
  border-radius: 9px;
  background: #FFFFFF;
}
.evidence-debug-row {
  display: grid;
  grid-template-columns: 0.92fr 1.25fr;
  gap: 0.5rem;
  padding: 0.38rem 0.5rem;
  border-bottom: 1px solid #E8EEF7;
  color: #24395F;
  font-size: 0.72rem;
}
.evidence-debug-row:last-child {
  border-bottom: 0;
}
.evidence-debug-row span {
  color: var(--navy);
  font-weight: 800;
}
.evidence-debug-row strong {
  font-weight: 700;
  overflow-wrap: anywhere;
}
.evidence-token-table {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border: 1px solid #DCE7F5;
  border-radius: 9px;
  background: #FFFFFF;
  color: #24395F;
  font-size: 0.71rem;
}
.evidence-token-table th,
.evidence-token-table td {
  padding: 0.34rem 0.42rem;
  border-bottom: 1px solid #E8EEF7;
  border-right: 1px solid #E8EEF7;
  text-align: left;
}
.evidence-token-table th {
  color: var(--navy);
  font-weight: 900;
  background: #F8FBFF;
}
.evidence-token-table td:last-child,
.evidence-token-table th:last-child {
  border-right: 0;
}
.evidence-token-table tr:last-child td {
  border-bottom: 0;
}
.evidence-empty {
  padding: 0.8rem;
  border: 1px dashed #CFE1FB;
  border-radius: 10px;
  background: #F8FBFF;
  color: #53637F;
  font-size: 0.82rem;
}
.evidence-empty.large {
  min-height: 335px;
  margin: 0.45rem 0 0.35rem;
  padding: 1.2rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.36rem;
  text-align: center;
  line-height: 1.45;
}
.evidence-empty-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 94px;
  margin: 0 0 0.25rem;
}
.evidence-empty-icon svg {
  display: block;
  width: 100%;
  height: auto;
  filter: drop-shadow(0 14px 26px rgba(16, 94, 221, 0.10));
}
.evidence-empty.large strong {
  color: var(--navy);
  font-size: 1rem;
}

.doc-table-card {
  margin: 1rem 0;
  overflow: hidden;
  border: 1px solid rgba(16, 94, 221, 0.12);
  border-radius: 18px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.99), rgba(248,251,255,0.97)),
    #FFFFFF;
  box-shadow: 0 18px 46px rgba(11, 48, 117, 0.10);
}
.st-key-documents_library_search_table_card,
.st-key-ingestion_status_documents_search_table_card {
  margin: 1rem 0;
  overflow: hidden;
  border: 1px solid rgba(16, 94, 221, 0.12);
  border-radius: 18px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.99), rgba(248,251,255,0.97)),
    #FFFFFF;
  box-shadow: 0 18px 46px rgba(11, 48, 117, 0.10);
}
.st-key-documents_library_search_table_card [data-testid="stHorizontalBlock"]:first-child,
.st-key-ingestion_status_documents_search_table_card [data-testid="stHorizontalBlock"]:first-child {
  align-items: center;
  padding: 1.15rem 1.35rem 1.05rem;
  background: linear-gradient(90deg, #FFFFFF, #F6FAFF);
}
.doc-table-header-inline {
  padding: 0;
}
.doc-table-header {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
  padding: 1.15rem 1.35rem 1.05rem;
  background:
    linear-gradient(90deg, #FFFFFF, #F6FAFF);
}
.doc-table-heading {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  min-width: 0;
}
.doc-title-icon {
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.doc-title-icon img {
  width: 34px;
  height: 34px;
  object-fit: contain;
  display: block;
}
.doc-table-title {
  color: var(--navy);
  font-size: 1.2rem;
  font-weight: 900;
  line-height: 1.1;
}
.doc-table-summary {
  color: #64708A;
  font-size: 0.82rem;
  font-weight: 650;
  margin-top: 0.18rem;
}
.doc-table-summary strong {
  color: #405072;
  font-weight: 750;
}
.doc-summary-dot {
  color: #7E8AA7;
  padding: 0 0.35rem;
  font-size: 0;
}
.doc-summary-dot:before {
  content: "\\2022";
  font-size: 0.82rem;
}
.doc-table-actions {
  display: flex;
  align-items: center;
  gap: 0.58rem;
  flex-shrink: 0;
}
.st-key-documents_library_search [data-testid="stWidgetLabel"],
.st-key-ingestion_status_documents_search [data-testid="stWidgetLabel"] {
  display: none;
}
.st-key-documents_library_search input,
.st-key-ingestion_status_documents_search input {
  height: 42px;
  border: 1px solid #CFE1FB !important;
  border-radius: 12px !important;
  background: #FFFFFF !important;
  color: var(--navy) !important;
  padding-left: 0.9rem !important;
  font-size: 0.84rem !important;
  font-weight: 750 !important;
  box-shadow: 0 8px 20px rgba(16,94,221,0.06) !important;
}
.st-key-documents_library_search input:focus,
.st-key-ingestion_status_documents_search input:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(88,172,244,0.22), 0 8px 20px rgba(16,94,221,0.08) !important;
}
.doc-table-scroll {
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0 0.95rem;
  scrollbar-gutter: stable;
}
.doc-table-grid {
  width: max(100%, 1120px);
  min-width: 1120px;
  border: 1px solid #E6EEF9;
  border-radius: 12px;
  overflow: hidden;
  background: #FFFFFF;
}
.doc-table-grid.has-selection {
  width: max(100%, 1170px);
  min-width: 1170px;
}
.doc-table-head,
.doc-table-row {
  display: grid;
  grid-template-columns: minmax(290px, 1fr) 76px 126px 112px 150px 112px 174px;
  align-items: center;
}
.doc-table-grid.has-selection .doc-table-head,
.doc-table-grid.has-selection .doc-table-row {
  grid-template-columns: 48px minmax(290px, 1fr) 76px 126px 112px 150px 112px 174px;
}
.doc-table-head {
  min-height: 48px;
  background: #F7FAFE;
  color: #405072;
  font-size: 0.83rem;
  font-weight: 800;
  border-bottom: 1px solid #E3ECF8;
}
.doc-head-label {
  display: inline-flex;
  align-items: center;
  gap: 0.44rem;
  min-width: 0;
  white-space: nowrap;
}
.doc-head-label img {
  width: 16px;
  height: 16px;
  object-fit: contain;
  display: block;
}
.doc-head-hash {
  width: 20px;
  height: 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #CFE1FB;
  border-radius: 999px;
  color: var(--blue);
  font-size: 0.72rem;
}
.doc-table-head .doc-cell:nth-child(6) .doc-head-hash,
.doc-table-head .doc-cell:nth-child(7) .doc-head-hash {
  font-size: 0;
}
.doc-table-head .doc-cell:nth-child(6) .doc-head-hash:before {
  content: "#";
  font-size: 0.72rem;
}
.doc-table-head .doc-cell:nth-child(7) .doc-head-hash:before {
  content: "\\22EE";
  font-size: 0.9rem;
}
.doc-table-grid.has-selection .doc-table-head .doc-cell:nth-child(7) .doc-head-hash:before {
  content: "#";
  font-size: 0.72rem;
}
.doc-table-grid.has-selection .doc-table-head .doc-cell:nth-child(8) .doc-head-hash {
  font-size: 0;
}
.doc-table-grid.has-selection .doc-table-head .doc-cell:nth-child(8) .doc-head-hash:before {
  content: "\\22EE";
  font-size: 0.9rem;
}
.doc-table-row {
  min-height: 80px;
  border-bottom: 1px solid #E8EFF8;
  background: #FFFFFF;
}
.doc-table-row.is-selected {
  background: linear-gradient(90deg, #EAF4FF, #F7FBFF);
  box-shadow: inset 3px 0 0 var(--blue), inset 0 0 0 1px rgba(16,94,221,0.28);
}
.doc-table-row:nth-child(even) {
  background: #FCFDFF;
}
.doc-table-row.is-selected:nth-child(even) {
  background: linear-gradient(90deg, #EAF4FF, #F7FBFF);
}
.doc-table-row:hover {
  background: #F2F8FF;
}
.doc-table-row.is-selected:hover {
  background: linear-gradient(90deg, #E4F1FF, #F4FAFF);
}
.doc-table-row:last-child {
  border-bottom: 0;
}
.doc-cell {
  box-sizing: border-box;
  min-width: 0;
  height: 100%;
  display: flex;
  align-items: center;
  padding: 0.52rem 0.62rem;
  color: #17233F;
}
.doc-table-head .doc-cell,
.doc-table-row .doc-cell {
  border-right: 1px solid #EAF0F9;
}
.doc-table-head .doc-cell:last-child,
.doc-table-row .doc-cell:last-child {
  border-right: 0;
}
.doc-table-head .doc-cell:last-child,
.doc-table-row .doc-cell:last-child {
  padding-left: 0.45rem;
  padding-right: 0.45rem;
}
.doc-select-cell {
  justify-content: center;
  padding-left: 0.45rem;
  padding-right: 0.45rem;
}
.doc-select-head,
.doc-select-control {
  width: 20px;
  height: 20px;
  border: 2px solid #9CB1D1;
  border-radius: 999px;
  box-sizing: border-box;
}
.doc-select-head {
  display: inline-block;
}
.doc-select-control {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #FFFFFF;
  text-decoration: none;
  transition: border-color 140ms ease, box-shadow 140ms ease, background 140ms ease;
}
.doc-select-control span {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: transparent;
}
.doc-select-control:hover,
.doc-select-control:focus {
  border-color: var(--blue);
  box-shadow: 0 0 0 4px rgba(88,172,244,0.18);
  text-decoration: none;
}
.doc-select-control.is-selected {
  border-color: var(--blue);
  background: var(--blue);
}
.doc-select-control.is-selected span {
  background: #FFFFFF;
}
.doc-main {
  display: flex;
  align-items: center;
  gap: 0.72rem;
  min-width: 0;
}
.doc-file-icon {
  width: 34px;
  height: 38px;
  flex: 0 0 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.doc-file-icon img {
  width: 34px;
  height: 38px;
  object-fit: contain;
  display: block;
}
.doc-file-text {
  min-width: 0;
}
.doc-file-name {
  overflow: hidden;
  color: var(--navy);
  font-size: 0.91rem;
  font-weight: 900;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.doc-file-meta {
  margin-top: 0.18rem;
  color: #71809A;
  font-size: 0.74rem;
  font-weight: 700;
}
.doc-selected-pill {
  display: inline-flex;
  align-items: center;
  width: fit-content;
  margin-top: 0.26rem;
  border: 1px solid #BBD6FF;
  border-radius: 999px;
  background: #EAF4FF;
  color: var(--blue);
  padding: 0.16rem 0.48rem;
  font-size: 0.68rem;
  font-weight: 900;
}
.doc-num {
  color: #17233F;
  font-size: 0.86rem;
  font-weight: 800;
}
.chunk-cell {
  display: flex;
  align-items: center;
  gap: 0.42rem;
}
.chunk-segments {
  display: inline-flex;
  align-items: center;
  gap: 3px;
}
.chunk-segments span {
  width: 11px;
  height: 14px;
  border-radius: 4px;
  background: #DCEBFF;
}
.chunk-segments span.is-filled {
  background: linear-gradient(180deg, #1C77FF, #105EDD);
  box-shadow: 0 4px 8px rgba(16,94,221,0.14);
}
.status-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.3rem;
  width: fit-content;
  max-width: 100%;
  border-radius: 999px;
  padding: 0.32rem 0.5rem;
  font-size: 0.78rem;
  font-weight: 900;
  white-space: nowrap;
}
.status-pill:before {
  content: "\\2713";
  width: 13px;
  height: 13px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 2px solid currentColor;
  border-radius: 999px;
  font-size: 0.58rem;
  line-height: 1;
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
  max-width: 92px;
  overflow: hidden;
  border: 1px solid #D7E6FA;
  border-radius: 10px;
  background: #FFFFFF;
  color: #405072;
  padding: 0.28rem 0.42rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.72rem;
  font-weight: 800;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.doc-row-actions {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.34rem;
  width: 100%;
}
.tiny-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 0.14rem;
  width: 54px;
  min-width: 54px;
  height: 40px;
  min-height: 40px;
  border: 1px solid #CFE1FB;
  border-radius: 12px;
  background: #FFFFFF;
  color: var(--blue);
  font-size: 0.58rem;
  font-weight: 800;
  white-space: nowrap;
  text-decoration: none;
  cursor: pointer;
  box-shadow: 0 8px 18px rgba(16,94,221,0.06);
  transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease, border-color 140ms ease, color 140ms ease;
}
.tiny-action:visited {
  color: var(--blue);
}
.tiny-action:hover {
  background: #F6FAFF;
  border-color: #BBD6FF;
  color: var(--navy);
  text-decoration: none;
  transform: translateY(-1px);
  box-shadow: 0 12px 24px rgba(16,94,221,0.12);
}
.tiny-action:focus-visible {
  outline: 3px solid rgba(88,172,244,0.35);
  outline-offset: 2px;
  border-color: var(--blue);
  text-decoration: none;
}
.tiny-action.alt {
  color: var(--blue);
}
.tiny-action.danger {
  border-color: #FFD3CA;
  color: #E52D18;
}
.tiny-action.danger:visited {
  color: #E52D18;
}
.tiny-action.danger:hover {
  background: #FFF5F2;
  border-color: #FFB5A8;
  color: #BA2B19;
}
.tiny-action img {
  width: 17px;
  height: 17px;
  object-fit: contain;
  display: block;
}
.tiny-action svg {
  width: 17px;
  height: 17px;
  display: block;
  stroke: currentColor;
}
.doc-info-strip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.85rem;
  margin: 0.9rem 1.1rem 1.05rem;
  border: 1px solid #D7E6FA;
  border-radius: 12px;
  background: linear-gradient(90deg, #EFF7FF, #F8FBFF);
  color: #405072;
  padding: 0.68rem 0.9rem;
  font-size: 0.86rem;
  font-weight: 700;
}
.doc-info-copy {
  display: flex;
  align-items: center;
  gap: 0.52rem;
  min-width: 0;
}
.doc-info-copy img {
  width: 21px;
  height: 21px;
  object-fit: contain;
  display: block;
  flex: 0 0 21px;
}
.doc-view-all {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.45rem;
  flex: 0 0 auto;
  border: 1px solid #D7E6FA;
  border-radius: 10px;
  background: #FFFFFF;
  color: var(--blue);
  min-height: 38px;
  padding: 0 0.85rem;
  font-size: 0.84rem;
  font-weight: 800;
  box-shadow: 0 8px 18px rgba(16,94,221,0.06);
}
.doc-view-all span[aria-hidden="true"] {
  font-size: 0;
}
.doc-view-all span[aria-hidden="true"]:before {
  content: "\\203A";
  font-size: 0.96rem;
}
.doc-date {
  color: #1E2A4A;
  font-weight: 500;
  font-size: 0.84rem;
  white-space: nowrap;
}
.doc-empty-row {
  padding: 1.8rem;
  color: #405072;
  background: #FFFFFF;
}
.doc-empty-title {
  color: var(--navy);
  font-size: 1rem;
  font-weight: 900;
}
.doc-empty-copy {
  margin-top: 0.32rem;
  font-size: 0.86rem;
  font-weight: 750;
}

.pdf-modal-overlay {
  position: fixed;
  inset: 0;
  z-index: 999999;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2.2rem;
  background: rgba(5, 9, 20, 0.58);
  backdrop-filter: blur(2px);
}
body.pdf-modal-open {
  overflow: hidden;
}
.pdf-modal-overlay.is-hidden {
  display: none;
}
.pdf-modal-overlay.is-hidden:target {
  display: flex;
}
.pdf-modal-dialog {
  width: min(920px, calc(100vw - 64px)) !important;
  max-width: min(920px, calc(100vw - 64px)) !important;
  max-height: calc(100vh - 48px);
  border-radius: 14px;
  border: 1px solid rgba(215,230,250,0.9);
  box-shadow: 0 30px 90px rgba(2,10,52,0.28);
  overflow: hidden;
  background: #FFFFFF;
}
.pdf-modal-shell {
  max-height: calc(100vh - 48px);
  overflow: auto;
  background: #FFFFFF;
  color: var(--ink);
}
.pdf-modal-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  padding: 1.05rem 1.25rem 0.7rem;
  border-bottom: 0;
}
.pdf-modal-heading {
  display: flex;
  align-items: center;
  gap: 0.78rem;
  min-width: 0;
}
.pdf-modal-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 42px;
  border-radius: 6px;
  background: #F21E1E;
  color: #FFFFFF;
  font-size: 0.62rem;
  font-weight: 900;
  box-shadow: 0 8px 18px rgba(242,30,30,0.14);
}
.pdf-modal-title {
  color: var(--navy);
  font-size: 1.15rem;
  font-weight: 900;
  line-height: 1.18;
  max-width: 540px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.pdf-modal-close {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  border-radius: 999px;
  border: 0;
  background: transparent;
  color: #405072;
  cursor: pointer;
  font-family: inherit;
  text-decoration: none !important;
  border-bottom: 0 !important;
  font-size: 1.55rem;
  line-height: 1;
  padding: 0;
}
.pdf-modal-close:hover {
  background: #F2F6FC;
  color: var(--navy);
  text-decoration: none !important;
  border-bottom: 0 !important;
}
.pdf-modal-close:focus-visible {
  outline: 3px solid rgba(88,172,244,0.35);
  outline-offset: 2px;
  text-decoration: none !important;
  border-bottom: 0 !important;
}
.pdf-modal-pills {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  flex-wrap: wrap;
  padding: 0 1.25rem 1rem;
  border-bottom: 1px solid #E6EEF9;
}
.pdf-modal-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  min-height: 32px;
  border: 1px solid #D7E6FA;
  border-radius: 8px;
  background: #FFFFFF;
  color: var(--navy);
  padding: 0 0.72rem;
  font-size: 0.78rem;
  font-weight: 800;
}
.pdf-modal-pill.is-indexed {
  border-color: rgba(40,143,71,0.24);
  background: #EAF8EF;
  color: #1D7F3B;
}
.pdf-modal-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  min-height: 520px;
}
.pdf-modal-preview {
  padding: 1rem 1.2rem 1rem 1.25rem;
  border-right: 1px solid #E6EEF9;
  background: linear-gradient(180deg, #FFFFFF, #FBFDFF);
}
.pdf-modal-section-title {
  color: var(--blue);
  font-size: 0.78rem;
  font-weight: 900;
  margin-bottom: 0.78rem;
}
.pdf-preview-stage {
  display: grid;
  grid-template-columns: 74px minmax(0, 1fr);
  gap: 0.85rem;
  min-height: 420px;
}
.pdf-thumb-rail {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.7rem;
  max-height: 520px;
  overflow: auto;
  padding-right: 0.18rem;
}
.pdf-thumb-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.32rem;
}
.pdf-thumb {
  width: 58px;
  height: 76px;
  border: 1px solid #D7E6FA;
  border-radius: 6px;
  background: linear-gradient(180deg, #FFFFFF, #F5F8FD);
  box-shadow: 0 8px 18px rgba(11,48,117,0.06);
  position: relative;
  display: block;
  padding: 0;
  overflow: hidden;
  cursor: pointer;
  transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
}
.pdf-thumb.is-active {
  border: 2px solid var(--blue);
  box-shadow: 0 10px 22px rgba(16,94,221,0.18);
}
.pdf-thumb:hover {
  transform: translateY(-1px);
  border-color: #BBD6FF;
  box-shadow: 0 12px 24px rgba(16,94,221,0.13);
}
.pdf-thumb:focus-visible {
  outline: 3px solid rgba(88,172,244,0.35);
  outline-offset: 2px;
}
.pdf-thumb img {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: cover;
  object-position: top center;
}
.pdf-thumb:empty:before,
.pdf-thumb:empty:after {
  content: "";
  position: absolute;
  left: 12px;
  right: 12px;
  height: 3px;
  border-radius: 999px;
  background: #C9D7EC;
}
.pdf-thumb:empty:before { top: 24px; }
.pdf-thumb:empty:after { top: 34px; }
.pdf-thumb-page {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 6px;
  border: 1px solid #D7E6FA;
  background: #FFFFFF;
  color: #405072;
  font-size: 0.72rem;
  font-weight: 900;
  box-shadow: 0 5px 12px rgba(11,48,117,0.05);
  transition: background 140ms ease, border-color 140ms ease, color 140ms ease, box-shadow 140ms ease;
}
.pdf-thumb.is-active + .pdf-thumb-page {
  border-color: var(--blue);
  background: var(--blue);
  color: #FFFFFF;
  box-shadow: 0 8px 16px rgba(16,94,221,0.18);
}
.pdf-frame-shell {
  min-height: 420px;
  border: 1px solid #DDE8F7;
  border-radius: 8px;
  background: #FFFFFF;
  box-shadow: 0 18px 38px rgba(11,48,117,0.10);
  overflow: hidden;
}
.pdf-page-scroll {
  height: 520px;
  overflow: auto;
  overscroll-behavior: contain;
  background: #EAF0F8;
  padding: 1rem;
}
.pdf-page-stack {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  width: 100%;
  min-width: 100%;
}
.pdf-modal-overlay.is-zoomed-in .pdf-page-stack {
  align-items: flex-start;
}
.pdf-page-image {
  display: block;
  width: 100%;
  flex: 0 0 auto;
  max-width: none !important;
  height: auto;
  border-radius: 4px;
  background: #FFFFFF;
  box-shadow: 0 10px 28px rgba(2,10,52,0.16);
  transition: width 120ms ease;
}
.pdf-modal-overlay.is-focus-zoom .pdf-page-image {
  cursor: zoom-in;
}
.pdf-preview-iframe {
  width: 100%;
  height: 520px;
  border: 0;
  display: block;
  background: #FFFFFF;
}
.pdf-preview-fallback .pdf-modal-note {
  margin: 0;
  border-width: 0 0 1px;
  border-radius: 0;
}
.pdf-missing-source {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 420px;
  padding: 2rem;
  color: #405072;
  text-align: center;
  font-size: 0.92rem;
  font-weight: 750;
}
.pdf-preview-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-top: 0.82rem;
  color: #405072;
  font-size: 0.78rem;
  font-weight: 750;
}
.pdf-modal-icon {
  display: block;
  object-fit: contain;
  flex: 0 0 auto;
}
.pdf-page-nav {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  min-height: 34px;
}
.pdf-page-nav-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  border: 1px solid #D7E6FA;
  border-radius: 7px;
  background: #FFFFFF;
  color: var(--navy);
  padding: 0;
  cursor: pointer;
  box-shadow: 0 7px 15px rgba(11,48,117,0.06);
  transition: background 140ms ease, border-color 140ms ease, transform 140ms ease, opacity 140ms ease;
}
.pdf-page-nav-button:hover:not(:disabled) {
  background: #F6FAFF;
  border-color: #BBD6FF;
  transform: translateY(-1px);
}
.pdf-page-nav-button:active:not(:disabled) {
  transform: translateY(0);
}
.pdf-page-nav-button:focus-visible {
  outline: 3px solid rgba(88,172,244,0.35);
  outline-offset: 1px;
}
.pdf-page-nav-button:disabled {
  opacity: 0.42;
  cursor: not-allowed;
  box-shadow: none;
}
.pdf-preview-controls {
  display: inline-flex;
  align-items: center;
  gap: 0.34rem;
  border: 1px solid #D7E6FA;
  border-radius: 8px;
  background: #FFFFFF;
  padding: 0.28rem 0.34rem;
  color: var(--navy);
  box-shadow: 0 8px 18px rgba(11,48,117,0.06);
}
.pdf-zoom-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  color: #405072;
}
.pdf-zoom-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: 1px solid transparent;
  border-radius: 6px;
  background: transparent;
  color: var(--navy);
  font-size: 1rem;
  line-height: 1;
  font-weight: 900;
  cursor: pointer;
  transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
}
.pdf-zoom-focus-button {
  border-color: transparent;
}
.pdf-zoom-focus-button.is-active,
.pdf-zoom-focus-button[aria-pressed="true"] {
  background: #ECF4FF;
  border-color: #BBD6FF;
  box-shadow: inset 0 0 0 1px rgba(16,94,221,0.08);
}
.pdf-zoom-focus-button.is-active .pdf-control-icon,
.pdf-zoom-focus-button[aria-pressed="true"] .pdf-control-icon {
  opacity: 1;
}
.pdf-control-icon {
  width: 16px;
  height: 16px;
  display: block;
  object-fit: contain;
  pointer-events: none;
}
.pdf-zoom-icon .pdf-control-icon {
  width: 17px;
  height: 17px;
  opacity: 0.84;
}
.pdf-control-fallback {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  font-size: 0.74rem;
  font-weight: 900;
  line-height: 1;
}
.pdf-zoom-button:hover:not(:disabled) {
  background: #F6FAFF;
  border-color: #BBD6FF;
  transform: translateY(-1px);
}
.pdf-zoom-button:active:not(:disabled) {
  transform: translateY(0);
}
.pdf-zoom-button:focus-visible {
  outline: 3px solid rgba(88,172,244,0.35);
  outline-offset: 1px;
}
.pdf-zoom-button:disabled {
  color: #9AA8BD;
  cursor: not-allowed;
}
.pdf-zoom-label {
  min-width: 44px;
  color: var(--navy);
  text-align: center;
  font-size: 0.78rem;
  font-weight: 900;
}
.pdf-modal-details {
  padding: 1.1rem 1rem;
  background: #FFFFFF;
}
.pdf-details-title {
  color: var(--navy);
  font-size: 0.86rem;
  font-weight: 900;
  margin-bottom: 0.9rem;
}
.pdf-detail-row {
  display: grid;
  grid-template-columns: 20px minmax(0, 1fr) minmax(88px, auto);
  gap: 0.48rem 0.65rem;
  align-items: center;
  padding: 0.43rem 0;
  color: #405072;
  font-size: 0.72rem;
}
.pdf-detail-icon {
  width: 18px;
  height: 18px;
  opacity: 0.9;
}
.pdf-detail-icon.is-empty {
  width: 18px;
  height: 18px;
}
.pdf-detail-label {
  color: #405072;
  font-weight: 750;
}
.pdf-detail-value {
  color: var(--navy);
  font-weight: 800;
  text-align: right;
  word-break: break-word;
}
.pdf-detail-value.is-yes {
  color: #1D7F3B;
}
.pdf-modal-note {
  border: 1px solid #CFE1FB;
  border-radius: 8px;
  background: #ECF4FF;
  color: #405072;
  padding: 0.72rem;
  font-size: 0.74rem;
  font-weight: 650;
  line-height: 1.45;
  margin: 0.9rem 0 1rem;
}
.pdf-preview-note {
  display: flex;
  align-items: flex-start;
  gap: 0.55rem;
}
.pdf-preview-note span {
  min-width: 0;
}
.pdf-preview-info-icon {
  width: 18px;
  height: 18px;
  margin-top: 0.02rem;
}
.pdf-preview-info-icon.is-empty {
  width: 18px;
  height: 18px;
}
.pdf-modal-actions-title {
  color: var(--navy);
  font-size: 0.78rem;
  font-weight: 900;
  margin-bottom: 0.55rem;
}
.pdf-modal-action {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 38px;
  width: 100%;
  border: 1px solid #CFE1FB;
  border-radius: 8px;
  background: #FFFFFF;
  color: var(--blue);
  text-decoration: none !important;
  border-bottom: 0 !important;
  font-size: 0.78rem;
  font-weight: 900;
  margin: 0.5rem 0;
}
.pdf-modal-action:hover,
.pdf-modal-action:focus,
.pdf-modal-action:visited {
  text-decoration: none !important;
  border-bottom: 0 !important;
}
.pdf-modal-action:focus-visible {
  outline: 3px solid rgba(88,172,244,0.35);
  outline-offset: 2px;
}
.pdf-modal-action.primary {
  border-color: var(--blue);
  background: var(--blue);
  color: #FFFFFF;
}
.pdf-modal-action.is-disabled {
  opacity: 0.55;
  pointer-events: none;
}
@media (max-width: 900px) {
  .pdf-modal-overlay {
    align-items: flex-start;
    padding: 1rem;
    overflow: auto;
  }
  .pdf-modal-layout {
    grid-template-columns: 1fr;
  }
  .pdf-modal-preview {
    border-right: 0;
    border-bottom: 1px solid #E6EEF9;
  }
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

.collection-stats-card {
  margin: 1rem 0 1.2rem;
  padding: 1.15rem;
  border: 1px solid var(--line);
  border-radius: 16px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,251,255,0.94)),
    #FFFFFF;
  box-shadow: var(--shadow);
}
.collection-stats-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1rem;
}
.collection-stats-title {
  color: var(--navy);
  font-size: 1.05rem;
  font-weight: 900;
}
.collection-stats-copy {
  color: #405072;
  font-size: 0.9rem;
  margin-top: 0.18rem;
}
.collection-stats-badge {
  flex: 0 0 auto;
  border: 1px solid #CFE1FB;
  border-radius: 999px;
  background: #F6FAFF;
  color: var(--blue);
  padding: 0.32rem 0.62rem;
  font-size: 0.76rem;
  font-weight: 900;
}
.collection-stats-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
  margin-bottom: 1rem;
}
.collection-stat-tile {
  border: 1px solid #E4ECF7;
  border-radius: 14px;
  background: #FFFFFF;
  padding: 0.9rem;
}
.collection-stat-label {
  color: #64708A;
  font-size: 0.78rem;
  font-weight: 850;
}
.collection-stat-value {
  color: var(--navy);
  font-size: 1.65rem;
  line-height: 1.05;
  font-weight: 950;
  margin-top: 0.25rem;
}
.collection-stat-helper {
  color: #405072;
  font-size: 0.78rem;
  font-weight: 750;
  margin-top: 0.35rem;
}
.collection-stats-section-title {
  color: var(--navy);
  font-size: 0.88rem;
  font-weight: 900;
  margin: 0.9rem 0 0.55rem;
}
.collection-doc-list {
  border: 1px solid #E6EEF9;
  border-radius: 12px;
  overflow: hidden;
  background: #FFFFFF;
}
.collection-doc-row {
  display: grid;
  grid-template-columns: minmax(220px, 1fr) 86px 92px 160px;
  gap: 0.85rem;
  align-items: center;
  padding: 0.78rem 0.9rem;
  border-top: 1px solid #EDF3FB;
}
.collection-doc-row:first-child {
  border-top: 0;
}
.collection-doc-name {
  color: var(--ink);
  font-size: 0.9rem;
  font-weight: 850;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.collection-doc-meta,
.collection-doc-date {
  color: #64708A;
  font-size: 0.78rem;
  font-weight: 750;
}
.collection-doc-date {
  text-align: right;
}
.collection-mini-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(16,94,221,0.18);
  border-radius: 999px;
  background: #EAF3FF;
  color: var(--blue);
  padding: 0.24rem 0.5rem;
  font-size: 0.75rem;
  font-weight: 900;
  white-space: nowrap;
}
.collection-bars {
  display: grid;
  gap: 0.62rem;
}
.collection-bar-row {
  display: grid;
  grid-template-columns: minmax(160px, 0.8fr) minmax(180px, 1.2fr) 48px;
  gap: 0.75rem;
  align-items: center;
}
.collection-bar-label {
  color: #405072;
  font-size: 0.82rem;
  font-weight: 800;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.collection-bar-track {
  height: 9px;
  border-radius: 999px;
  background: #EAF1FA;
  overflow: hidden;
}
.collection-bar-fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--blue), var(--sky));
}
.collection-bar-value {
  color: var(--navy);
  font-size: 0.82rem;
  font-weight: 900;
  text-align: right;
}
.collection-empty {
  border: 1px dashed #B7D1F8;
  border-radius: 12px;
  background: #F6FAFF;
  color: #405072;
  padding: 1rem;
  font-weight: 750;
}

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
.documents-section-head {
  position: relative;
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 0.55rem 0 1rem;
  overflow: hidden;
}
.documents-section-title {
  color: var(--navy);
  font-size: 1.55rem;
  line-height: 1.1;
  font-weight: 900;
  flex: 0 0 auto;
}
.documents-section-rule {
  height: 1px;
  flex: 1 1 auto;
  background: linear-gradient(90deg, #CAD8EA, rgba(202,216,234,0));
}
.documents-circuit {
  position: absolute;
  right: 0;
  top: -18px;
  width: min(430px, 40vw);
  height: 92px;
  pointer-events: none;
  opacity: 0.42;
  background:
    radial-gradient(circle at 88% 28%, transparent 0 3px, rgba(16,94,221,0.4) 4px, transparent 5px),
    radial-gradient(circle at 70% 55%, transparent 0 3px, rgba(16,94,221,0.35) 4px, transparent 5px),
    linear-gradient(135deg, transparent 0 48%, rgba(88,172,244,0.38) 49% 51%, transparent 52%),
    repeating-linear-gradient(0deg, transparent 0 18px, rgba(88,172,244,0.34) 19px, transparent 20px);
  clip-path: polygon(18% 16%, 100% 16%, 100% 84%, 7% 84%, 28% 58%, 68% 58%, 78% 45%, 54% 45%, 46% 32%, 18% 32%);
}
.documents-upload-card,
.st-key-documents_upload_card,
.documents-progress-card,
.selected-document-card {
  margin: 0 0 1rem;
  padding: 1.15rem 1.25rem;
  border: 1px solid rgba(16,94,221,0.13);
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,251,255,0.96));
  box-shadow: 0 18px 46px rgba(11, 48, 117, 0.09);
  box-sizing: border-box;
  max-width: 100%;
  overflow: hidden;
}
.documents-card-title {
  color: var(--navy);
  font-size: 1.04rem;
  font-weight: 900;
  margin-bottom: 0.8rem;
}
.st-key-documents_upload_card [data-testid="stMarkdownContainer"] p {
  margin: 0;
}
.st-key-documents_upload_card {
  overflow: visible;
}
.st-key-documents_upload_zone {
  position: relative;
  max-width: 100%;
  min-height: 178px;
  margin: 0;
  padding: 0.82rem;
  border: 1.5px dashed #6CA8FF;
  border-radius: 14px;
  background: linear-gradient(180deg, #FFFFFF, #F5FAFF);
  color: #405072;
  text-align: center;
  box-sizing: border-box;
  overflow: hidden;
}
.st-key-documents_upload_zone:hover {
  border-color: var(--blue);
  background: linear-gradient(180deg, #FFFFFF, #EFF7FF);
}
.st-key-documents_upload_input_layer {
  position: absolute;
  top: 0.82rem;
  right: 0.82rem;
  left: 0.82rem;
  z-index: 4;
  height: 150px;
  opacity: 0;
  overflow: hidden;
}
.st-key-documents_upload_zone:has(.documents-upload-pending) .st-key-documents_upload_input_layer {
  height: 94px;
}
.st-key-documents_upload_input_layer .stElementContainer,
.st-key-documents_upload_input_layer [class*="stElementContainer"] {
  height: 100% !important;
  min-height: 100% !important;
  overflow: hidden !important;
}
.st-key-documents_upload_input_layer [data-testid="stFileUploader"] {
  height: 100% !important;
  min-height: 100% !important;
  margin: 0;
  overflow: hidden;
}
.st-key-documents_upload_input_layer [data-testid="stFileUploaderDropzone"] {
  position: absolute;
  inset: 0;
  z-index: 1;
  width: 100%;
  height: 100% !important;
  min-height: 100% !important;
  margin: 0;
  border: 0;
  background: transparent;
  cursor: pointer;
}
.st-key-documents_upload_input_layer [data-testid="stFileUploaderDropzone"] button {
  position: absolute !important;
  inset: 0 !important;
  z-index: 2 !important;
  width: 100% !important;
  height: 100% !important;
  min-height: 100% !important;
  margin: 0 !important;
  cursor: pointer !important;
}
.st-key-documents_upload_input_layer [data-testid="stFileUploaderDropzone"] svg,
.st-key-documents_upload_input_layer [data-testid="stFileUploaderDropzoneInstructions"],
.st-key-documents_upload_input_layer [data-testid="stFileUploader"] [data-testid*="FileUploaderFile"],
.st-key-documents_upload_input_layer [data-testid="stFileUploader"] [data-testid*="stFileUploaderFile"],
.st-key-documents_upload_input_layer [data-testid="stFileUploader"] [data-testid*="UploadedFile"] {
  display: none !important;
}
.documents-upload-visual {
  position: relative;
  z-index: 1;
  min-height: 150px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 0.55rem;
  color: #405072;
  text-align: center;
  font-weight: 700;
  pointer-events: none;
}
.documents-upload-visual.has-file {
  min-height: 0;
}
.documents-upload-cloud {
  width: 122px;
  height: 96px;
  display: block;
  margin: 0 auto 0.36rem;
  object-fit: contain;
}
.documents-upload-visual.has-file .documents-upload-cloud {
  width: 104px;
  height: 76px;
  margin-bottom: 0;
}
.documents-upload-pending {
  margin: 0 auto;
  width: min(100%, 430px);
}
.documents-upload-file {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.62rem;
  min-height: 48px;
  border: 1px solid #DCE8F7;
  border-radius: 12px;
  background: #FFFFFF;
  color: #405072;
  padding: 0.5rem 0.75rem;
  box-shadow: 0 10px 22px rgba(16,94,221,0.06);
  box-sizing: border-box;
}
.documents-upload-file img {
  width: 28px;
  height: 28px;
  object-fit: contain;
  display: block;
  flex: 0 0 28px;
}
.documents-upload-file-name {
  max-width: min(290px, 58vw);
  overflow: hidden;
  color: #1E2A4A;
  font-size: 0.86rem;
  font-weight: 900;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.documents-upload-file-size {
  color: #71809A;
  font-size: 0.75rem;
  font-weight: 800;
  text-align: center;
}
.documents-upload-helper {
  color: #405072;
  font-size: 0.88rem;
  font-weight: 800;
  line-height: 1.2;
  margin: 0.5rem 0 0.64rem;
}
.documents-upload-visual:not(.has-file) .documents-upload-helper {
  font-size: 0.92rem;
  font-weight: 750;
  margin: 0;
}
.st-key-documents_upload_submit,
.st-key-documents_upload_cancel {
  position: relative;
  z-index: 6;
}
.st-key-documents_upload_submit button {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 0.52rem !important;
  min-height: 48px !important;
  height: 48px !important;
  background: #FFFFFF !important;
  border-color: #BBD6FF !important;
  color: var(--navy) !important;
  box-shadow: 0 12px 28px rgba(16,94,221,0.10) !important;
}
.st-key-documents_upload_submit button:hover {
  background: #F6FAFF !important;
  border-color: #8EBBFF !important;
  color: var(--blue) !important;
}
.st-key-documents_upload_cancel button {
  min-height: 48px !important;
  height: 48px !important;
  min-width: 2.65rem !important;
  padding: 0 !important;
  border-color: #FFD3CA !important;
  color: #E52D18 !important;
  background: #FFFFFF !important;
  font-size: 1.15rem !important;
  box-shadow: 0 10px 22px rgba(200,71,44,0.06) !important;
}
.st-key-documents_upload_cancel button:hover {
  background: #FFF5F2 !important;
  border-color: #FFB5A8 !important;
  color: #BA2B19 !important;
}
.documents-badges {
  display: grid;
  grid-template-columns: minmax(78px, 0.76fr) minmax(124px, 1.2fr) minmax(112px, 1.08fr);
  gap: 0.32rem;
  margin: 0.62rem 0 0.1rem;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
}
.documents-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.22rem;
  min-width: 0;
  min-height: 34px;
  border: 1px solid #DCE8F7;
  border-radius: 10px;
  background: #FFFFFF;
  color: #405072;
  padding: 0.32rem 0.26rem;
  font-size: 0.68rem;
  font-weight: 850;
  line-height: 1;
  white-space: nowrap;
  box-sizing: border-box;
  overflow: visible;
}
.documents-badge span {
  min-width: max-content;
  overflow: visible;
  text-overflow: clip;
}
.documents-badge img {
  width: 16px;
  height: 16px;
  object-fit: contain;
  display: block;
  flex: 0 0 16px;
}
.documents-badge.is-red { color: #D92013; }
.documents-badge.is-blue { color: var(--blue); }
.documents-badge.is-green { color: #1D8B42; }
.progress-pipeline {
  display: flex;
  flex-direction: column;
}
.pipeline-row {
  display: grid;
  grid-template-columns: 26px minmax(150px, 1fr) minmax(92px, auto);
  gap: 0.62rem;
  align-items: center;
  min-height: 44px;
  border-bottom: 1px solid #E7EFF9;
  color: #405072;
  font-size: 0.84rem;
  font-weight: 750;
}
.pipeline-row:last-child { border-bottom: 0; }
.pipeline-dot {
  width: 21px;
  height: 21px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 2px solid #8BA2C5;
  border-radius: 999px;
  color: #8BA2C5;
  font-size: 0.68rem;
  font-weight: 900;
}
.pipeline-row.is-complete .pipeline-dot {
  border-color: #0A9B3F;
  background: #0A9B3F;
  color: #FFFFFF;
}
.pipeline-row.is-active .pipeline-dot {
  border-color: var(--blue);
  color: var(--blue);
  box-shadow: 0 0 0 4px rgba(16,94,221,0.10);
}
.pipeline-value {
  color: var(--navy);
  text-align: right;
  white-space: nowrap;
}
.ingestion-progress-notice {
  margin-top: 1rem;
  border-radius: 12px;
  padding: 0.78rem 0.9rem;
  font-size: 0.9rem;
  font-weight: 800;
  line-height: 1.35;
}
.ingestion-progress-notice.is-success {
  background: #E7F8EE;
  color: #087D32;
}
.ingestion-progress-notice.is-warning {
  background: #FFF6DF;
  color: #8A6200;
}
.ingestion-progress-notice.is-error {
  background: #FFF1EE;
  color: #B42318;
}
.ingestion-progress-notice.is-info {
  background: #EAF3FF;
  color: var(--blue);
}
.documents-metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1rem;
  margin: 0 0 1rem;
}
.documents-metric-card {
  position: relative;
  min-height: 128px;
  overflow: hidden;
  display: flex;
  align-items: center;
  gap: 0.9rem;
  padding: 1.05rem 1.15rem;
  border: 1px solid #DFE7F3;
  border-left: 3px solid var(--accent);
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,251,255,0.95));
  box-shadow: 0 16px 38px rgba(11, 48, 117, 0.09);
}
.documents-metric-card.is-warm { --accent: #FF3B16; --helper: #FF3B16; --icon-bg: #FFF0EA; }
.documents-metric-card.is-cool { --accent: #105EDD; --helper: #105EDD; --icon-bg: #EAF3FF; }
.documents-metric-card.is-gold { --accent: #F5B400; --helper: #A87400; --icon-bg: #FFF7DF; }
.documents-metric-icon {
  width: 58px;
  height: 58px;
  flex: 0 0 58px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: var(--icon-bg);
  color: var(--accent);
}
.documents-metric-icon svg {
  width: 31px;
  height: 31px;
  stroke: currentColor;
}
.documents-metric-label {
  color: var(--navy);
  font-size: 0.88rem;
  font-weight: 900;
}
.documents-metric-value {
  color: #020A34;
  font-size: 2.35rem;
  line-height: 1;
  font-weight: 900;
  margin-top: 0.25rem;
}
.documents-metric-helper {
  color: var(--helper);
  font-size: 0.78rem;
  font-weight: 850;
  margin-top: 0.65rem;
}
.selected-document-card {
  position: sticky;
  top: 1rem;
  width: calc(100% + 3rem);
  max-width: none;
  margin-left: -1.5rem;
  margin-right: -1.5rem;
}
.selected-document-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.7rem;
}
.selected-close {
  width: 30px;
  height: 30px;
  flex: 0 0 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #405072;
  text-decoration: none !important;
  font-size: 1.35rem;
  line-height: 0.85;
  margin-top: -0.16rem;
}
.selected-close:hover,
.selected-close:focus,
.selected-close:visited {
  color: #0B3075;
  text-decoration: none !important;
}
.st-key-selected_document_panel_shell {
  position: relative;
}
.st-key-selected_document_close {
  position: absolute !important;
  top: 0.88rem;
  right: -0.25rem;
  z-index: 4;
  width: 30px !important;
  height: 30px !important;
}
.st-key-selected_document_close [data-testid="stButton"] {
  width: 30px !important;
  height: 30px !important;
}
.st-key-selected_document_close button {
  width: 30px !important;
  height: 30px !important;
  min-height: 30px !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 999px !important;
  background: transparent !important;
  color: #405072 !important;
  box-shadow: none !important;
}
.st-key-selected_document_close button:hover,
.st-key-selected_document_close button:focus,
.st-key-selected_document_close button:focus-visible {
  color: #0B3075 !important;
  background: #F2F6FC !important;
  box-shadow: none !important;
  outline: 3px solid rgba(88,172,244,0.35) !important;
  outline-offset: 2px !important;
}
.st-key-selected_document_close button p {
  color: inherit !important;
  font-size: 1.55rem !important;
  font-weight: 400 !important;
  line-height: 1 !important;
  margin: 0 !important;
}
.selected-doc-identity {
  display: flex;
  align-items: center;
  gap: 0.72rem;
  margin: 0.8rem 0 0.75rem;
}
.selected-pdf-mark {
  width: 42px;
  height: 48px;
  flex: 0 0 42px;
  border-radius: 8px;
  background: #E51912;
  color: #FFFFFF;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.72rem;
  font-weight: 900;
}
.selected-doc-name {
  color: var(--navy);
  font-weight: 900;
  line-height: 1.18;
  word-break: break-word;
}
.selected-doc-size {
  color: #405072;
  font-size: 0.78rem;
  font-weight: 750;
  margin-top: 0.2rem;
}
.selected-meta-row {
  display: grid;
  grid-template-columns: 24px minmax(104px, 0.9fr) minmax(96px, 1.2fr);
  gap: 0.56rem;
  align-items: center;
  min-height: 40px;
  border-bottom: 1px solid #E7EFF9;
  color: #405072;
  font-size: 0.78rem;
  font-weight: 750;
}
.selected-meta-row:last-child { border-bottom: 0; }
.selected-meta-icon {
  width: 20px;
  height: 20px;
  object-fit: contain;
  justify-self: center;
}
.selected-meta-icon.is-empty {
  display: inline-block;
}
.selected-meta-label {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.selected-meta-value {
  color: var(--navy);
  text-align: right;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.selected-preview {
  margin-top: 1rem;
  border: 1px solid #E3ECF8;
  border-radius: 12px;
  background: #F8FBFF;
  overflow: hidden;
}
.selected-preview-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.58rem 0.75rem;
  color: var(--navy);
  font-size: 0.8rem;
  font-weight: 900;
}
.selected-preview img {
  width: 100%;
  display: block;
  background: #FFFFFF;
  border-top: 1px solid #E3ECF8;
}
.selected-preview-empty {
  padding: 1.1rem 0.75rem;
  color: #64708A;
  font-size: 0.82rem;
  font-weight: 750;
  text-align: center;
}
.selected-delete {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  min-height: 48px;
  margin-top: 1rem;
  border: 1px solid #F3442C;
  border-radius: 10px;
  background: #FFFFFF;
  color: #E52D18;
  font-size: 0.9rem;
  font-weight: 900;
  text-decoration: none;
}
.selected-delete:visited { color: #E52D18; }
.selected-delete:hover {
  background: #FFF5F2;
  color: #BA2B19;
  text-decoration: none;
}
.selected-delete-copy {
  margin-top: 0.55rem;
  color: #405072;
  font-size: 0.76rem;
  font-weight: 750;
}
.document-delete-modal-open {
  overflow: hidden;
}
.document-delete-modal-overlay {
  position: fixed;
  inset: 0;
  z-index: 100000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
  background: rgba(11, 18, 34, 0.48);
  backdrop-filter: blur(3px);
}
.document-delete-modal {
  width: min(528px, calc(100vw - 2rem));
  max-height: calc(100vh - 3rem);
  overflow: auto;
  border-radius: 18px;
  background: #FFFFFF;
  box-shadow: 0 24px 70px rgba(15, 23, 42, 0.26);
}
.delete-confirm-panel {
  margin: 0;
  padding: 1.35rem 1.45rem 1.2rem;
  border: 0;
  border-radius: 18px;
  background: #FFFFFF;
}
.delete-confirm-header {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 0.8rem;
  margin-bottom: 0.78rem;
}
.delete-confirm-badge {
  width: 38px;
  height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: #FFF1ED;
  color: #EF2B18;
}
.delete-confirm-badge svg {
  width: 20px;
  height: 20px;
}
.delete-confirm-title {
  color: #020A34;
  font-size: 1.32rem;
  font-weight: 950;
  line-height: 1.15;
}
.delete-modal-close {
  width: 34px;
  height: 34px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: #31415F;
  font-size: 1.55rem;
  line-height: 1;
  cursor: pointer;
}
.delete-modal-close:hover,
.delete-modal-close:focus-visible {
  background: #F2F6FC;
  outline: 2px solid rgba(16, 94, 221, 0.22);
  outline-offset: 2px;
}
.delete-confirm-copy {
  color: #243451;
  font-size: 0.9rem;
  line-height: 1.48;
}
.delete-summary-card {
  display: flex;
  gap: 0.72rem;
  align-items: center;
  margin: 0.88rem 0 0.95rem;
  padding: 0.75rem;
  border: 1px solid #D9E4F3;
  border-radius: 10px;
  background: #FBFDFF;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}
.delete-summary-card .selected-pdf-mark {
  width: 42px;
  height: 42px;
  flex: 0 0 42px;
}
.delete-check-row {
  display: flex;
  align-items: flex-start;
  gap: 0.62rem;
  color: #243451;
  font-size: 0.88rem;
  font-weight: 780;
  margin: 0.62rem 0;
}
.delete-check-row strong {
  color: #17233D;
  font-weight: 950;
}
.delete-check-dot {
  width: 18px;
  height: 18px;
  flex: 0 0 18px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-top: 0.24rem;
  border-radius: 999px;
  background: #0A9B3F;
  color: #FFFFFF;
  font-size: 0.7rem;
  font-weight: 950;
  line-height: 1;
}
.delete-check-dot.info {
  background: #FFFFFF;
  border: 1px solid #8BA2C5;
  color: #405072;
  line-height: 1;
  transform: translateY(1px);
}
.delete-warning {
  margin-top: 0.9rem;
  padding-top: 0.78rem;
  border-top: 1px solid #E6ECF5;
  color: #E52D18;
  font-size: 0.84rem;
  font-weight: 900;
}
.delete-modal-footer {
  display: grid;
  grid-template-columns: 1fr 1.28fr;
  gap: 0.75rem;
  margin-top: 1rem;
}
.delete-modal-cancel,
.delete-modal-confirm {
  min-height: 42px;
  border-radius: 9px;
  font-size: 0.88rem;
  font-weight: 900;
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.48rem;
  cursor: pointer;
}
.delete-modal-cancel {
  border: 1px solid #D5E2F4;
  background: #FFFFFF;
  color: #243451;
}
.delete-modal-confirm,
.delete-modal-confirm:visited {
  border: 1px solid #EF2B18;
  background: #EF2B18;
  color: #FFFFFF;
}
.delete-modal-confirm svg {
  width: 17px;
  height: 17px;
}
.delete-modal-cancel:hover,
.delete-modal-cancel:focus-visible {
  background: #F7FAFF;
  outline: 2px solid rgba(16, 94, 221, 0.18);
  outline-offset: 2px;
}
.delete-modal-confirm:hover,
.delete-modal-confirm:focus-visible {
  background: #D92817;
  border-color: #D92817;
  color: #FFFFFF;
  text-decoration: none;
  outline: 2px solid rgba(239, 43, 24, 0.22);
  outline-offset: 2px;
}
.document-delete-modal.is-deleting .delete-modal-close,
.document-delete-modal.is-deleting .delete-modal-cancel {
  opacity: 0.48;
  cursor: not-allowed;
}
.document-delete-modal.is-deleting .delete-modal-confirm {
  pointer-events: none;
  background: #BA2B19;
  border-color: #BA2B19;
}
.st-key-document_delete_triggers,
.st-key-document_selection_triggers {
  position: fixed !important;
  left: -10000px !important;
  top: auto !important;
  width: 1px !important;
  height: 1px !important;
  overflow: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
}
.delete-button-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.42);
  border-top-color: #FFFFFF;
  border-radius: 999px;
  animation: deleteSpin 0.75s linear infinite;
}
.delete-processing-note {
  margin-top: 0.95rem;
  padding: 0.72rem 0.82rem;
  border-radius: 10px;
  background: #FFF7F4;
  color: #BA2B19;
  font-size: 0.84rem;
  font-weight: 850;
  text-align: center;
}
@keyframes deleteSpin {
  to { transform: rotate(360deg); }
}
.delete-result-panel {
  text-align: center;
  padding: 0.45rem 0.2rem 0.05rem;
}
.polished-delete-result {
  max-width: 430px;
  margin: 0 auto;
}
div[role="dialog"]:has(.polished-delete-result) h2 {
  display: none !important;
}
div[role="dialog"]:has(.polished-delete-result) [data-testid="stDialogHeader"] {
  min-height: 0 !important;
  padding-bottom: 0 !important;
}
.delete-result-hero {
  position: relative;
  width: 112px;
  height: 86px;
  margin: 0.15rem auto 0.4rem;
  display: flex;
  align-items: center;
  justify-content: center;
}
.delete-result-icon {
  width: 62px;
  height: 62px;
  margin: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  color: #FFFFFF;
  font-size: 2rem;
  font-weight: 950;
  box-shadow: 0 10px 24px rgba(10, 155, 63, 0.22);
}
.delete-result-icon.success {
  background: #0A9B3F;
  outline: 8px solid rgba(10, 155, 63, 0.13);
}
.delete-result-icon.error {
  background: #E52D18;
}
.delete-success-sparkle {
  position: absolute;
  color: #0A9B3F;
  font-size: 1rem;
  font-weight: 950;
}
.delete-success-sparkle.one {
  left: 9px;
  top: 33px;
}
.delete-success-sparkle.two {
  right: 5px;
  top: 31px;
}
.delete-success-sparkle.three {
  right: 28px;
  top: 9px;
  color: #D7B83D;
  font-size: 0.8rem;
}
.delete-result-title {
  color: #020A34;
  font-size: 1.32rem;
  font-weight: 950;
}
.delete-result-copy {
  margin-top: 0.75rem;
  color: #243451;
  font-size: 0.95rem;
  font-weight: 900;
}
.delete-result-detail {
  margin-top: 0.32rem;
  color: #6A7894;
  font-size: 0.86rem;
  font-weight: 820;
}
.delete-result-summary-card {
  display: flex;
  align-items: center;
  gap: 0.72rem;
  margin: 1.05rem 0 0.9rem;
  padding: 0.74rem 0.82rem;
  border: 1px solid #D9E4F3;
  border-radius: 10px;
  background: #FBFDFF;
  text-align: left;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}
.delete-result-summary-card .selected-pdf-mark {
  width: 42px;
  height: 42px;
  flex: 0 0 42px;
}
.st-key-delete_result_done button {
  background: #105EDD !important;
  border-color: #105EDD !important;
  color: #FFFFFF !important;
  box-shadow: none !important;
}
.st-key-delete_result_done button:hover,
.st-key-delete_result_done button:focus,
.st-key-delete_result_done button:focus-visible {
  background: #0B4AB7 !important;
  border-color: #0B4AB7 !important;
  color: #FFFFFF !important;
  outline: 2px solid rgba(16, 94, 221, 0.22) !important;
  outline-offset: 2px !important;
  box-shadow: none !important;
}
.st-key-confirm_delete_doc button,
.st-key-inline_confirm_delete_doc button,
.st-key-dialog_confirm_delete_doc button {
  background: #E52D18 !important;
  border-color: #E52D18 !important;
  color: #FFFFFF !important;
}
.st-key-confirm_delete_doc button:hover,
.st-key-inline_confirm_delete_doc button:hover,
.st-key-dialog_confirm_delete_doc button:hover {
  background: #BA2B19 !important;
  border-color: #BA2B19 !important;
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
  .st-key-app_header_shell [data-testid="stHorizontalBlock"] {
    gap: 0.6rem;
  }
  .app-header { flex-direction: column; }
  .app-title {
    font-size: 2.05rem;
    white-space: normal;
  }
  .st-key-header_actions {
    justify-content: flex-start;
    padding-top: 0;
    flex-wrap: wrap;
  }
  .hero-title { font-size: 2.05rem; }
  .ingestion-status-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .workflow, .debug-grid { grid-template-columns: 1fr 1fr; }
  .collection-stats-grid { grid-template-columns: 1fr; }
  .collection-doc-row {
    grid-template-columns: 1fr 86px;
  }
  .collection-doc-date {
    grid-column: 1 / -1;
    text-align: left;
  }
  .collection-bar-row {
    grid-template-columns: 1fr;
    gap: 0.35rem;
  }
  .collection-bar-value {
    text-align: left;
  }
  .chat-user { max-width: 92%; }
  .st-key-chat_canvas_card {
    min-height: 520px;
  }
  .chat-user-bubble,
  .chat-assistant-row {
    max-width: 100%;
  }
  [class*="st-key-answer_footer_"] {
    max-width: 100%;
    margin-left: 0;
    padding-left: 0;
    padding-right: 0;
  }
  .chat-pipeline-strip {
    align-items: flex-start;
  }
  .pipeline-time {
    width: 100%;
    margin-left: 0;
  }
  .evidence-header-row {
    align-items: flex-start;
    flex-direction: column;
  }
  .evidence-selected-pill {
    white-space: normal;
  }
  .evidence-selected-preview-body {
    grid-template-columns: 34px minmax(0, 1fr);
  }
  .evidence-score-row {
    grid-template-columns: 70px 38px minmax(0, 1fr);
  }
  .evidence-source-actions {
    grid-template-columns: 1fr;
  }
  .doc-table-header {
    flex-direction: column;
    align-items: stretch;
  }
  .doc-table-actions {
    justify-content: flex-start;
    flex-wrap: wrap;
  }
  .doc-table-scroll {
    overflow-x: auto;
  }
  .doc-table-grid {
    min-width: 1060px;
  }
  .documents-metric-grid {
    grid-template-columns: 1fr;
  }
  .selected-document-card {
    position: static;
    width: 100%;
    max-width: 100%;
    margin-left: 0;
    margin-right: 0;
  }
  .documents-badges {
    gap: 0.34rem;
  }
  .documents-badge {
    padding-inline: 0.32rem;
    font-size: 0.68rem;
  }
}
@media (max-width: 620px) {
  .ingestion-status-grid { grid-template-columns: 1fr; }
  .ingestion-status-card { min-height: 150px; }
  .workflow, .debug-grid { grid-template-columns: 1fr; }
  .doc-table-heading {
    align-items: flex-start;
  }
  .doc-table-summary {
    line-height: 1.55;
  }
  .doc-info-strip {
    align-items: stretch;
    flex-direction: column;
  }
  .doc-view-all {
    width: 100%;
  }
  .documents-badges {
    grid-template-columns: 1fr;
  }
  .documents-upload-file-name {
    max-width: 58vw;
  }
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
  <div class="side-row"><strong>Admin</strong><span class="side-pill">Local</span></div>
  <div class="side-row"><span>portfolio-ready build</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

    return section


def render_header() -> dict[str, bool]:
    with st.container(key="app_header_shell"):
        left, right = st.columns([1.55, 1.15], gap="large", vertical_alignment="top")
        with left:
            st.markdown(
                """
<div class="app-header">
  <div class="app-title-block">
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
                    width=150,
                )
                ingest_clicked = st.button(
                    "Ingest",
                    key="header_ingest",
                    type="primary",
                    width=116,
                )
                clear_clicked = st.button(
                    "Clear chat",
                    key="header_clear",
                    width=132,
                )
    return {"upload": upload_clicked, "ingest": ingest_clicked, "clear": clear_clicked}


def get_status_card_icon_svg(icon_name: str) -> str:
    icons = {
        "document": """
<svg viewBox="0 0 40 40" aria-hidden="true" focusable="false" fill="none" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round">
  <path d="M13 6.5h10.2L30 13.3V33.5H13z" />
  <path d="M23 6.8v7h6.8" />
  <path d="M17 20h8.5" />
  <path d="M17 25.5h9.5" />
</svg>
""",
        "layers": """
<svg viewBox="0 0 40 40" aria-hidden="true" focusable="false" fill="none" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round">
  <path d="M20 7.5 32 14.2 20 20.9 8 14.2z" />
  <path d="m8 20 12 6.7L32 20" />
  <path d="m8 25.8 12 6.7 12-6.7" />
</svg>
""",
        "database": """
<svg viewBox="0 0 40 40" aria-hidden="true" focusable="false" fill="none" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round">
  <ellipse cx="20" cy="10.5" rx="10.5" ry="4.8" />
  <path d="M9.5 10.5v18.8c0 2.7 4.7 4.8 10.5 4.8s10.5-2.1 10.5-4.8V10.5" />
  <path d="M30.5 20c0 2.7-4.7 4.8-10.5 4.8S9.5 22.7 9.5 20" />
</svg>
""",
    }
    return icons.get(icon_name, "")


def render_ingestion_status_cards(stats: dict[str, Any]) -> None:
    cards = [
        {
            "label": "Documents indexed",
            "value": str(stats.get("total_documents", 0)),
            "helper": "Across persistent collection",
            "tone": "warm",
            "icon": get_status_card_icon_svg("document"),
            "is_text": False,
        },
        {
            "label": "Chunks stored",
            "value": f'{stats.get("total_chunks", 0):,}',
            "helper": "Semantic chunks",
            "tone": "cool",
            "icon": get_status_card_icon_svg("layers"),
            "is_text": False,
        },
        {
            "label": "Vector DB",
            "value": "ChromaDB",
            "helper": "Local persistence",
            "tone": "gold",
            "icon": get_status_card_icon_svg("database"),
            "is_text": True,
        },
    ]

    card_html = []
    for card in cards:
        value_class = "ingestion-status-value is-text" if card["is_text"] else "ingestion-status-value"
        card_html.append(
            f"""
  <div class="ingestion-status-card is-{html.escape(card["tone"])}">
    <div class="ingestion-status-icon" aria-label="{html.escape(card["label"])} icon">{card["icon"]}</div>
    <div class="ingestion-status-copy">
      <div class="ingestion-status-label">{html.escape(card["label"])}</div>
      <div class="{value_class}">{html.escape(card["value"])}</div>
      <div class="ingestion-status-helper">{html.escape(card["helper"])}</div>
    </div>
  </div>
"""
        )

    st.markdown(
        f"""
<div class="ingestion-status-grid">
{''.join(card_html)}
</div>
""",
        unsafe_allow_html=True,
    )


def _strip_inline_source_citations(text: str) -> str:
    cleaned = re.sub(r"\[source:[^\]]+\]", " ", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)\]])", r"\1", cleaned)
    cleaned = re.sub(r"([(])\s+", r"\1", cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    lines = [line for line in lines if line and line not in {".", ",", ";", ":"}]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or "See cited sources below."


def _format_answer_html(answer: str) -> str:
    return html.escape(_strip_inline_source_citations(answer))


def _unique_source_filenames(sources: list[dict[str, Any]]) -> list[str]:
    filenames: list[str] = []
    seen: set[str] = set()
    for source in sources:
        filename = str(source.get("source", "") or "").strip()
        if not filename:
            continue
        key = filename.casefold()
        if key in seen:
            continue
        seen.add(key)
        filenames.append(filename)
    if sources and not filenames:
        return ["Unknown source"]
    return filenames


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


def _coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


def _format_score(value: Any) -> str:
    score = _coerce_score(value)
    return "n/a" if score is None else f"{score:.2f}"


def _score_width(value: Any) -> int:
    score = _coerce_score(value)
    return 0 if score is None else int(score * 100)


def _truncate_text(value: Any, limit: int = 290) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "No snippet preview is available for this source."
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _selected_answer_preview_text(message: dict[str, Any], limit: int = 170) -> str:
    content = str(message.get("content", "") or "")
    content = re.sub(r"\[source:[^\]]+\]", " ", content, flags=re.IGNORECASE)
    text = " ".join(content.split())
    if not text:
        return "No answer preview is available."
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _source_document_target(source: dict[str, Any]) -> str:
    filename = str(source.get("source", "") or "").strip()
    document_hash = str(source.get("document_hash", "") or "").strip()
    return document_hash or filename or "Unknown source"


def _source_modal_href(source: dict[str, Any], source_section: str = "Chat / Answer") -> str:
    target = quote(_source_document_target(source), safe="")
    section = quote(source_section, safe="")
    return f"?view_doc={target}&from_section={section}"


def build_evidence_source_card_html(source: dict[str, Any], source_section: str = "Chat / Answer") -> str:
    filename = str(source.get("source", "") or "").strip() or "Unknown source"
    page = str(source.get("page_number", "") or "?")
    chunk_id = str(source.get("chunk_id", "") or "n/a")
    similarity = source.get("similarity")
    rerank = source.get("rerank_score")
    snippet = _truncate_text(source.get("text"))
    href = _source_modal_href(source, source_section)
    similarity_width = _score_width(similarity)
    rerank_width = _score_width(rerank)
    similarity_label = _format_score(similarity)
    rerank_label = _format_score(rerank)
    return f"""
<details class="evidence-source-card">
  <summary class="evidence-source-summary">
    <div class="evidence-pdf-badge">PDF</div>
    <div class="evidence-source-title-wrap">
      <div class="evidence-source-name" title="{html.escape(filename)}">{html.escape(filename)}</div>
      <div class="evidence-source-meta">Page {html.escape(page)} &middot; Chunk {html.escape(chunk_id)}</div>
      <div class="evidence-source-compact-meta">
        <span class="evidence-compact-score">Similarity <strong>{html.escape(similarity_label)}</strong></span>
        <span class="evidence-compact-score">Rerank <strong>{html.escape(rerank_label)}</strong></span>
      </div>
    </div>
    <span class="evidence-source-chevron" aria-hidden="true">&rsaquo;</span>
  </summary>
  <div class="evidence-source-expanded">
    <div class="evidence-score-row">
      <span>Similarity</span>
      <strong>{html.escape(similarity_label)}</strong>
      <div class="evidence-score-track"><span style="width: {similarity_width}%"></span></div>
    </div>
    <div class="evidence-score-row">
      <span>Rerank</span>
      <strong>{html.escape(rerank_label)}</strong>
      <div class="evidence-score-track"><span style="width: {rerank_width}%"></span></div>
    </div>
    <div class="evidence-snippet">{html.escape(snippet)}</div>
    <div class="evidence-source-actions">
      <a class="evidence-action-link" href="{html.escape(href, quote=True)}" target="_self">Open source</a>
      <a class="evidence-action-link is-primary" href="{html.escape(href, quote=True)}" target="_self">View document</a>
    </div>
  </div>
</details>
"""


def render_chat_user_bubble(message: dict[str, Any]) -> None:
    content = str(message.get("content", "") or "")
    timestamp = str(message.get("timestamp", "") or "Now")
    st.markdown(
        f"""
<div class="chat-user-row">
  <div class="chat-user-bubble">
    <span>{html.escape(content)}</span>
    <span class="chat-user-time">{html.escape(timestamp)}</span>
    <span class="chat-user-avatar" aria-hidden="true"></span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_chat_answer_card(message: dict[str, Any], index: int, selected: bool = False) -> None:
    content = str(message.get("content", "") or "")
    timestamp = str(message.get("timestamp", "") or "Now")
    sources = message.get("sources", []) or []
    selected_class = " is-selected" if selected else ""
    app_icon_uri = _load_app_icon_data_uri()
    source_filenames = _unique_source_filenames(sources)
    source_label = "source" if len(source_filenames) == 1 else "sources"
    source_text = "; ".join(source_filenames)
    source_pill = (
        f'<span class="chat-source-pill">{html.escape(source_label)}: {html.escape(source_text)}</span>'
        if source_filenames
        else ""
    )
    st.markdown(
        f"""
<div class="chat-assistant-row{selected_class}" data-chat-message-index="{index}">
  <div class="chat-bot-avatar" aria-hidden="true"><img src="{app_icon_uri}" alt="" loading="lazy" /></div>
  <div class="chat-answer-card">
    <div class="chat-answer-head">
      <div class="assistant-label">RAG Knowledge Assistant</div>
      <div class="chat-assistant-time">{html.escape(timestamp)}</div>
    </div>
    <div class="answer-body">{_format_answer_html(content)}</div>
    {source_pill}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_evidence_sources(sources: list[dict[str, Any]], source_section: str = "Chat / Answer") -> None:
    if not sources:
        st.markdown(
            """
<div class="evidence-empty">No cited sources are attached to this answer.</div>
""",
            unsafe_allow_html=True,
        )
        return
    cards = "".join(build_evidence_source_card_html(source, source_section) for source in sources)
    st.markdown(cards, unsafe_allow_html=True)


def render_evidence_debug(debug: dict[str, Any] | None) -> None:
    if not debug:
        st.markdown(
            """
<div class="evidence-empty">Behind-the-scenes details are not available for this answer.</div>
""",
            unsafe_allow_html=True,
        )
        return

    token_usage = debug.get("token_usage", {}) or {}
    token_total = token_usage.get("total", {}) or {}
    answer_total = debug.get("answer_total_tokens") or token_total.get("total_tokens", 0)
    response_time = debug.get("response_time", 0) or 0
    rows = [
        ("Original query", debug.get("original_query", "")),
        ("Rewritten query", debug.get("rewritten_query", "")),
        ("Model", debug.get("model", CHAT_MODEL)),
        ("Response time", f"{float(response_time):.2f}s"),
        ("Total tokens", f"{int(token_total.get('total_tokens', 0) or 0):,}"),
        ("Answer tokens", f"{int(answer_total or 0):,}"),
    ]
    debug_rows = "".join(
        f'<div class="evidence-debug-row"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'
        for label, value in rows
    )
    steps = token_usage.get("steps", []) or []
    step_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(step.get('task', '')))}</td>"
        f"<td>{int(step.get('prompt_tokens', 0) or 0):,}</td>"
        f"<td>{int(step.get('completion_tokens', 0) or 0):,}</td>"
        f"<td>{int(step.get('total_tokens', 0) or 0):,}</td>"
        "</tr>"
        for step in steps
    )
    token_table = (
        f"""
<div class="evidence-section-title compact">Token usage by step</div>
<table class="evidence-token-table">
  <thead><tr><th>Step</th><th>Input tokens</th><th>Output tokens</th><th>Total tokens</th></tr></thead>
  <tbody>{step_rows}</tbody>
</table>
"""
        if step_rows
        else ""
    )
    st.markdown(
        f"""
<div class="evidence-debug-card">
  {debug_rows}
</div>
{token_table}
""",
        unsafe_allow_html=True,
    )


def _render_chat_pipeline_status_legacy_unused(
    debug: dict[str, Any] | None = None,
    *,
    active_step: int | None = None,
    completed_steps: int = 0,
    is_loading: bool = False,
    response_time: Any = None,
    failed_step: int | None = None,
) -> None:
    if not debug and not is_loading and failed_step is None:
        return
    response_time = response_time if response_time is not None else ((debug or {}).get("response_time", 0) or 0)
    if debug and failed_step is None and debug.get("pipeline_failed_step") is not None:
        try:
            failed_step = int(debug.get("pipeline_failed_step"))
        except (TypeError, ValueError):
            failed_step = 0
        try:
            completed_steps = int(debug.get("pipeline_completed_steps", failed_step) or 0)
        except (TypeError, ValueError):
            completed_steps = failed_step
    elif debug and not is_loading and failed_step is None:
        completed_steps = 4
    steps = ["Query rewrite", "Retrieve top 10", "Rerank top 5", "Generate answer"]
    step_markup = "".join(
        f'<div class="pipeline-step"><span class="pipeline-check">✓</span><span>{html.escape(step)}</span></div>'
        for step in steps
    )
    st.markdown(
        f"""
<div class="chat-pipeline-strip">
  {step_markup}
  <div class="pipeline-time">{float(response_time):.2f}s</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_chat_pipeline_status(
    debug: dict[str, Any] | None = None,
    *,
    active_step: int | None = None,
    completed_steps: int = 0,
    is_loading: bool = False,
    response_time: Any = None,
    failed_step: int | None = None,
) -> None:
    if not debug and not is_loading and failed_step is None:
        return
    response_time = response_time if response_time is not None else ((debug or {}).get("response_time", 0) or 0)
    if debug and failed_step is None and debug.get("pipeline_failed_step") is not None:
        try:
            failed_step = int(debug.get("pipeline_failed_step"))
        except (TypeError, ValueError):
            failed_step = 0
        try:
            completed_steps = int(debug.get("pipeline_completed_steps", failed_step) or 0)
        except (TypeError, ValueError):
            completed_steps = failed_step
    elif debug and not is_loading and failed_step is None:
        completed_steps = 4

    steps = ["Query rewrite", "Retrieve top 10", "Rerank top 5", "Generate answer"]
    step_markup_parts = []
    for index, step in enumerate(steps):
        if failed_step == index:
            state = "failed"
            indicator = "!"
        elif index < completed_steps:
            state = "complete"
            indicator = "&#10003;"
        elif active_step == index:
            state = "active"
            indicator = ""
        else:
            state = "pending"
            indicator = ""
        step_markup_parts.append(
            f'<div class="pipeline-step is-{state}"><span class="pipeline-check is-{state}">{indicator}</span><span>{html.escape(step)}</span></div>'
        )
        if index < len(steps) - 1:
            step_markup_parts.append('<span class="pipeline-arrow" aria-hidden="true"></span>')

    if failed_step is not None:
        time_markup = '<div class="pipeline-time is-failed">Failed</div>'
    elif is_loading:
        time_markup = '<div class="pipeline-time is-loading">Working...</div>'
    else:
        time_markup = (
            '<div class="pipeline-time">'
            '<span class="pipeline-clock" aria-hidden="true"></span>'
            f'<span>{float(response_time or 0):.2f}s</span>'
            '</div>'
        )
    st.markdown(
        f"""
<div class="chat-pipeline-strip">
  {"".join(step_markup_parts)}
  {time_markup}
</div>
""",
        unsafe_allow_html=True,
    )


def render_chat_evidence_panel(message: dict[str, Any] | None, mode: str = "sources") -> None:
    opened = "Behind the scenes" if mode == "debug" else "Sources used"
    if not message:
        empty_icon_svg = _build_evidence_empty_icon_svg()
        st.markdown(
            f"""
<div class="evidence-header-row">
  <div class="evidence-header">Answer Evidence</div>
</div>
<div class="evidence-empty large">
  <div class="evidence-empty-icon">{empty_icon_svg}</div>
  <strong>No answer selected yet</strong>
  <span>Ask a question or select an answer control to inspect citations and RAG details.</span>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    sources = message.get("sources", []) or []
    debug = message.get("debug")
    preview_text = _selected_answer_preview_text(message)
    app_icon_uri = html.escape(_load_app_icon_data_uri(), quote=True)
    info_icon_uri = html.escape(_load_indexed_docs_icon_data_uri("info-icon.png"), quote=True)
    st.markdown(
        f"""
<div class="evidence-header-row">
  <div class="evidence-header">Answer Evidence</div>
  <div class="evidence-selected-pill">
    <span class="evidence-selected-pill-icon" aria-hidden="true"><img src="{info_icon_uri}" alt="" loading="lazy" /></span>
    <span>Showing evidence for selected answer</span>
  </div>
</div>
<div class="evidence-selected-preview">
  <div class="evidence-selected-preview-title">Selected answer preview</div>
  <div class="evidence-selected-preview-body">
    <div class="evidence-selected-preview-icon" aria-hidden="true"><img src="{app_icon_uri}" alt="" loading="lazy" /></div>
    <div class="evidence-selected-preview-text">{html.escape(preview_text)}</div>
  </div>
</div>
<div class="evidence-opened-pill">Opened from: <strong>{html.escape(opened)}</strong></div>
""",
        unsafe_allow_html=True,
    )
    if mode == "debug":
        st.markdown('<div class="evidence-section-title">Behind the scenes</div>', unsafe_allow_html=True)
        render_evidence_debug(debug)
        st.markdown('<div class="evidence-section-title">Sources used</div>', unsafe_allow_html=True)
        render_evidence_sources(sources)
    else:
        st.markdown('<div class="evidence-section-title">Sources used</div>', unsafe_allow_html=True)
        render_evidence_sources(sources)
        st.markdown('<div class="evidence-section-title">Behind the scenes</div>', unsafe_allow_html=True)
        render_evidence_debug(debug)


def render_chat_empty_canvas(stats: dict[str, Any]) -> None:
    total_chunks = int(stats.get("total_chunks", 0) or 0)
    if total_chunks <= 0:
        title = "No indexed documents yet"
        copy = "Upload PDFs in Documents, then ask questions grounded only in indexed files."
    else:
        title = "Ask across your indexed PDFs"
        copy = "Your assistant will retrieve, rerank, and cite evidence from the documents you ingested."
    flow_uri = _load_chat_empty_state_asset_data_uri("empty_state_flow.png")
    st.markdown(
        f"""
<div class="chat-empty-state">
  <div class="chat-empty-graphic" aria-label="Source documents are retrieved, reranked, and used to generate a grounded answer.">
    <img class="chat-empty-flow-image" src="{html.escape(flow_uri, quote=True)}" alt="Source Documents to Retrieve, Rerank, Generate, and Grounded answer flow" loading="lazy" />
  </div>
  <div class="chat-empty-title">{html.escape(title)}</div>
  <div class="chat-empty-copy">{html.escape(copy)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


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


def _indexed_docs_icon(filename: str, alt: str, class_name: str = "") -> str:
    icon_uri = _load_indexed_docs_icon_data_uri(filename)
    if not icon_uri:
        return '<span aria-hidden="true">•</span>'
    class_attr = f' class="{html.escape(class_name)}"' if class_name else ""
    return f'<img{class_attr} src="{icon_uri}" alt="{html.escape(alt)}" loading="lazy" />'


def _trash_icon() -> str:
    return (
        '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
        '<path d="M4 7h16" stroke-width="2" stroke-linecap="round"/>'
        '<path d="M10 11v6M14 11v6" stroke-width="2" stroke-linecap="round"/>'
        '<path d="M6 7l1 14h10l1-14" stroke-width="2" stroke-linejoin="round"/>'
        '<path d="M9 7V4h6v3" stroke-width="2" stroke-linejoin="round"/>'
        "</svg>"
    )


def _format_file_size(size_bytes: Any) -> str:
    try:
        size = int(size_bytes or 0)
    except (TypeError, ValueError):
        size = 0
    if size <= 0:
        return ""
    size_mb = size / (1024 * 1024)
    if size_mb >= 1:
        return f"{size_mb:.1f} MB"
    size_kb = max(1, round(size / 1024))
    return f"{size_kb:,} KB"


def _chunk_segments(chunks: int, max_chunks: int, segment_count: int = 6) -> str:
    if chunks <= 0 or max_chunks <= 0:
        filled = 0
    else:
        filled = max(1, min(segment_count, round((chunks / max_chunks) * segment_count)))
    segments = [
        f'<span class="{"is-filled" if index < filled else ""}"></span>'
        for index in range(segment_count)
    ]
    return "".join(segments)


def _doc_head(label: str, icon_filename: str | None = None, fallback: str = "") -> str:
    if icon_filename:
        icon = _indexed_docs_icon(icon_filename, f"{label} icon")
    else:
        icon = f'<span class="doc-head-hash">{fallback}</span>'
    return f'<span class="doc-head-label">{icon}<span>{html.escape(label)}</span></span>'


def _pdf_modal_id(document: dict[str, Any]) -> str:
    filename = str(document.get("filename", "") or "document")
    document_hash = str(document.get("document_hash", "") or "").strip()
    target = quote(document_hash or filename, safe="")
    return f"pdf-modal-{target}"


def render_document_table_search_script(search_key: str, table_id: str) -> None:
    st.iframe(
        f"""
<script>
(() => {{
  const parentDocument = window.parent.document;
  const input = parentDocument.querySelector('.st-key-{html.escape(search_key, quote=True)} input[placeholder="Search documents"]');
  const tableBody = parentDocument.querySelector('[data-doc-table-id="{html.escape(table_id, quote=True)}"]');
  const card = parentDocument.querySelector('.st-key-{html.escape(table_id.replace("-", "_"), quote=True)}_table_card') || tableBody;
  if (!input || !tableBody || !card || tableBody.dataset.searchBound === "true") return;
  tableBody.dataset.searchBound = "true";

  const rows = Array.from(tableBody.querySelectorAll('.doc-table-row'));
  const emptyRow = tableBody.querySelector('.doc-search-empty-row');
  const countNode = card.querySelector('.doc-summary-count');
  const chunkNode = card.querySelector('.doc-summary-chunks');
  const totalDocuments = rows.length;

  const label = (count, singular, plural) => count === 1 ? singular : plural;
  const applyFilter = () => {{
    const query = (input.value || '').trim().toLowerCase();
    let visibleDocuments = 0;
    let visibleChunks = 0;
    rows.forEach((row) => {{
      const matches = !query || (row.dataset.searchText || '').includes(query);
      row.hidden = !matches;
      row.classList.toggle('is-search-hidden', !matches);
      if (matches) {{
        visibleDocuments += 1;
        visibleChunks += Number(row.dataset.chunks || 0);
      }}
    }});
    if (emptyRow) {{
      emptyRow.hidden = visibleDocuments !== 0;
    }}
    if (countNode) {{
      countNode.textContent = query
        ? `${{visibleDocuments.toLocaleString()}} of ${{totalDocuments.toLocaleString()}} ${{label(totalDocuments, 'document', 'documents')}}`
        : `${{totalDocuments.toLocaleString()}} ${{label(totalDocuments, 'document', 'documents')}}`;
    }}
    if (chunkNode) {{
      chunkNode.textContent = `${{visibleChunks.toLocaleString()}} ${{label(visibleChunks, 'chunk', 'chunks')}}`;
    }}
  }};

  input.addEventListener('input', applyFilter);
  input.addEventListener('search', applyFilter);
  window.parent.requestAnimationFrame(applyFilter);
}})();
</script>
""",
        height=1,
        width=1,
    )


def render_document_table(
    documents: list[dict[str, Any]],
    title: str = "Indexed documents",
    source_section: str | None = None,
    enable_delete: bool = False,
    enable_selection: bool = False,
    selected_document_hash: str | None = None,
    selection_section: str | None = None,
    show_search: bool = False,
    search_key: str | None = None,
    search_placeholder: str = "Search documents",
    total_document_count: int | None = None,
    info_copy: str = "All documents are chunked semantically and stored as high-quality embeddings for accurate retrieval.",
    empty_title: str = "No indexed documents yet",
    empty_copy: str = "Upload PDFs and run ingestion to populate this table.",
) -> None:
    total_documents = len(documents)
    total_chunks = sum(int(doc.get("chunks") or 0) for doc in documents)
    unfiltered_documents = total_document_count if total_document_count is not None else total_documents
    max_chunks = max((int(doc.get("chunks") or 0) for doc in documents), default=0)
    formatted_timestamps = [_format_ingested_timestamp(doc.get("last_ingested")) for doc in documents]
    last_updated = next((timestamp for timestamp in formatted_timestamps if timestamp), "Not ingested yet")
    selected_document_hash = (selected_document_hash or "").strip()
    table_classes = "doc-table-grid has-selection" if enable_selection else "doc-table-grid"
    search_key = search_key or f"{re.sub(r'[^a-zA-Z0-9_]+', '_', title.lower()).strip('_')}_search"
    table_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", search_key).strip("-") or "document-search"
    card_classes = "doc-table-card has-client-search" if show_search else "doc-table-card"
    selection_section = selection_section or source_section

    document_icon = _indexed_docs_icon("document-outline-icon.png", "Documents")
    pdf_icon = _indexed_docs_icon("pdf-file-icon.png", "PDF file")
    view_icon = _indexed_docs_icon("view-eye-icon.png", "View")
    sync_icon = _indexed_docs_icon("sync-icon.png", "Re-ingest")
    trash_icon = _trash_icon()
    lightbulb_icon = _indexed_docs_icon("lightbulb-info-icon.png", "Info")

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
        chunking = str(doc.get("chunking_strategy", "") or "")
        is_selected = bool(enable_selection and document_hash and document_hash == selected_document_hash)
        status_class = _status_class(status)
        file_size = _format_file_size(doc.get("file_size"))
        location_text = " ".join(
            str(doc.get(key, "") or "")
            for key in ("location", "source", "source_path", "source_file", "path")
        )
        search_text = " ".join(
            [filename, status, timestamp, document_hash, short_hash, chunking, location_text, "docs/", "uploaded_docs/"]
        ).lower()
        file_meta_html = (
            f"{html.escape(extension)} &middot; {html.escape(file_size)}"
            if file_size
            else f"{html.escape(extension)} source document"
        )
        selected_pill_html = '<div class="doc-selected-pill">Selected</div>' if is_selected else ""
        chunk_segments = _chunk_segments(chunks, max_chunks)
        view_target = quote(document_hash or filename, safe="")
        source_query = f"&from_section={quote(source_section, safe='')}" if source_section else ""
        selected_query = f"&selected_doc={view_target}" if document_hash else ""
        selection_section_query = f"section={quote(selection_section, safe='')}&" if selection_section else ""
        selection_href = (
            f"?{selection_section_query}"
            f"selected_doc={view_target}"
        )
        reingest_href = f"?reingest_doc={view_target}{source_query}{selected_query}"
        delete_section_query = f"section={quote(selection_section or source_section or 'Documents', safe='')}&"
        delete_href = f"?{delete_section_query}delete_doc={view_target}{selected_query}" if enable_delete else ""
        modal_id = _pdf_modal_id(doc)
        selection_cell_html = (
            '<div class="doc-cell doc-select-cell">'
            f'<a class="doc-select-control{" is-selected" if is_selected else ""}" href="{selection_href}" '
            'target="_self" '
            'data-doc-select-control '
            f'data-selected-doc-hash="{html.escape(document_hash, quote=True)}" '
            f'data-selection-section="{html.escape(selection_section or "", quote=True)}" '
            f'aria-label="Select {html.escape(filename)}" title="Select {html.escape(filename)}">'
            '<span aria-hidden="true"></span></a>'
            '</div>'
            if enable_selection
            else ""
        )
        delete_action_html = (
            f'<a class="tiny-action danger" href="{delete_href}" target="_self" '
            'data-delete-doc-control '
            f'data-delete-doc-hash="{html.escape(document_hash, quote=True)}" '
            f'data-delete-doc-filename="{html.escape(filename, quote=True)}" '
            f'title="Delete {html.escape(filename)}">'
            f'{trash_icon}<span>Delete</span></a>'
            if enable_delete
            else ""
        )

        row_html.append(
            f'<div class="doc-table-row{" is-selected" if is_selected else ""}" '
            f'data-search-text="{html.escape(search_text, quote=True)}" data-chunks="{chunks}" '
            f'data-selected-doc-hash="{html.escape(document_hash, quote=True)}">'
            f'{selection_cell_html}'
            '<div class="doc-cell">'
            '<div class="doc-main">'
            f'<div class="doc-file-icon">{pdf_icon}</div>'
            '<div class="doc-file-text">'
            f'<div class="doc-file-name" title="{html.escape(filename)}">{html.escape(filename)}</div>'
            f'<div class="doc-file-meta">{file_meta_html}</div>'
            f'{selected_pill_html}'
            '</div>'
            '</div>'
            '</div>'
            f'<div class="doc-cell doc-num">{pages:,}</div>'
            '<div class="doc-cell chunk-cell">'
            f'<span class="doc-num">{chunks:,}</span>'
            f'<div class="chunk-segments" aria-label="Chunk density">{chunk_segments}</div>'
            '</div>'
            f'<div class="doc-cell"><span class="status-pill {status_class}">{html.escape(status)}</span></div>'
            f'<div class="doc-cell doc-date">{html.escape(timestamp)}</div>'
            f'<div class="doc-cell"><span class="hash-chip" title="{html.escape(document_hash)}">{html.escape(short_hash)}</span></div>'
            '<div class="doc-cell">'
            '<div class="doc-row-actions">'
            f'<a class="tiny-action" href="#{html.escape(modal_id, quote=True)}" data-pdf-modal-target="{html.escape(modal_id, quote=True)}" '
            f'data-pdf-modal-fallback="?view_doc={view_target}{source_query}" title="View {html.escape(filename)}">'
            f'{view_icon}<span>Preview</span></a>'
            f'<a class="tiny-action alt" href="{reingest_href}" title="Re-ingest {html.escape(filename)}">'
            f'{sync_icon}<span>Re-ingest</span></a>'
            f'{delete_action_html}'
            '</div>'
            '</div>'
            '</div>'
        )

    if row_html:
        rows_markup = "".join(row_html)
    else:
        rows_markup = f"""
<div class="doc-empty-row">
  <div class="doc-empty-title">{html.escape(empty_title)}</div>
  <div class="doc-empty-copy">{html.escape(empty_copy)}</div>
</div>
"""
    if show_search:
        rows_markup += (
            '<div class="doc-empty-row doc-search-empty-row" hidden>'
            '<div class="doc-empty-title">No documents match your search.</div>'
            '<div class="doc-empty-copy">Try another filename, hash, status, location, or chunking strategy.</div>'
            '</div>'
        )

    document_label = "document" if total_documents == 1 else "documents"
    chunk_label = "chunk" if total_chunks == 1 else "chunks"
    document_summary_text = f"{total_documents:,} {document_label}"
    if unfiltered_documents != total_documents:
        total_label = "document" if unfiltered_documents == 1 else "documents"
        document_summary_text = f"{total_documents:,} of {unfiltered_documents:,} {total_label}"
    summary_dot = "&bull;"
    actions_icon = "&vellip;"
    chevron_icon = "&rsaquo;"
    selection_head_html = '<div class="doc-cell"><span class="doc-select-head" aria-hidden="true"></span></div>' if enable_selection else ""
    head_cells = "".join(
        [
            selection_head_html,
            f'<div class="doc-cell">{_doc_head("Document", "document-outline-icon.png")}</div>',
            f'<div class="doc-cell">{_doc_head("Pages", "document-outline-icon.png")}</div>',
            f'<div class="doc-cell">{_doc_head("Chunks", "stacked-layers-icon.png")}</div>',
            f'<div class="doc-cell">{_doc_head("Status", "status-shield-icon.png")}</div>',
            f'<div class="doc-cell">{_doc_head("Last ingested", "calendar-icon.png")}</div>',
            f'<div class="doc-cell">{_doc_head("Hash", None, "#")}</div>',
            f'<div class="doc-cell">{_doc_head("Actions", None, actions_icon)}</div>',
        ]
    )
    table_markup = (
        f'<div class="{card_classes}" data-doc-table-id="{html.escape(table_id, quote=True)}">'
        '<div class="doc-table-header">'
        '<div class="doc-table-heading">'
        f'<div class="doc-title-icon">{document_icon}</div>'
        '<div>'
        f'<div class="doc-table-title">{html.escape(title)}</div>'
        '<div class="doc-table-summary">'
        f'<strong class="doc-summary-count">{html.escape(document_summary_text)}</strong>'
        f'<span class="doc-summary-dot">{summary_dot}</span>'
        f'<strong class="doc-summary-chunks">{total_chunks:,} {chunk_label}</strong>'
        f'<span class="doc-summary-dot">{summary_dot}</span>'
        f'Last updated {html.escape(last_updated)}'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
        '<div class="doc-table-scroll">'
        f'<div class="{table_classes}">'
        f'<div class="doc-table-head">{head_cells}</div>'
        f'{rows_markup}'
        '</div>'
        '</div>'
        '<div class="doc-info-strip">'
        '<div class="doc-info-copy">'
        f'{lightbulb_icon}'
        f'<span>{html.escape(info_copy)}</span>'
        '</div>'
        f'<span class="doc-view-all">View all documents <span aria-hidden="true">{chevron_icon}</span></span>'
        '</div>'
        '</div>'
    )
    body_markup = (
        '<div class="doc-table-scroll">'
        f'<div class="{table_classes}">'
        f'<div class="doc-table-head">{head_cells}</div>'
        f'{rows_markup}'
        '</div>'
        '</div>'
        '<div class="doc-info-strip">'
        '<div class="doc-info-copy">'
        f'{lightbulb_icon}'
        f'<span>{html.escape(info_copy)}</span>'
        '</div>'
        f'<span class="doc-view-all">View all documents <span aria-hidden="true">{chevron_icon}</span></span>'
        '</div>'
    )
    if show_search:
        container_key = f"{table_id.replace('-', '_')}_table_card"
        with st.container(key=container_key):
            header_left, header_right = st.columns([1, 0.34], gap="large")
            with header_left:
                st.markdown(
                    (
                        '<div class="doc-table-header-inline">'
                        '<div class="doc-table-heading">'
                        f'<div class="doc-title-icon">{document_icon}</div>'
                        '<div>'
                        f'<div class="doc-table-title">{html.escape(title)}</div>'
                        '<div class="doc-table-summary">'
                        f'<strong class="doc-summary-count">{html.escape(document_summary_text)}</strong>'
                        f'<span class="doc-summary-dot">{summary_dot}</span>'
                        f'<strong class="doc-summary-chunks">{total_chunks:,} {chunk_label}</strong>'
                        f'<span class="doc-summary-dot">{summary_dot}</span>'
                        f'Last updated {html.escape(last_updated)}'
                        '</div>'
                        '</div>'
                        '</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )
            with header_right:
                st.text_input(
                    "Search documents",
                    key=search_key,
                    placeholder=search_placeholder,
                    label_visibility="collapsed",
                )
            st.markdown(
                f'<div class="doc-table-body has-client-search" data-doc-table-id="{html.escape(table_id, quote=True)}">{body_markup}</div>',
                unsafe_allow_html=True,
            )
            render_document_table_search_script(search_key, table_id)
    else:
        st.markdown(table_markup, unsafe_allow_html=True)


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
