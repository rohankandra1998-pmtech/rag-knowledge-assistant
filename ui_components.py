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
    st.markdown(
        f"""
<style>
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
  border: 1px solid rgba(16, 94, 221, 0.12);
  border-radius: 18px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.99), rgba(248,251,255,0.97)),
    #FFFFFF;
  box-shadow: 0 18px 46px rgba(11, 48, 117, 0.10);
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
.doc-icon-btn {
  border: 1px solid #CFE1FB;
  border-radius: 11px;
  background: #FFFFFF;
  color: var(--blue);
  min-width: 42px;
  height: 42px;
  padding: 0 0.9rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.45rem;
  font-size: 0.9rem;
  font-weight: 800;
  box-shadow: 0 8px 20px rgba(16,94,221,0.08);
}
.doc-icon-btn img {
  width: 19px;
  height: 19px;
  object-fit: contain;
  display: block;
}
.doc-icon-btn.icon-only {
  width: 42px;
  padding: 0;
  border-radius: 999px;
  color: #6B7896;
}
.doc-table-scroll {
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0 0.95rem;
  scrollbar-gutter: stable;
}
.doc-table-grid {
  width: max(100%, 1060px);
  min-width: 1060px;
  border: 1px solid #E6EEF9;
  border-radius: 12px;
  overflow: hidden;
  background: #FFFFFF;
}
.doc-table-head,
.doc-table-row {
  display: grid;
  grid-template-columns: minmax(290px, 1fr) 76px 126px 112px 150px 112px 136px;
  align-items: center;
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
.doc-table-row {
  min-height: 80px;
  border-bottom: 1px solid #E8EFF8;
  background: #FFFFFF;
}
.doc-table-row:nth-child(even) {
  background: #FCFDFF;
}
.doc-table-row:hover {
  background: #F2F8FF;
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
.tiny-action img {
  width: 17px;
  height: 17px;
  object-fit: contain;
  display: block;
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
  .doc-table-header {
    flex-direction: column;
    align-items: stretch;
  }
  .doc-table-actions {
    justify-content: flex-start;
  }
  .doc-table-scroll {
    overflow-x: auto;
  }
  .doc-table-grid {
    min-width: 980px;
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


def render_ingestion_status_cards(stats: dict[str, Any]) -> None:
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
    cards = [
        {
            "label": "Documents indexed",
            "value": str(stats.get("total_documents", 0)),
            "helper": "Across persistent collection",
            "tone": "warm",
            "icon": icons["document"],
            "is_text": False,
        },
        {
            "label": "Chunks stored",
            "value": f'{stats.get("total_chunks", 0):,}',
            "helper": "Semantic chunks",
            "tone": "cool",
            "icon": icons["layers"],
            "is_text": False,
        },
        {
            "label": "Vector DB",
            "value": "ChromaDB",
            "helper": "Local persistence",
            "tone": "gold",
            "icon": icons["database"],
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


def _indexed_docs_icon(filename: str, alt: str, class_name: str = "") -> str:
    icon_uri = _load_indexed_docs_icon_data_uri(filename)
    if not icon_uri:
        return '<span aria-hidden="true">•</span>'
    class_attr = f' class="{html.escape(class_name)}"' if class_name else ""
    return f'<img{class_attr} src="{icon_uri}" alt="{html.escape(alt)}" loading="lazy" />'


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


def render_document_table(
    documents: list[dict[str, Any]],
    title: str = "Indexed documents",
    source_section: str | None = None,
) -> None:
    total_documents = len(documents)
    total_chunks = sum(int(doc.get("chunks") or 0) for doc in documents)
    max_chunks = max((int(doc.get("chunks") or 0) for doc in documents), default=0)
    formatted_timestamps = [_format_ingested_timestamp(doc.get("last_ingested")) for doc in documents]
    last_updated = next((timestamp for timestamp in formatted_timestamps if timestamp), "Not ingested yet")

    document_icon = _indexed_docs_icon("document-outline-icon.png", "Documents")
    refresh_icon = _indexed_docs_icon("refresh-icon.png", "Refresh")
    info_icon = _indexed_docs_icon("info-icon.png", "Info")
    pdf_icon = _indexed_docs_icon("pdf-file-icon.png", "PDF file")
    view_icon = _indexed_docs_icon("view-eye-icon.png", "View")
    sync_icon = _indexed_docs_icon("sync-icon.png", "Re-ingest")
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
        status_class = _status_class(status)
        file_size = _format_file_size(doc.get("file_size"))
        file_meta_html = (
            f"{html.escape(extension)} &middot; {html.escape(file_size)}"
            if file_size
            else f"{html.escape(extension)} source document"
        )
        chunk_segments = _chunk_segments(chunks, max_chunks)
        view_target = quote(document_hash or filename, safe="")
        source_query = f"&from_section={quote(source_section, safe='')}" if source_section else ""
        modal_id = _pdf_modal_id(doc)

        row_html.append(
            '<div class="doc-table-row">'
            '<div class="doc-cell">'
            '<div class="doc-main">'
            f'<div class="doc-file-icon">{pdf_icon}</div>'
            '<div class="doc-file-text">'
            f'<div class="doc-file-name" title="{html.escape(filename)}">{html.escape(filename)}</div>'
            f'<div class="doc-file-meta">{file_meta_html}</div>'
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
            f'{view_icon}<span>View</span></a>'
            f'<span class="tiny-action alt" title="Re-ingest">{sync_icon}<span>Re-ingest</span></span>'
            '</div>'
            '</div>'
            '</div>'
        )

    if row_html:
        rows_markup = "".join(row_html)
    else:
        rows_markup = """
      <div class="doc-empty-row">
        <div class="doc-empty-title">No indexed documents yet</div>
        <div class="doc-empty-copy">Upload PDFs and run ingestion to populate this table.</div>
      </div>
"""

    document_label = "document" if total_documents == 1 else "documents"
    chunk_label = "chunk" if total_chunks == 1 else "chunks"
    summary_dot = "&bull;"
    actions_icon = "&vellip;"
    chevron_icon = "&rsaquo;"
    table_markup = f"""
<div class="doc-table-card">
  <div class="doc-table-header">
    <div class="doc-table-heading">
      <div class="doc-title-icon">{document_icon}</div>
      <div>
        <div class="doc-table-title">{html.escape(title)}</div>
        <div class="doc-table-summary">
          <strong>{total_documents:,} {document_label}</strong>
          <span class="doc-summary-dot">{summary_dot}</span>
          <strong>{total_chunks:,} {chunk_label}</strong>
          <span class="doc-summary-dot">{summary_dot}</span>
          Last updated {html.escape(last_updated)}
        </div>
      </div>
    </div>
    <div class="doc-table-actions" aria-label="Document table actions">
      <span class="doc-icon-btn" title="Refresh">{refresh_icon}<span>Refresh</span></span>
      <span class="doc-icon-btn icon-only" title="Info">{info_icon}</span>
    </div>
  </div>
  <div class="doc-table-scroll">
    <div class="doc-table-grid">
      <div class="doc-table-head">
        <div class="doc-cell">{_doc_head("Document", "document-outline-icon.png")}</div>
        <div class="doc-cell">{_doc_head("Pages", "document-outline-icon.png")}</div>
        <div class="doc-cell">{_doc_head("Chunks", "stacked-layers-icon.png")}</div>
        <div class="doc-cell">{_doc_head("Status", "status-shield-icon.png")}</div>
        <div class="doc-cell">{_doc_head("Last ingested", "calendar-icon.png")}</div>
        <div class="doc-cell">{_doc_head("Hash", None, "#")}</div>
        <div class="doc-cell">{_doc_head("Actions", None, actions_icon)}</div>
      </div>
{rows_markup}
    </div>
  </div>
  <div class="doc-info-strip">
    <div class="doc-info-copy">
      {lightbulb_icon}
      <span>All documents are chunked semantically and stored as high-quality embeddings for accurate retrieval.</span>
    </div>
    <span class="doc-view-all">View all documents <span aria-hidden="true">{chevron_icon}</span></span>
  </div>
</div>
"""
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
