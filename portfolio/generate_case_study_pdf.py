from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


SOURCE_MD = Path(__file__).with_name("RAG_Knowledge_Assistant_Case_Study.md")
OUTPUT_PATH = Path(__file__).with_name("Rohan_Kandra_RAG_Knowledge_Assistant_Case_Study.pdf")

LIVE_DEMO_URL = "https://ragknowledgeassistant.streamlit.app/"
GITHUB_URL = "https://github.com/rohankandra1998-pmtech/rag-knowledge-assistant"

PAGE_W, PAGE_H = letter
MARGIN = 44
CONTENT_W = PAGE_W - (2 * MARGIN)

NAVY = colors.HexColor("#102331")
CHARCOAL = colors.HexColor("#24323A")
IVORY = colors.HexColor("#F7F3EA")
CARD = colors.HexColor("#FFFDF8")
TEAL = colors.HexColor("#159CA6")
AQUA = colors.HexColor("#DDF4F2")
SOFT_TEAL = colors.HexColor("#EFFAF8")
CORAL = colors.HexColor("#E86E5D")
GOLD = colors.HexColor("#D7A94A")
MUTED = colors.HexColor("#60737C")
LINE = colors.HexColor("#D9E0DD")
WHITE = colors.white


def clean(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def section(markdown: str, title: str) -> str:
    match = re.search(rf"^## {re.escape(title)}\s*$([\s\S]*?)(?=^## |\Z)", markdown, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def field(markdown: str, label: str, default: str = "") -> str:
    match = re.search(rf"^\*\*{re.escape(label)}:\*\*\s*(.+)$", markdown, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def bullets(text: str) -> list[str]:
    return [line.strip()[2:].strip() for line in text.splitlines() if line.strip().startswith("- ")]


def first_paragraph(text: str) -> str:
    for part in re.split(r"\n\s*\n", text.strip()):
        part = re.sub(r"\*\*[^*]+:\*\*", "", part).strip()
        if part and not part.startswith("- "):
            return clean(part)
    return ""


def label_body_items(text: str) -> list[tuple[str, str]]:
    items = []
    for item in bullets(text):
        label, _, body = item.partition(":")
        items.append((label.strip(), body.strip()))
    return items


def load_content() -> dict[str, object]:
    markdown = SOURCE_MD.read_text(encoding="utf-8")
    title = re.search(r"^#\s+(.+)$", markdown, flags=re.MULTILINE)
    solution = section(markdown, "Product Solution")
    architecture = section(markdown, "Architecture Snapshot")

    return {
        "title": title.group(1).strip() if title else "RAG Knowledge Assistant",
        "label": "Additional Information | Product + AI Portfolio Artifact",
        "name": "Rohan Singh Kandra",
        "subtitle": field(markdown, "Subtitle"),
        "problem": first_paragraph(section(markdown, "Problem")),
        "solution": first_paragraph(solution),
        "workflow_steps": [step.strip() for step in field(solution, "Workflow").split("->") if step.strip()],
        "value": first_paragraph(section(markdown, "Product Value")),
        "highlights": bullets(section(markdown, "Product Highlights"))[:5],
        "decisions": label_body_items(section(markdown, "Product Decisions I Made"))[:6],
        "architecture_steps": [step.strip() for step in field(architecture, "Pipeline").split("->") if step.strip()],
        "architecture_groups": label_body_items(architecture)[:6],
        "trust": bullets(section(markdown, "Trust & Evaluation"))[:5],
        "ux": bullets(section(markdown, "UX / Observability Decisions"))[:5],
        "stack": [item.strip() for item in first_paragraph(section(markdown, "Tech Stack")).split("|") if item.strip()],
    }


def wrap_text(c: canvas.Canvas, text: str, max_width: float, font: str, size: float) -> list[str]:
    words = clean(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if c.stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    font: str = "Helvetica",
    size: float = 9.0,
    leading: float = 12.0,
    color=CHARCOAL,
    max_lines: int | None = None,
) -> float:
    c.setFillColor(color)
    c.setFont(font, size)
    lines = wrap_text(c, text, width, font, size)
    if max_lines is not None:
        lines = lines[:max_lines]
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def draw_card(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str, *, accent=TEAL) -> None:
    c.setFillColor(colors.HexColor("#ECE6DA"))
    c.roundRect(x + 1.1, y - h - 1.1, w, h, 10, stroke=0, fill=1)
    c.setFillColor(CARD)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.75)
    c.roundRect(x, y - h, w, h, 10, stroke=1, fill=1)
    c.setFillColor(accent)
    c.roundRect(x + 14, y - 17, 28, 3.4, 1.7, stroke=0, fill=1)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 10.2)
    c.drawString(x + 14, y - 33, title)


def draw_bullets(c: canvas.Canvas, items: list[str], x: float, y: float, width: float, *, size=8.4, leading=10.6) -> float:
    for item in items:
        c.setFillColor(TEAL)
        c.circle(x, y - 3, 2, stroke=0, fill=1)
        y = draw_text(c, item, x + 11, y - 6, width - 11, size=size, leading=leading)
        y -= 3
    return y


def draw_page_background(c: canvas.Canvas) -> None:
    c.setFillColor(IVORY)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)


def draw_footer(c: canvas.Canvas, page_number: int) -> None:
    c.setStrokeColor(colors.HexColor("#D5DED9"))
    c.line(MARGIN, 35, PAGE_W - MARGIN, 35)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN, 21, "Rohan Singh Kandra")
    c.drawCentredString(PAGE_W / 2, 21, "RAG Knowledge Assistant")
    c.drawRightString(PAGE_W - MARGIN, 21, f"{page_number} / 2")


def draw_link_button(c: canvas.Canvas, x: float, y: float, label: str, url: str, *, fill, fixed_width: float | None = None) -> tuple[float, float]:
    c.setFont("Helvetica-Bold", 8.7)
    w = fixed_width or c.stringWidth(label, "Helvetica-Bold", 8.7) + 30
    h = 23
    c.setFillColor(fill)
    c.setStrokeColor(colors.HexColor("#A7D6D2"))
    c.roundRect(x, y - h, w, h, 11.5, stroke=1, fill=1)
    c.setFillColor(NAVY)
    c.drawCentredString(x + w / 2, y - 15.2, label)
    c.linkURL(url, (x, y - h, x + w, y), relative=0, thickness=0)
    return w, h


def draw_header(c: canvas.Canvas, content: dict[str, object]) -> None:
    c.setFillColor(NAVY)
    c.roundRect(MARGIN, PAGE_H - 140, CONTENT_W, 106, 16, stroke=0, fill=1)
    left_x = MARGIN + 18
    right_x = PAGE_W - MARGIN - 18 - 116
    c.setFillColor(TEAL)
    c.roundRect(left_x, PAGE_H - 55, 44, 4, 2, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#D7E6E6"))
    c.setFont("Helvetica-Bold", 7.8)
    c.drawString(left_x, PAGE_H - 70, str(content["label"]))
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 23.0)
    c.drawString(left_x, PAGE_H - 96, str(content["title"]))
    draw_text(
        c,
        str(content["subtitle"]),
        left_x,
        PAGE_H - 115,
        320,
        size=9.1,
        leading=10.5,
        color=colors.HexColor("#DDEBEC"),
        max_lines=2,
    )
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 8.8)
    c.drawCentredString(right_x + 58, PAGE_H - 68, str(content["name"]))
    draw_link_button(c, right_x, PAGE_H - 86, "View Live Demo", LIVE_DEMO_URL, fill=colors.HexColor("#E3F8F5"), fixed_width=116)
    draw_link_button(c, right_x, PAGE_H - 112, "GitHub Repo", GITHUB_URL, fill=colors.HexColor("#FFF3DD"), fixed_width=116)


def draw_arrow(c: canvas.Canvas, x1: float, y: float, x2: float, *, color=TEAL) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.2)
    c.line(x1, y, x2, y)
    c.line(x2, y, x2 - 4, y + 3)
    c.line(x2, y, x2 - 4, y - 3)


def draw_workflow(c: canvas.Canvas, x: float, y: float, width: float, steps: list[str]) -> None:
    row_gap = 18
    col_gap = 26
    box_w = (width - 2 * col_gap) / 3
    box_h = 29
    for index, step in enumerate(steps):
        row = index // 3
        col = index % 3
        bx = x + col * (box_w + col_gap)
        by = y - row * (box_h + row_gap)
        c.setFillColor(AQUA if index in {0, 3} else SOFT_TEAL)
        c.setStrokeColor(colors.HexColor("#B7DCD9"))
        c.roundRect(bx, by - box_h, box_w, box_h, 8, stroke=1, fill=1)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 7.4)
        for line_i, line in enumerate(wrap_text(c, step, box_w - 10, "Helvetica-Bold", 7.4)[:2]):
            c.drawCentredString(bx + box_w / 2, by - 12 - (line_i * 8), line)
        if col < 2:
            draw_arrow(c, bx + box_w + 5, by - box_h / 2, bx + box_w + col_gap - 7)


def draw_decision_grid(c: canvas.Canvas, items: list[tuple[str, str]], x: float, y: float, width: float) -> None:
    col_gap = 10
    row_gap = 10
    card_w = (width - 2 * col_gap) / 3
    card_h = 53
    for index, (label, body) in enumerate(items):
        row = index // 3
        col = index % 3
        bx = x + col * (card_w + col_gap)
        by = y - row * (card_h + row_gap)
        c.setFillColor(colors.HexColor("#F2FAF8") if index % 2 == 0 else colors.HexColor("#FFF7E9"))
        c.setStrokeColor(colors.HexColor("#CFE0DC"))
        c.roundRect(bx, by - card_h, card_w, card_h, 9, stroke=1, fill=1)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 8.0)
        c.drawString(bx + 9, by - 13, label)
        draw_text(c, body, bx + 9, by - 25, card_w - 18, size=6.7, leading=7.7, max_lines=3)


def page_one(c: canvas.Canvas, content: dict[str, object]) -> None:
    draw_page_background(c)
    draw_header(c, content)

    y = PAGE_H - 150
    left_w = 310
    gap = 18
    right_w = CONTENT_W - left_w - gap
    right_x = MARGIN + left_w + gap

    draw_card(c, MARGIN, y, left_w, 104, "Problem", accent=CORAL)
    draw_text(c, str(content["problem"]), MARGIN + 15, y - 49, left_w - 30, size=8.6, leading=11.5, max_lines=5)

    draw_card(c, right_x, y, right_w, 104, "Why It Matters", accent=GOLD)
    draw_text(c, str(content["value"]), right_x + 15, y - 49, right_w - 30, size=8.5, leading=11.5, max_lines=5)

    y -= 121
    draw_card(c, MARGIN, y, CONTENT_W, 124, "Product Solution Workflow", accent=TEAL)
    draw_workflow(c, MARGIN + 16, y - 47, CONTENT_W - 32, list(content["workflow_steps"]))

    y -= 146
    draw_card(c, MARGIN, y, CONTENT_W, 99, "Product Highlights", accent=TEAL)
    highlight_w = (CONTENT_W - 42) / 2
    draw_bullets(c, list(content["highlights"])[:3], MARGIN + 18, y - 51, highlight_w, size=8.3, leading=10.1)
    draw_bullets(c, list(content["highlights"])[3:], MARGIN + 18 + highlight_w + 32, y - 51, highlight_w, size=8.3, leading=10.1)

    y -= 118
    draw_card(c, MARGIN, y, CONTENT_W, 166, "Product Decisions I Made", accent=GOLD)
    draw_decision_grid(c, list(content["decisions"]), MARGIN + 15, y - 49, CONTENT_W - 30)

    draw_footer(c, 1)


def draw_architecture(c: canvas.Canvas, x: float, y: float, width: float, groups: list[tuple[str, str]]) -> None:
    gap = 8
    box_w = (width - 5 * gap) / 6
    box_h = 58
    for index, (label, body) in enumerate(groups):
        bx = x + index * (box_w + gap)
        dark = index in {2, 4}
        c.setFillColor(NAVY if dark else SOFT_TEAL)
        c.setStrokeColor(colors.HexColor("#B7DCD9"))
        c.roundRect(bx, y - box_h, box_w, box_h, 9, stroke=1, fill=1)
        c.setFillColor(WHITE if dark else NAVY)
        c.setFont("Helvetica-Bold", 7.4)
        c.drawCentredString(bx + box_w / 2, y - 14, label)
        body_color = colors.HexColor("#DDEBEC") if dark else CHARCOAL
        lines = wrap_text(c, body, box_w - 10, "Helvetica", 6.4)[:3]
        c.setFillColor(body_color)
        c.setFont("Helvetica", 6.4)
        for line_i, line in enumerate(lines):
            c.drawCentredString(bx + box_w / 2, y - 28 - (line_i * 7.2), line)
        if index < len(groups) - 1:
            draw_arrow(c, bx + box_w + 2, y - box_h / 2, bx + box_w + gap - 3, color=TEAL)


def draw_stack_chips(c: canvas.Canvas, items: list[str], x: float, y: float, width: float) -> float:
    cx = x
    cy = y
    for item in items:
        c.setFont("Helvetica-Bold", 7.8)
        chip_w = c.stringWidth(item, "Helvetica-Bold", 7.8) + 18
        if cx + chip_w > x + width:
            cx = x
            cy -= 24
        c.setFillColor(colors.HexColor("#F2FAF8"))
        c.setStrokeColor(colors.HexColor("#C7DEDA"))
        c.roundRect(cx, cy - 17, chip_w, 17, 8.5, stroke=1, fill=1)
        c.setFillColor(NAVY)
        c.drawString(cx + 9, cy - 11.5, item)
        cx += chip_w + 7
    return cy - 22


def page_two(c: canvas.Canvas, content: dict[str, object]) -> None:
    draw_page_background(c)
    c.setFillColor(NAVY)
    c.roundRect(MARGIN, PAGE_H - 92, CONTENT_W, 56, 14, stroke=0, fill=1)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 19)
    c.drawString(MARGIN + 18, PAGE_H - 68, "Supporting Detail")
    c.setFillColor(colors.HexColor("#DDEBEC"))
    c.setFont("Helvetica", 9.0)
    c.drawRightString(PAGE_W - MARGIN - 18, PAGE_H - 67, "Architecture, trust, evaluation, observability, stack")

    y = PAGE_H - 116
    draw_card(c, MARGIN, y, CONTENT_W, 133, "Architecture Snapshot", accent=TEAL)
    draw_architecture(c, MARGIN + 16, y - 52, CONTENT_W - 32, list(content["architecture_groups"]))

    y -= 155
    col_gap = 18
    col_w = (CONTENT_W - col_gap) / 2
    draw_card(c, MARGIN, y, col_w, 186, "Trust & Evaluation", accent=CORAL)
    draw_bullets(c, list(content["trust"]), MARGIN + 18, y - 55, col_w - 36, size=8.6, leading=10.9)

    rx = MARGIN + col_w + col_gap
    draw_card(c, rx, y, col_w, 186, "UX & Observability", accent=GOLD)
    draw_bullets(c, list(content["ux"]), rx + 18, y - 55, col_w - 36, size=8.6, leading=10.9)

    y -= 211
    draw_card(c, MARGIN, y, CONTENT_W, 118, "Technical Stack", accent=TEAL)
    draw_stack_chips(c, list(content["stack"]), MARGIN + 16, y - 52, CONTENT_W - 32)

    draw_footer(c, 2)


def build_pdf(output_path: Path = OUTPUT_PATH) -> Path:
    content = load_content()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    c.setTitle("RAG Knowledge Assistant - Product + AI Portfolio Case Study")
    c.setAuthor("Rohan Singh Kandra")
    c.setSubject("Additional Information | Product + AI Portfolio Artifact")
    page_one(c, content)
    c.showPage()
    page_two(c, content)
    c.save()
    return output_path


if __name__ == "__main__":
    print(f"Wrote {build_pdf()}")
