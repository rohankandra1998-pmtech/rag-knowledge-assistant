from __future__ import annotations

import unittest
from types import SimpleNamespace

import ui_components


class DocumentTableSelectionMarkupTest(unittest.TestCase):
    def test_selection_control_uses_same_tab_documents_query(self) -> None:
        captured_markup: list[str] = []
        original_st = ui_components.st
        ui_components.st = SimpleNamespace(markdown=lambda markup, **_: captured_markup.append(markup))

        try:
            ui_components.render_document_table(
                [
                    {
                        "filename": "First.pdf",
                        "document_hash": "hash-one",
                        "pages": 3,
                        "chunks": 8,
                        "status": "Indexed",
                    },
                    {
                        "filename": "Second.pdf",
                        "document_hash": "hash-two",
                        "pages": 2,
                        "chunks": 5,
                        "status": "Indexed",
                    },
                ],
                source_section="Documents",
                enable_delete=True,
                enable_selection=True,
                selected_document_hash="hash-one",
                selection_section="Documents",
            )
        finally:
            ui_components.st = original_st

        markup = "".join(captured_markup)
        self.assertIn('class="doc-select-control is-selected" href="?section=Documents&selected_doc=hash-one" target="_self" data-doc-select-control data-selected-doc-hash="hash-one"', markup)
        self.assertIn('class="doc-select-control" href="?section=Documents&selected_doc=hash-two" target="_self" data-doc-select-control data-selected-doc-hash="hash-two"', markup)
        self.assertIn('data-selection-section="Documents"', markup)
        self.assertIn('class="doc-table-row is-selected" data-search-text=', markup)
        self.assertIn('data-selected-doc-hash="hash-two"><div class="doc-cell doc-select-cell">', markup)
        self.assertIn('class="tiny-action danger" href="?section=Documents&delete_doc=hash-two&selected_doc=hash-two" target="_self"', markup)
        self.assertIn('data-delete-doc-control data-delete-doc-hash="hash-two" data-delete-doc-filename="Second.pdf"', markup)
        self.assertNotIn('class="doc-select-control" href="?section=Documents&selected_doc=hash-two" target="_blank"', markup)


class EvidenceSourceCardMarkupTest(unittest.TestCase):
    def test_evidence_source_card_uses_document_hash_modal_link(self) -> None:
        markup = ui_components.build_evidence_source_card_html(
            {
                "source": "I765_Additional_Responses.pdf",
                "page_number": 1,
                "chunk_id": "72849a0b87ff875d-0001",
                "similarity": 0.59,
                "rerank_score": 0.95,
                "text": "Evidence snippet",
                "document_hash": "hash-source",
            }
        )

        self.assertIn("?view_doc=hash-source&amp;from_section=Chat%20%2F%20Answer", markup)
        self.assertIn('type="button" data-open-evidence-chunk=', markup)
        self.assertIn("Selected evidence chunk", markup)
        self.assertIn("Evidence snippet", markup)
        self.assertNotIn("?view_chunk=", markup)
        self.assertIn("I765_Additional_Responses.pdf", markup)
        self.assertIn("Page 1 &middot; Chunk 72849a0b87ff875d-0001", markup)
        self.assertIn("Open Chunk", markup)
        self.assertIn("View document", markup)
        self.assertNotIn("Open source", markup)
        self.assertIn("Similarity", markup)
        self.assertIn("width: 59%", markup)
        self.assertIn("width: 95%", markup)

    def test_evidence_source_card_handles_missing_scores(self) -> None:
        markup = ui_components.build_evidence_source_card_html(
            {
                "source": "",
                "page_number": None,
                "chunk_id": None,
                "similarity": "not-a-score",
                "rerank_score": None,
                "text": "",
            }
        )

        self.assertIn("?view_doc=Unknown%20source&amp;from_section=Chat%20%2F%20Answer", markup)
        self.assertIn("Unknown source", markup)
        self.assertIn("Page ? &middot; Chunk n/a", markup)
        self.assertIn(">n/a</strong>", markup)
        self.assertIn("width: 0%", markup)
        self.assertIn("No snippet preview is available for this source.", markup)


class ChatEmptyCanvasMarkupTest(unittest.TestCase):
    def test_empty_canvas_uses_no_indexed_documents_copy(self) -> None:
        captured_markup: list[str] = []
        original_st = ui_components.st
        original_loader = ui_components._load_chat_empty_state_asset_data_uri
        ui_components.st = SimpleNamespace(markdown=lambda markup, **_: captured_markup.append(markup))
        ui_components._load_chat_empty_state_asset_data_uri = lambda filename: f"data:image/png;base64,{filename}"

        try:
            ui_components.render_chat_empty_canvas({"total_chunks": 0})
        finally:
            ui_components.st = original_st
            ui_components._load_chat_empty_state_asset_data_uri = original_loader

        markup = "".join(captured_markup)
        self.assertIn("No indexed documents yet", markup)
        self.assertIn("Upload PDFs in Documents, then ask questions grounded only in indexed files.", markup)
        self.assertIn('class="chat-empty-graphic"', markup)
        self.assertIn('class="chat-empty-flow-image"', markup)
        self.assertIn("empty_state_flow.png", markup)
        self.assertNotIn('class="chat-empty-assistant-badge"', markup)
        self.assertNotIn('class="chat-empty-step"', markup)
        self.assertNotIn("Example questions", markup)
        self.assertNotIn("chat_example", markup)

    def test_empty_canvas_uses_indexed_documents_copy(self) -> None:
        captured_markup: list[str] = []
        original_st = ui_components.st
        original_loader = ui_components._load_chat_empty_state_asset_data_uri
        ui_components.st = SimpleNamespace(markdown=lambda markup, **_: captured_markup.append(markup))
        ui_components._load_chat_empty_state_asset_data_uri = lambda filename: f"data:image/png;base64,{filename}"

        try:
            ui_components.render_chat_empty_canvas({"total_chunks": 12})
        finally:
            ui_components.st = original_st
            ui_components._load_chat_empty_state_asset_data_uri = original_loader

        markup = "".join(captured_markup)
        self.assertIn("Ask across your indexed PDFs", markup)
        self.assertIn("Your assistant will retrieve, rerank, and cite evidence from the documents you ingested.", markup)
        self.assertIn("empty_state_flow.png", markup)
        self.assertNotIn("empty_state_source_documents.png", markup)
        self.assertNotIn("empty_state_retrieve_step.png", markup)
        self.assertNotIn("empty_state_rerank_step.png", markup)
        self.assertNotIn("empty_state_generate_step.png", markup)
        self.assertNotIn("empty_state_grounded_answer.png", markup)
        self.assertNotIn("Example questions", markup)
        self.assertNotIn("chat_example", markup)


class ChatEvidencePanelEmptyMarkupTest(unittest.TestCase):
    def test_empty_evidence_panel_renders_clean_svg_icon(self) -> None:
        captured_markup: list[str] = []
        original_st = ui_components.st
        ui_components.st = SimpleNamespace(markdown=lambda markup, **_: captured_markup.append(markup))

        try:
            ui_components.render_chat_evidence_panel(None)
        finally:
            ui_components.st = original_st

        markup = "".join(captured_markup)
        self.assertIn("Answer Evidence", markup)
        self.assertIn('class="evidence-empty-icon"', markup)
        self.assertIn("<svg", markup)
        self.assertIn('aria-label="Evidence document search"', markup)
        self.assertIn("#105EDD", markup)
        self.assertIn("#F8B400", markup)
        self.assertNotIn("answer-evidence-empty-icon.png", markup)
        self.assertIn("No answer selected yet", markup)
        self.assertIn("Ask a question or select an answer control to inspect citations and RAG details.", markup)


if __name__ == "__main__":
    unittest.main()
