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
        self.assertNotIn('class="doc-select-control" href="?section=Documents&selected_doc=hash-two" target="_blank"', markup)


if __name__ == "__main__":
    unittest.main()
