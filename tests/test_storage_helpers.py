from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import rag_utils


class StorageHelperTest(unittest.TestCase):
    def tearDown(self) -> None:
        rag_utils._CLI_SESSION_ID = ""

    @patch("rag_utils.load_env", lambda: None)
    @patch.dict(os.environ, {}, clear=True)
    def test_local_mode_returns_local_paths_by_default(self) -> None:
        self.assertEqual(rag_utils.get_storage_mode(), "local")
        self.assertEqual(rag_utils.get_active_docs_dir().as_posix(), "docs")
        self.assertEqual(rag_utils.get_active_uploaded_docs_dir().as_posix(), "uploaded_docs")
        self.assertEqual(rag_utils.get_active_chroma_db_dir().as_posix(), "chroma_db")
        self.assertEqual(rag_utils.get_active_collection_name(), "rag_docs")

    @patch("rag_utils.load_env", lambda: None)
    @patch.dict(os.environ, {"RAG_STORAGE_MODE": "session"}, clear=True)
    def test_session_mode_returns_runtime_session_paths(self) -> None:
        session_id = rag_utils.get_session_id()

        self.assertTrue(session_id)
        self.assertEqual(rag_utils.get_storage_mode(), "session")
        self.assertEqual(
            rag_utils.get_active_uploaded_docs_dir().as_posix(),
            f"runtime_sessions/{session_id}/uploaded_docs",
        )
        self.assertEqual(
            rag_utils.get_active_chroma_db_dir().as_posix(),
            f"runtime_sessions/{session_id}/chroma_db",
        )

    @patch("rag_utils.load_env", lambda: None)
    @patch.dict(os.environ, {"RAG_STORAGE_MODE": "session"}, clear=True)
    def test_session_mode_does_not_use_shared_chroma_path(self) -> None:
        self.assertNotEqual(rag_utils.get_active_uploaded_docs_dir().as_posix(), "uploaded_docs")
        self.assertNotEqual(rag_utils.get_active_chroma_db_dir().as_posix(), "chroma_db")


if __name__ == "__main__":
    unittest.main()
