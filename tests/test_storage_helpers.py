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

    def test_valid_session_ids_pass_validation(self) -> None:
        self.assertTrue(rag_utils.is_valid_session_id("a" * 32))
        self.assertTrue(rag_utils.is_valid_session_id("01234567-89ab-cdef-0123-456789abcdef"))
        self.assertTrue(rag_utils.is_valid_session_id("F" * 64))

    def test_invalid_session_ids_fail_validation(self) -> None:
        invalid_values = [
            "",
            "../chroma_db",
            "runtime_sessions/abc",
            "abc.def",
            "abc def",
            "abc123",
            "g" * 32,
            "01234567-89ab-cdef-0123-456789abcdeg",
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                self.assertFalse(rag_utils.is_valid_session_id(value))

    @patch("rag_utils.load_env", lambda: None)
    @patch("rag_utils.get_session_id", lambda: "b" * 32)
    @patch.dict(os.environ, {"RAG_STORAGE_MODE": "session"}, clear=True)
    def test_session_mode_paths_use_validated_runtime_session_id(self) -> None:
        self.assertEqual(
            rag_utils.get_active_uploaded_docs_dir().as_posix(),
            f"runtime_sessions/{'b' * 32}/uploaded_docs",
        )
        self.assertEqual(
            rag_utils.get_active_chroma_db_dir().as_posix(),
            f"runtime_sessions/{'b' * 32}/chroma_db",
        )

    def test_cleanup_ttl_is_seven_days(self) -> None:
        self.assertEqual(rag_utils.SESSION_TTL_SECONDS, 7 * 24 * 60 * 60)


if __name__ == "__main__":
    unittest.main()
