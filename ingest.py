from __future__ import annotations

from rag_utils import DOCS_DIR, get_chroma_collection, ingest_folder


def main() -> None:
    print("RAG Knowledge Assistant ingestion")
    print(f"Scanning {DOCS_DIR}/ for PDFs...")
    collection = get_chroma_collection()

    def log(message: str) -> None:
        print(f"- {message}")

    try:
        results = ingest_folder(DOCS_DIR, collection=collection, status_callback=log)
    except Exception as exc:
        print(f"Ingestion failed: {exc}")
        return

    if not results:
        print("No PDFs found. Add files to docs/ and run this script again.")
        return

    print("\nSummary")
    for result in results:
        status = result.get("status", "unknown")
        filename = result.get("filename", "unknown")
        pages = result.get("pages", 0)
        chunks = result.get("chunks", 0)
        reason = result.get("reason", "")
        print(f"- {filename}: {status} | pages={pages} | chunks={chunks} | {reason}")


if __name__ == "__main__":
    main()
