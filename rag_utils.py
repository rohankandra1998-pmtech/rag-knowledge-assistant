from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import chromadb

try:
    from dotenv import load_dotenv as _python_dotenv_load
except Exception:
    _python_dotenv_load = None


CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"
DOCS_DIR = "docs"
UPLOADED_DOCS_DIR = "uploaded_docs"
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"
CROSS_PAGE_CONTEXT_MODE = "full_adjacent_pages"
MAX_ADJACENT_PAGE_CONTEXT_CHARS = 4000


def load_env() -> None:
    """Load local environment variables from .env without overwriting the shell."""
    if _python_dotenv_load is not None:
        _python_dotenv_load(override=False)
        return

    env_path = Path(".env")
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def ensure_project_dirs() -> None:
    for folder in (DOCS_DIR, UPLOADED_DOCS_DIR, CHROMA_DB_DIR):
        Path(folder).mkdir(parents=True, exist_ok=True)


def get_openai_client():
    load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env file from .env.example and add your key."
        )
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "The openai package is not installed correctly. Run `pip install -r requirements.txt` "
            "inside your virtual environment."
        ) from exc
    return OpenAI(api_key=api_key)


def get_chroma_collection():
    ensure_project_dirs()
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _usage_to_dict(usage: Any | None) -> dict[str, Any]:
    """Normalize OpenAI usage objects into plain dictionaries for the UI."""
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    data = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    if data["total_tokens"] == 0:
        data["total_tokens"] = data["prompt_tokens"] + data["completion_tokens"]
    return data


def _empty_usage() -> dict[str, Any]:
    return _usage_to_dict(None)


def _sum_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt_tokens": sum(int(usage.get("prompt_tokens", 0) or 0) for usage in usages),
        "completion_tokens": sum(int(usage.get("completion_tokens", 0) or 0) for usage in usages),
        "total_tokens": sum(int(usage.get("total_tokens", 0) or 0) for usage in usages),
    }


def get_document_hash(file_path: str | Path) -> str:
    """Compute a stable SHA-256 hash so unchanged documents are not re-ingested."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_pdf_pages(file_path: str | Path) -> list[dict[str, Any]]:
    """Extract text page by page, preserving source and page metadata."""
    path = Path(file_path)
    pages: list[dict[str, Any]] = []

    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError(
            "The pypdf package is not installed correctly. Run `pip install -r requirements.txt` "
            "inside your virtual environment."
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RuntimeError(f"Could not open {path.name}: {exc}") from exc

    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        cleaned = re.sub(r"\s+\n", "\n", text).strip()
        if not cleaned:
            continue

        pages.append(
            {
                "source": path.name,
                "page_number": index,
                "text": cleaned,
            }
        )

    return pages


def _fallback_character_chunks(text: str) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _format_page_range(start_page: int, end_page: int) -> str:
    if start_page == end_page:
        return str(start_page)
    return f"{start_page}-{end_page}"


def _trim_previous_page_context(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[-max_chars:].strip()


def _trim_next_page_context(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].strip()


def _page_context_window(pages: list[dict[str, Any]], index: int) -> dict[str, Any]:
    page = pages[index]
    primary_page = int(page["page_number"])
    sections: list[tuple[int, str, str]] = []

    if index > 0:
        previous_page = pages[index - 1]
        previous_page_number = int(previous_page["page_number"])
        previous_text = _trim_previous_page_context(
            str(previous_page.get("text", "")),
            MAX_ADJACENT_PAGE_CONTEXT_CHARS,
        )
        if previous_text:
            sections.append(
                (
                    previous_page_number,
                    f"[Previous page context: page {previous_page_number}]",
                    previous_text,
                )
            )

    current_text = str(page.get("text", "")).strip()
    sections.append((primary_page, f"[Current page: page {primary_page}]", current_text))

    if index + 1 < len(pages):
        next_page = pages[index + 1]
        next_page_number = int(next_page["page_number"])
        next_text = _trim_next_page_context(
            str(next_page.get("text", "")),
            MAX_ADJACENT_PAGE_CONTEXT_CHARS,
        )
        if next_text:
            sections.append((next_page_number, f"[Next page context: page {next_page_number}]", next_text))

    represented_pages = [page_number for page_number, _, text in sections if text.strip()]
    start_page = min(represented_pages) if represented_pages else primary_page
    end_page = max(represented_pages) if represented_pages else primary_page
    combined_text = "\n\n".join(
        f"{marker}\n{text.strip()}"
        for _, marker, text in sections
        if text.strip()
    )

    return {
        "text": combined_text,
        "primary_page": primary_page,
        "page_number": primary_page,
        "start_page": start_page,
        "end_page": end_page,
        "page_range": _format_page_range(start_page, end_page),
        "included_previous_page_context": index > 0 and any(
            marker.startswith("[Previous page context") and text.strip()
            for _, marker, text in sections
        ),
        "included_next_page_context": index + 1 < len(pages) and any(
            marker.startswith("[Next page context") and text.strip()
            for _, marker, text in sections
        ),
        "current_page_text": current_text,
    }


def _normalized_chunk_text(text: str) -> str:
    text = re.sub(r"\[(?:Previous page context|Current page|Next page context): page \d+\]", " ", text)
    return " ".join(text.split()).casefold()


def _chunk_has_current_page_context(chunk_text: str, current_page_text: str, primary_page: int) -> bool:
    cleaned = chunk_text.strip()
    if not cleaned:
        return False

    current_marker = f"[Current page: page {primary_page}]".casefold()
    if current_marker in cleaned.casefold():
        return True

    chunk_core = _normalized_chunk_text(cleaned)
    current_core = " ".join(current_page_text.split()).casefold()
    if not chunk_core or not current_core:
        return False

    if len(chunk_core) >= 25 and chunk_core in current_core:
        return True

    current_fragments = [
        " ".join(fragment.split()).casefold()
        for fragment in re.split(r"(?:\n+|(?<=[.!?])\s+)", current_page_text)
    ]
    for fragment in current_fragments:
        if len(fragment) >= 25 and fragment in chunk_core:
            return True

    current_words = {word for word in re.findall(r"\w+", current_core) if len(word) > 2}
    chunk_words = {word for word in re.findall(r"\w+", chunk_core) if len(word) > 2}
    if current_words and chunk_words:
        overlap = current_words & chunk_words
        required_overlap = min(4, max(1, len(chunk_words) // 2))
        if len(overlap) >= required_overlap and len(overlap) / len(chunk_words) >= 0.45:
            return True

    return False


def _chunk_from_window(window: dict[str, Any], text: str, chunking_strategy: str) -> dict[str, Any]:
    return {
        "source": window.get("source", ""),
        "page_number": int(window["primary_page"]),
        "primary_page": int(window["primary_page"]),
        "start_page": int(window["start_page"]),
        "end_page": int(window["end_page"]),
        "page_range": str(window["page_range"]),
        "text": text.strip(),
        "chunking_strategy": chunking_strategy,
        "cross_page_context_mode": CROSS_PAGE_CONTEXT_MODE,
    }


def semantic_chunk_text(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Split page text into meaning-aware chunks.

    Semantic chunking groups nearby sentences by embedding similarity, which usually
    keeps policies, definitions, and procedures together better than blind character
    windows. If the semantic splitter or embedding call fails, the app falls back to a
    recursive character splitter so ingestion still works.
    """
    chunks: list[dict[str, Any]] = []

    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=75,
        )

        for page_index, page in enumerate(pages):
            window = _page_context_window(pages, page_index)
            window["source"] = page.get("source", "")
            page_chunks = splitter.split_text(window["text"])
            for text in page_chunks:
                cleaned = text.strip()
                if cleaned and _chunk_has_current_page_context(
                    cleaned,
                    str(window.get("current_page_text", "")),
                    int(window["primary_page"]),
                ):
                    chunks.append(_chunk_from_window(window, cleaned, "semantic_full_adjacent_page_overlap"))

    except Exception:
        chunks = []
        for page_index, page in enumerate(pages):
            window = _page_context_window(pages, page_index)
            window["source"] = page.get("source", "")
            for text in _fallback_character_chunks(window["text"]):
                cleaned = text.strip()
                if cleaned and _chunk_has_current_page_context(
                    cleaned,
                    str(window.get("current_page_text", "")),
                    int(window["primary_page"]),
                ):
                    chunks.append(_chunk_from_window(window, cleaned, "recursive_full_adjacent_page_overlap"))

    return chunks


def get_embedding(text: str, client: Any | None = None) -> list[float]:
    return get_embedding_with_usage(text, client=client)["embedding"]


def get_embedding_with_usage(text: str, client: Any | None = None) -> dict[str, Any]:
    client = client or get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return {
        "embedding": response.data[0].embedding,
        "usage": _usage_to_dict(getattr(response, "usage", None)),
    }


def get_embeddings(texts: list[str], client: Any | None = None) -> list[list[float]]:
    return get_embeddings_with_usage(texts, client=client)["embeddings"]


def get_embeddings_with_usage(texts: list[str], client: Any | None = None) -> dict[str, Any]:
    if not texts:
        return {"embeddings": [], "usage": _empty_usage()}

    client = client or get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    return {
        "embeddings": [item.embedding for item in ordered],
        "usage": _usage_to_dict(getattr(response, "usage", None)),
    }


def is_document_already_ingested(collection, document_hash: str) -> bool:
    try:
        result = collection.get(where={"document_hash": document_hash}, limit=1)
        return bool(result.get("ids"))
    except Exception:
        return False


def delete_document_by_hash(collection, document_hash: str) -> int:
    if not document_hash:
        return 0

    deleted_count = 0
    try:
        result = collection.get(where={"document_hash": document_hash})
        ids = result.get("ids", []) if isinstance(result, dict) else []
        if ids:
            collection.delete(ids=ids)
            return len(ids)
    except Exception:
        ids = []

    try:
        collection.delete(where={"document_hash": document_hash})
    except Exception:
        return deleted_count

    return deleted_count


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ingest_pdf(
    file_path: str | Path,
    collection=None,
    force: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Extract, chunk, embed, and store one PDF in persistent ChromaDB."""
    ensure_project_dirs()
    path = Path(file_path)
    collection = collection or get_chroma_collection()
    document_hash = get_document_hash(path)
    file_size = path.stat().st_size

    if is_document_already_ingested(collection, document_hash):
        if not force:
            return {
                "filename": path.name,
                "document_hash": document_hash,
                "status": "skipped",
                "reason": "Already indexed",
                "pages": 0,
                "chunks": 0,
            }
        delete_document_by_hash(collection, document_hash)

    if status_callback:
        status_callback(f"Extracting pages from {path.name}...")
    pages = extract_pdf_pages(path)
    if not pages:
        return {
            "filename": path.name,
            "document_hash": document_hash,
            "status": "failed",
            "reason": "No extractable text found",
            "pages": 0,
            "chunks": 0,
        }

    if status_callback:
        status_callback(f"Semantic chunking {path.name}...")
    chunks = semantic_chunk_text(pages)
    if not chunks:
        return {
            "filename": path.name,
            "document_hash": document_hash,
            "status": "failed",
            "reason": "No chunks were created",
            "pages": len(pages),
            "chunks": 0,
        }

    if status_callback:
        status_callback(f"Embedding {len(chunks)} chunks from {path.name}...")
    client = get_openai_client()
    texts = [chunk["text"] for chunk in chunks]
    embedding_result = get_embeddings_with_usage(texts, client=client)
    embeddings = embedding_result["embeddings"]
    embedding_usage = embedding_result["usage"]
    ingestion_timestamp = _now_iso()

    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = f"{document_hash[:16]}-{index:04d}"
        ids.append(chunk_id)
        metadatas.append(
            {
                "source": path.name,
                "page_number": int(chunk["page_number"]),
                "primary_page": int(chunk.get("primary_page", chunk["page_number"])),
                "start_page": int(chunk.get("start_page", chunk["page_number"])),
                "end_page": int(chunk.get("end_page", chunk["page_number"])),
                "page_range": str(chunk.get("page_range", chunk["page_number"])),
                "chunk_id": chunk_id,
                "chunk_index": int(index),
                "document_hash": document_hash,
                "ingestion_timestamp": ingestion_timestamp,
                "chunking_strategy": chunk.get("chunking_strategy", "semantic"),
                "embedding_model": EMBEDDING_MODEL,
                "cross_page_context_mode": chunk.get("cross_page_context_mode", CROSS_PAGE_CONTEXT_MODE),
                "file_size": int(file_size),
                "status": "Indexed",
            }
        )

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    return {
        "filename": path.name,
        "document_hash": document_hash,
        "status": "indexed",
        "reason": "Indexed successfully",
        "pages": len(pages),
        "chunks": len(chunks),
        "ingestion_timestamp": ingestion_timestamp,
        "embedding_tokens": embedding_usage.get("total_tokens", 0),
        "embedding_prompt_tokens": embedding_usage.get("prompt_tokens", 0),
    }


def ingest_folder(
    folder_path: str | Path = DOCS_DIR,
    collection=None,
    force: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    collection = collection or get_chroma_collection()
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    results = []

    for pdf_path in sorted(folder.glob("*.pdf")):
        results.append(
            ingest_pdf(
                pdf_path,
                collection=collection,
                force=force,
                status_callback=status_callback,
            )
        )

    return results


def format_chat_history(chat_history: list[dict[str, Any]], max_turns: int = 6) -> str:
    relevant = chat_history[-max_turns * 2 :]
    lines = []
    for message in relevant:
        role = message.get("role", "user")
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def rewrite_query_result(current_query: str, chat_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Rewrite follow-up questions into standalone questions for better retrieval."""
    history_text = format_chat_history(chat_history)
    if not history_text:
        return {
            "query": current_query.strip(),
            "usage": _empty_usage(),
            "skipped": True,
        }

    client = get_openai_client()
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the user's latest question as a standalone search query for a "
                "document retrieval system. Use the conversation only to resolve pronouns "
                "or missing context. If the question is already standalone, return it unchanged. "
                "Return only the rewritten query."
            ),
        },
        {
            "role": "user",
            "content": f"Conversation:\n{history_text}\n\nLatest question:\n{current_query}",
        },
    ]
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
        )
        rewritten = response.choices[0].message.content or current_query
        return {
            "query": rewritten.strip().strip('"'),
            "usage": _usage_to_dict(getattr(response, "usage", None)),
            "skipped": False,
        }
    except Exception as exc:
        return {
            "query": current_query.strip(),
            "usage": _empty_usage(),
            "skipped": True,
            "error": str(exc),
        }


def rewrite_query(current_query: str, chat_history: list[dict[str, Any]]) -> str:
    return rewrite_query_result(current_query, chat_history)["query"]


def retrieve_context(
    query: str,
    collection=None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    return retrieve_context_with_usage(query, collection=collection, top_k=top_k)["chunks"]


def retrieve_context_with_usage(
    query: str,
    collection=None,
    top_k: int = 10,
) -> dict[str, Any]:
    collection = collection or get_chroma_collection()
    if collection.count() == 0:
        return {"chunks": [], "usage": _empty_usage()}

    embedding_result = get_embedding_with_usage(query)
    query_embedding = embedding_result["embedding"]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    chunks: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] or {}
        distance = distances[index] if index < len(distances) else None
        similarity = None if distance is None else max(0.0, 1.0 - float(distance))
        chunk_id = metadata.get("chunk_id") or (ids[index] if index < len(ids) else str(index))
        chunks.append(
            {
                "rank": index + 1,
                "text": document,
                "source": metadata.get("source", "Unknown source"),
                "page_number": metadata.get("page_number", "?"),
                "primary_page": metadata.get("primary_page", metadata.get("page_number", "?")),
                "start_page": metadata.get("start_page", metadata.get("page_number", "?")),
                "end_page": metadata.get("end_page", metadata.get("page_number", "?")),
                "page_range": metadata.get("page_range", str(metadata.get("page_number", "?"))),
                "embedding_model": metadata.get("embedding_model", ""),
                "cross_page_context_mode": metadata.get("cross_page_context_mode", ""),
                "chunk_id": chunk_id,
                "document_hash": metadata.get("document_hash", ""),
                "similarity": similarity,
                "distance": distance,
                "metadata": metadata,
            }
        )

    return {"chunks": chunks, "usage": embedding_result["usage"]}


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    return rerank_chunks_with_usage(query, chunks, top_n=top_n)["chunks"]


def rerank_chunks_with_usage(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int = 5,
) -> dict[str, Any]:
    """Use gpt-4.1-mini to score retrieved chunks, then keep the strongest context."""
    if not chunks:
        return {"chunks": [], "usage": _empty_usage()}

    payload = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "page": chunk.get("primary_page", chunk["page_number"]),
            "page_range": chunk.get("page_range", str(chunk["page_number"])),
            "text": chunk["text"][:1400],
        }
        for chunk in chunks
    ]

    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document retrieval reranker. Score each chunk from 0 to 1 "
                        "for relevance to the query. Return compact JSON only in this shape: "
                        '{"scores":[{"chunk_id":"...","score":0.92,"reason":"..."}]}.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nChunks:\n{json.dumps(payload, ensure_ascii=True)}",
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        usage = _usage_to_dict(getattr(response, "usage", None))
        data = _extract_json_object(content)
        scores = {
            item.get("chunk_id"): {
                "score": float(item.get("score", 0)),
                "reason": item.get("reason", ""),
            }
            for item in data.get("scores", [])
        }
    except Exception:
        scores = {}
        usage = _empty_usage()

    reranked: list[dict[str, Any]] = []
    for chunk in chunks:
        score_data = scores.get(chunk["chunk_id"], {})
        fallback_score = chunk.get("similarity")
        if fallback_score is None:
            fallback_score = 0.0
        reranked.append(
            {
                **chunk,
                "rerank_score": float(score_data.get("score", fallback_score)),
                "rerank_reason": score_data.get("reason", "Fallback score from vector similarity"),
            }
        )

    reranked.sort(key=lambda item: item.get("rerank_score", 0), reverse=True)
    return {"chunks": reranked[:top_n], "usage": usage}


def build_rag_prompt(query: str, context_chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    context_blocks = []
    for index, chunk in enumerate(context_chunks, start=1):
        page_range = str(chunk.get("page_range") or chunk.get("page_number", "?"))
        page_label = "pages" if "-" in page_range else "page"
        context_blocks.append(
            "\n".join(
                [
                    f"[Context {index}]",
                    f"source: {chunk['source']}",
                    f"{page_label}: {page_range}",
                    f"chunk: {chunk['chunk_id']}",
                    "text:",
                    chunk["text"],
                ]
            )
        )

    system_prompt = """
You are RAG Knowledge Assistant, a careful document Q&A assistant.

Rules:
- Answer only from the retrieved document context.
- Do not use outside knowledge.
- If the answer is not present in the context, say: "I don't know based on the uploaded documents."
- Cite factual claims inline using either [source: filename.pdf, page 4, chunk abc-0001] or [source: filename.pdf, pages 4-5, chunk abc-0001].
- Cite only sources you actually use.
- If sources are incomplete or disagree, say so clearly.
- Be concise, helpful, and specific.
""".strip()

    user_prompt = f"""
Standalone question:
{query}

Retrieved document context:
{chr(10).join(context_blocks)}

Write the answer now with inline citations.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _sources_used_from_answer(answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used = []
    lowered_answer = answer.lower()
    for chunk in chunks:
        source = str(chunk.get("source", "")).lower()
        page = str(chunk.get("page_number", ""))
        chunk_id = str(chunk.get("chunk_id", "")).lower()
        if (source and source in lowered_answer and page in lowered_answer) or (
            chunk_id and chunk_id in lowered_answer
        ):
            used.append(chunk)
    return used or chunks


def generate_answer(
    original_query: str,
    rewritten_query: str,
    context_chunks: list[dict[str, Any]],
    chat_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not context_chunks:
        usage = _empty_usage()
        return {
            "answer": "I don't know based on the uploaded documents.",
            "sources": [],
            "response_time": 0.0,
            "model": CHAT_MODEL,
            "prompt_tokens_estimate": 0,
            "completion_tokens_estimate": 0,
            "usage": usage,
            "answer_prompt_tokens": 0,
            "answer_completion_tokens": 0,
            "answer_total_tokens": 0,
        }

    client = get_openai_client()
    messages = build_rag_prompt(rewritten_query, context_chunks)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    elapsed = time.perf_counter() - started
    answer = response.choices[0].message.content or "I don't know based on the uploaded documents."
    sources = _sources_used_from_answer(answer, context_chunks)
    prompt_text = "\n".join(message["content"] for message in messages)
    usage = _usage_to_dict(getattr(response, "usage", None))
    prompt_estimate = max(1, len(prompt_text) // 4)
    completion_estimate = max(1, len(answer) // 4)

    return {
        "answer": answer,
        "sources": sources,
        "response_time": elapsed,
        "model": CHAT_MODEL,
        "prompt_tokens_estimate": prompt_estimate,
        "completion_tokens_estimate": completion_estimate,
        "usage": usage,
        "answer_prompt_tokens": usage.get("prompt_tokens") or prompt_estimate,
        "answer_completion_tokens": usage.get("completion_tokens") or completion_estimate,
        "answer_total_tokens": usage.get("total_tokens") or (prompt_estimate + completion_estimate),
        "original_query": original_query,
        "rewritten_query": rewritten_query,
    }


def build_token_usage_summary(
    rewrite_usage: dict[str, Any],
    retrieval_usage: dict[str, Any],
    rerank_usage: dict[str, Any],
    answer_usage: dict[str, Any],
) -> dict[str, Any]:
    steps = [
        {"task": "Query rewrite", **rewrite_usage},
        {"task": "Retrieval embedding", **retrieval_usage},
        {"task": "Reranking", **rerank_usage},
        {"task": "Answer generation", **answer_usage},
    ]
    return {
        "steps": steps,
        "total": _sum_usage(steps),
    }


def get_collection_stats(collection=None) -> dict[str, Any]:
    collection = collection or get_chroma_collection()
    stats: dict[str, Any] = {
        "total_chunks": 0,
        "total_documents": 0,
        "documents": [],
        "chunks_by_document": {},
        "storage_estimate_mb": 0.0,
    }

    try:
        stats["total_chunks"] = int(collection.count())
        if stats["total_chunks"] == 0:
            return stats

        data = collection.get(include=["metadatas"])
        documents: dict[str, dict[str, Any]] = {}
        for metadata in data.get("metadatas", []):
            metadata = metadata or {}
            key = metadata.get("document_hash") or metadata.get("source", "Unknown")
            doc = documents.setdefault(
                key,
                {
                    "filename": metadata.get("source", "Unknown"),
                    "document_hash": metadata.get("document_hash", ""),
                    "pages": 0,
                    "chunks": 0,
                    "status": metadata.get("status", "Indexed"),
                    "last_ingested": metadata.get("ingestion_timestamp", ""),
                    "file_size": int(metadata.get("file_size", 0) or 0),
                    "chunking_strategy": metadata.get("chunking_strategy", "semantic"),
                    "embedding_model": metadata.get("embedding_model", ""),
                    "cross_page_context_mode": metadata.get("cross_page_context_mode", ""),
                },
            )
            doc["chunks"] += 1
            if metadata.get("embedding_model") and not doc.get("embedding_model"):
                doc["embedding_model"] = metadata.get("embedding_model", "")
            if metadata.get("cross_page_context_mode") and not doc.get("cross_page_context_mode"):
                doc["cross_page_context_mode"] = metadata.get("cross_page_context_mode", "")
            try:
                doc["pages"] = max(int(doc["pages"]), int(metadata.get("page_number", 0) or 0))
            except ValueError:
                pass
            timestamp = metadata.get("ingestion_timestamp", "")
            if timestamp and timestamp > doc["last_ingested"]:
                doc["last_ingested"] = timestamp

        docs = sorted(
            documents.values(),
            key=lambda item: item.get("last_ingested", ""),
            reverse=True,
        )
        stats["documents"] = docs
        stats["total_documents"] = len(docs)
        stats["chunks_by_document"] = {doc["filename"]: doc["chunks"] for doc in docs}
        stats["storage_estimate_mb"] = round(
            sum(doc.get("file_size", 0) for doc in docs) / (1024 * 1024),
            2,
        )
    except Exception:
        pass

    return stats


def reset_vector_db():
    ensure_project_dirs()
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
