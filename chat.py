from __future__ import annotations

from rag_utils import (
    CHAT_MODEL,
    build_token_usage_summary,
    generate_answer,
    get_chroma_collection,
    retrieve_context_with_usage,
    rerank_chunks_with_usage,
    rewrite_query_result,
)


def print_sources(sources: list[dict]) -> None:
    if not sources:
        return
    print("\nSources used")
    for index, source in enumerate(sources, start=1):
        similarity = source.get("similarity")
        rerank = source.get("rerank_score")
        page_range = str(source.get("page_range") or source.get("page_number") or "?")
        page_label = "pages" if "-" in page_range else "page"
        print(
            f"{index}. {source.get('source')} {page_label} {page_range} "
            f"chunk {source.get('chunk_id')} | similarity={similarity} | rerank={rerank}"
        )


def print_token_usage(token_usage: dict) -> None:
    if not token_usage:
        return
    print("\nToken usage")
    for step in token_usage.get("steps", []):
        print(
            f"- {step.get('task')}: input={step.get('prompt_tokens', 0)}, "
            f"output={step.get('completion_tokens', 0)}, total={step.get('total_tokens', 0)}"
        )
    total = token_usage.get("total", {})
    print(f"Total: {total.get('total_tokens', 0)} tokens")


def main() -> None:
    print("RAG Knowledge Assistant terminal chat")
    print("Type exit to stop.\n")
    collection = get_chroma_collection()
    if collection.count() == 0:
        print("No indexed chunks found. Run `python ingest.py` or use the Streamlit Documents page first.")
        return

    chat_history: list[dict] = []
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        try:
            rewrite_result = rewrite_query_result(question, chat_history)
            rewritten = rewrite_result["query"]
            retrieval_result = retrieve_context_with_usage(rewritten, collection=collection, top_k=10)
            retrieved = retrieval_result["chunks"]
            rerank_result = rerank_chunks_with_usage(rewritten, retrieved, top_n=5)
            reranked = rerank_result["chunks"]
            result = generate_answer(question, rewritten, reranked, chat_history)
            token_usage = build_token_usage_summary(
                rewrite_result.get("usage", {}),
                retrieval_result.get("usage", {}),
                rerank_result.get("usage", {}),
                result.get("usage", {}),
            )
        except Exception as exc:
            print(f"Could not answer: {exc}")
            continue

        print(f"\nAssistant ({CHAT_MODEL}): {result['answer']}")
        print_sources(result.get("sources", []))
        print_token_usage(token_usage)
        print(f"\nRewritten query: {rewritten}\n")

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": result["answer"]})
        chat_history = chat_history[-12:]


if __name__ == "__main__":
    main()
