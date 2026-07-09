# RAG Knowledge Assistant

**Additional Information | Product + AI Portfolio Artifact**

**Rohan Singh Kandra**

**Subtitle:** Source-grounded AI assistant for document search, cited answers, and retrieval transparency.

**Live demo:** https://ragknowledgeassistant.streamlit.app/

**GitHub:** https://github.com/rohankandra1998-pmtech/rag-knowledge-assistant

## Problem

Knowledge workers often search across long PDFs and internal documents to answer specific questions. Traditional search returns documents, not answers. Generic LLM responses can hallucinate or lack evidence. This product focuses on fast, source-grounded answers users can verify.

## Product Solution

The assistant turns static PDFs into an interactive knowledge layer where users can move from answer to evidence to source.

**Workflow:** Upload PDFs -> Ingest documents -> Ask questions -> Get cited answers -> Inspect sources -> Review retrieval logic

The app extracts PDF text, chunks documents, embeds content, stores searchable chunks in ChromaDB, retrieves relevant evidence, reranks sources, and generates answers with inline citations.

## Product Value

Built for trust-sensitive document workflows where users need citations, source visibility, and honest no-answer behavior instead of unsupported AI guesses.

Designed for document-heavy workflows where users need answers they can verify, not just AI-generated summaries. The assistant prioritizes source grounding, citation visibility, and honest no-answer behavior.

## Product Highlights

- Multi-PDF upload and ingestion
- Conversational Q&A over indexed documents
- Inline citations with filename, page, and chunk
- Source panels and PDF preview for verification
- Behind-the-scenes retrieval/debug visibility

## Product Decisions I Made

- Problem Framing: Framed the product around a trust gap: users need verifiable answers from long documents, not another generic chatbot.
- User Journey Design: Mapped the workflow from upload -> ingestion -> question -> cited answer -> source inspection to reduce cognitive load.
- MVP Prioritization: Prioritized citations, source panels, PDF preview, and no-answer behavior before advanced features.
- Trust-First UX: Designed the experience so users can validate the answer, inspect evidence, and understand when the AI lacks support.
- AI Evaluation: Created positive, negative, edge-case, and unsupported-question tests to check grounding and hallucination risk.
- Observability: Added retrieval/debug visibility, source chunks, and token usage to make AI behavior easier to inspect and improve.

## Architecture Snapshot

**Pipeline:** Input -> Indexing -> Storage -> Retrieval -> Answer -> Verification

- Input: PDF Upload + pypdf Extraction
- Indexing: Semantic Chunking + Embeddings
- Storage: Persistent ChromaDB
- Retrieval: Query Rewrite + Vector Retrieval + Reranking
- Answer: LLM Answer + Inline Citations
- Verification: Source Panel + PDF Preview + Retrieval Debug

Repository-supported technical details:

- Python
- Streamlit
- OpenAI LLM
- OpenAI text-embedding-3-large
- gpt-4.1-mini for query rewriting, reranking, and answer generation
- LangChain SemanticChunker
- Persistent ChromaDB vector store
- pypdf page extraction
- SHA-256 duplicate document prevention
- Inline citations with filename, page, and chunk
- Token usage and retrieval observability

## Trust & Evaluation

- Answers grounded in retrieved document chunks
- Citations help users verify claims
- No-answer behavior for unsupported questions
- Edge-case prompts test hallucination risk
- Source panels and PDF preview reduce black-box behavior

## UX / Observability Decisions

- Ingestion status cards show processing state
- Indexed document views expose stored files and chunks
- Source inspection shows retrieved evidence
- Retrieval/debug panels help diagnose answer quality
- Token usage visibility supports cost awareness

## Tech Stack

Python | Streamlit | OpenAI | text-embedding-3-large | gpt-4.1-mini | LangChain SemanticChunker | ChromaDB | pypdf | RAG | Retrieval/Reranking | Citations
