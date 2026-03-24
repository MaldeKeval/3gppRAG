"""Retrieve context from Chroma and answer with Ollama chat."""

from __future__ import annotations

import httpx

from rag.chroma_client import persistent_chroma_client
from rag.config import Settings
from rag.ollama_api import chat_stream, embed_text


SYSTEM_PROMPT = """You are a technical assistant for 3GPP 5G New Radio (NR) specifications.
Answer using ONLY the provided context snippets when they are relevant.
If the context does not contain enough information, say so clearly and answer only with what can be inferred from the context.
Use precise telecommunications terminology when appropriate."""


def _truncate_context(blocks: list[str], max_chars: int) -> str:
    parts: list[str] = []
    total = 0
    for i, block in enumerate(blocks, start=1):
        header = f"--- Snippet {i} ---\n"
        piece = header + block
        if total + len(piece) > max_chars:
            remaining = max_chars - total - len(header)
            if remaining <= 0:
                break
            piece = header + block[:remaining] + "\n[truncated]"
            parts.append(piece)
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


def load_collection(settings: Settings):
    if not settings.chroma_path.exists():
        raise FileNotFoundError(
            f"Chroma data not found at {settings.chroma_path}. "
            "Run with --rebuild-index first."
        )
    client = persistent_chroma_client(settings.chroma_path)
    col = client.get_collection(name=settings.collection_name)
    meta = col.metadata or {}
    stored = meta.get("embed_model")
    if stored and stored != settings.embed_model:
        raise ValueError(
            f"Index was built with embed_model={stored!r} but RAG_EMBED_MODEL is "
            f"{settings.embed_model!r}. Rebuild the index or fix the env var."
        )
    return col


def retrieve_context(
    settings: Settings,
    question: str,
    http: httpx.Client,
    collection,
) -> tuple[str, list[str]]:
    q_emb = embed_text(http, settings.ollama_host, settings.embed_model, question.strip())
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=settings.top_k,
        include=["documents", "distances"],
    )
    docs = (res.get("documents") or [[]])[0]
    context = _truncate_context(docs, settings.max_context_chars)
    return context, docs


def answer_question(
    settings: Settings,
    question: str,
    http: httpx.Client,
    collection,
) -> None:
    context, _docs = retrieve_context(settings, question, http, collection)
    user_content = (
        "Context from the knowledge base:\n"
        f"{context}\n\n"
        f"User question: {question.strip()}"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    for chunk in chat_stream(http, settings.ollama_host, settings.chat_model, messages):
        print(chunk, end="", flush=True)
    print()
