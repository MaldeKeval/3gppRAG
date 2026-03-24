"""HTTP helpers for Ollama embeddings and chat."""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx


def embed_text(client: httpx.Client, host: str, model: str, text: str) -> list[float]:
    url = f"{host}/api/embeddings"
    r = client.post(url, json={"model": model, "prompt": text}, timeout=120.0)
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list):
        raise RuntimeError(f"Unexpected embeddings response: {data!r}")
    return emb


def chat_stream(
    client: httpx.Client,
    host: str,
    model: str,
    messages: list[dict[str, str]],
) -> Iterator[str]:
    url = f"{host}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    with client.stream("POST", url, json=payload, timeout=300.0) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chunk.get("done"):
                break
            msg = chunk.get("message") or {}
            content = msg.get("content")
            if content:
                yield content
