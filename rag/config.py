"""Configuration loaded from environment variables with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v.strip() if v else default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None or not v.strip():
        return default
    return int(v)


@dataclass(frozen=True)
class Settings:
    """RAG runtime settings. Override via environment variables."""

    ollama_host: str
    embed_model: str
    chat_model: str
    chroma_path: Path
    collection_name: str
    top_k: int
    max_context_chars: int
    dataset_name: str
    dataset_split: str


def load_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    default_chroma = root / "data" / "chroma"

    return Settings(
        ollama_host=_env_str("RAG_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
        embed_model=_env_str("RAG_EMBED_MODEL", "nomic-embed-text"),
        chat_model=_env_str("RAG_CHAT_MODEL", "qwen2.5:7b"),
        chroma_path=Path(_env_str("RAG_CHROMA_PATH", str(default_chroma))),
        collection_name=_env_str("RAG_COLLECTION_NAME", "3gpp_5g_nr_qa"),
        top_k=_env_int("RAG_TOP_K", 5),
        max_context_chars=_env_int("RAG_MAX_CONTEXT_CHARS", 12000),
        dataset_name=_env_str("RAG_DATASET", "raoulbia/3gpp-5g-nr-qa"),
        dataset_split=_env_str("RAG_DATASET_SPLIT", "train"),
    )
