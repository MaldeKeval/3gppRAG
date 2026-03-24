"""CLI: rebuild index, single question, or interactive REPL."""

from __future__ import annotations

import argparse
import sys

import httpx

from rag.config import load_settings
from rag.ingest import build_index
from rag.rag_engine import answer_question, load_collection


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local RAG over raoulbia/3gpp-5g-nr-qa using Ollama + Chroma."
    )
    p.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the Chroma index from the Hugging Face dataset (requires Ollama embeddings model).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Index only the first N rows (for smoke tests).",
    )
    p.add_argument(
        "-q",
        "--question",
        type=str,
        default=None,
        help="Ask a single question and exit (non-interactive).",
    )
    return p.parse_args(argv)


def run_repl(settings, http: httpx.Client, collection) -> None:
    print("3GPP NR RAG — type a question (empty line to quit).")
    print(f"Chat model: {settings.chat_model} | Embed model: {settings.embed_model}")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        print()
        answer_question(settings, line, http, collection)
        print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()

    http = httpx.Client()

    try:
        if args.rebuild_index:
            n = build_index(settings, limit=args.limit, http_client=http)
            print(f"Indexed {n} documents into {settings.chroma_path}")
            if args.question is None:
                return 0

        collection = load_collection(settings)

        if args.question is not None:
            answer_question(settings, args.question, http, collection)
            return 0

        run_repl(settings, http, collection)
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        print(
            "Hint: run with --rebuild-index (and ensure Ollama is running with the embedding model).",
            file=sys.stderr,
        )
        return 1
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    except httpx.HTTPError as e:
        print(f"HTTP error talking to Ollama: {e}", file=sys.stderr)
        return 1
    finally:
        http.close()


if __name__ == "__main__":
    raise SystemExit(main())
