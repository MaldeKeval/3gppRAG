"""Chroma persistent client with safe PostHog / telemetry handling.

Chroma calls ``posthog.capture(distinct_id, event_name, properties)``, which matches
PostHog's **legacy** API. PostHog 6+ only allows ``capture(event, **kwargs)``, which
raises ``capture() takes 1 positional argument but 3 were given``.

Chroma also invokes ``capture`` even when ``anonymized_telemetry`` is off (it sets
``posthog.disabled`` but does not skip the call). We patch ``posthog.capture`` before
importing Chroma so those calls are either no-ops (when disabled) or mapped to the new
keyword form.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import posthog


def _install_posthog_chroma_shim() -> None:
    """Must run before ``import chromadb`` (first use of this module installs it)."""
    if getattr(posthog, "_rag_chroma_shim_installed", False):
        return

    _orig = posthog.capture

    def capture_compat(*args: Any, **kwargs: Any):
        if getattr(posthog, "disabled", False):
            return None
        # Legacy 3-arg form used by chromadb.telemetry.product.posthog.Posthog
        if (
            len(args) == 3
            and not kwargs
            and isinstance(args[0], str)
            and isinstance(args[1], str)
            and isinstance(args[2], dict)
        ):
            distinct_id, event_name, properties = args
            return _orig(event_name, distinct_id=distinct_id, properties=properties)
        return _orig(*args, **kwargs)

    posthog.capture = capture_compat  # type: ignore[assignment]
    posthog._rag_chroma_shim_installed = True  # type: ignore[attr-defined]


_install_posthog_chroma_shim()

import chromadb  # noqa: E402  # import after shim
from chromadb.config import Settings as ChromaSettings  # noqa: E402


def persistent_chroma_client(path: str | Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
