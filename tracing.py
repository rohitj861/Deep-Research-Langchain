"""Langfuse tracing.

Every model call, tool call, and subagent delegation shows up as a nested span in
one trace, which is the only practical way to see what a deep agent actually did:
the orchestrator's turns and each subagent's isolated run are otherwise invisible.

Entirely optional. Without `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` the
callback list is empty and nothing changes.
"""

import logging

from langchain_core.callbacks import BaseCallbackHandler

from config import ensure_tracing_env

logger = logging.getLogger(__name__)


def build_callbacks() -> list[BaseCallbackHandler]:
    """Langfuse callback handler, or an empty list when tracing is not configured."""
    if not ensure_tracing_env():
        return []

    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception as exc:  # tracing must never break a research run
        logger.warning("Langfuse tracing disabled: %s", exc)
        return []


def flush() -> None:
    """Push buffered spans before the request ends.

    Langfuse batches in a background thread. Streamlit finishes a script run long
    before that thread would flush on its own, so traces can be lost without this.
    """
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:
        logger.debug("Langfuse flush skipped: %s", exc)


def trace_url() -> str:
    """Base URL of the configured Langfuse instance, for linking from the UI.

    Mirrors the SDK's own precedence — `LANGFUSE_BASE_URL` first — so the sidebar
    link points at the region the traces actually go to.
    """
    import os

    host = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com"
    return host.rstrip("/")
