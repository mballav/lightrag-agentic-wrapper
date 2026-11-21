"""Langfuse switches shared by the API and the evaluation CLI.

``LANGFUSE_TRACING_ENABLED`` turns LiteLLM and LangGraph tracing on or off.
The app service reads it from ``env/wrapper/.env.*`` through Compose ``env_file``.
Evaluation loads that same file before the first model call.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from lightrag_wrapper.log import get_logger

logger = get_logger(__name__)

_TRUE = frozenset({"1", "true", "yes", "on"})
_litellm_tracing: bool | None = None
_langgraph_handler: Any | None = None
_suppress_langfuse: ContextVar[bool] = ContextVar("suppress_langfuse", default=False)


def langfuse_tracing_enabled() -> bool:
    """True only when ``LANGFUSE_TRACING_ENABLED`` is an explicit true value."""
    raw = os.getenv("LANGFUSE_TRACING_ENABLED", "false")
    return raw.strip().lower() in _TRUE


def langfuse_suppressed() -> bool:
    """True while an ingest call has turned Langfuse off for its model calls."""
    return _suppress_langfuse.get()


@contextmanager
def without_langfuse() -> Iterator[None]:
    """Skip Langfuse for model calls made inside this block, including child tasks."""
    token = _suppress_langfuse.set(True)
    try:
        yield
    finally:
        _suppress_langfuse.reset(token)


def apply_call_tracing(kwargs: dict[str, Any]) -> None:
    """Tell LiteLLM to skip callbacks for one call when ingest has suppressed Langfuse.

    ``no-log`` is per request, so a query running at the same time still follows
    ``LANGFUSE_TRACING_ENABLED``. When that flag is off there is no callback to skip.
    """
    if langfuse_suppressed() and langfuse_tracing_enabled():
        kwargs["no-log"] = True


def apply_litellm_tracing(litellm: Any) -> None:
    """Add or remove the LiteLLM Langfuse callback to match the env flag."""
    global _litellm_tracing
    enabled = langfuse_tracing_enabled()
    current = list(getattr(litellm, "callbacks", None) or [])
    if enabled:
        if "langfuse_otel" not in current:
            current.append("langfuse_otel")
    else:
        current = [item for item in current if item != "langfuse_otel"]
    litellm.callbacks = current
    if _litellm_tracing != enabled:
        logger.info("Langfuse LiteLLM tracing", enabled=enabled)
        _litellm_tracing = enabled


def langgraph_callbacks() -> list[Any]:
    """LangGraph callbacks for one plan run. Empty when tracing is off."""
    global _langgraph_handler
    if not langfuse_tracing_enabled():
        return []
    if _langgraph_handler is None:
        from langfuse.langchain import CallbackHandler

        _langgraph_handler = CallbackHandler()
        logger.info("Langfuse LangGraph tracing", enabled=True)
    return [_langgraph_handler]
