"""Token usage taken from a completion response the client already received."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_usage_bucket: ContextVar[list[TokenUsage] | None] = ContextVar(
    "token_usage_bucket",
    default=None,
)


@dataclass(frozen=True)
class TokenUsage:
    """One completion's provider usage. Cost is set only when the provider sent it."""

    model: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    actual_cost: float | None = None


def begin_token_usage() -> None:
    """Start a fresh usage list for the current task."""
    _usage_bucket.set([])


def note_token_usage(response: Any) -> None:
    """Append usage when a bucket is active and the response reports it."""
    bucket = _usage_bucket.get()
    if bucket is None:
        return
    usage = token_usage_from_response(response)
    if usage is not None:
        bucket.append(usage)


def take_token_usage() -> list[TokenUsage]:
    """Return and clear the usage recorded for the current task."""
    bucket = _usage_bucket.get()
    _usage_bucket.set(None)
    return list(bucket or [])


def token_usage_from_response(response: Any) -> TokenUsage | None:
    """Read provider usage from a LiteLLM or OpenAI-style response."""
    raw = _field(response, "usage")
    if raw is None:
        return None
    input_tokens = _int(raw, "prompt_tokens", "input_tokens")
    output_tokens = _int(raw, "completion_tokens", "output_tokens")
    total_tokens = _int(raw, "total_tokens")
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    if input_tokens == 0 and output_tokens == 0 and total_tokens == 0:
        return None
    cost = _float(raw, "cost")
    if cost is None:
        cost = _float(response, "_hidden_params", "response_cost")
    model = _field(response, "model")
    return TokenUsage(
        model=model if isinstance(model, str) else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        actual_cost=cost,
    )


def _field(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _int(value: Any, *names: str) -> int:
    for name in names:
        found = _field(value, name)
        if isinstance(found, bool):
            continue
        if isinstance(found, (int, float)):
            return int(found)
    return 0


def _float(value: Any, *path: str) -> float | None:
    current = value
    for name in path:
        current = _field(current, name)
        if current is None:
            return None
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)
