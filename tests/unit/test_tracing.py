"""Langfuse is off unless LANGFUSE_TRACING_ENABLED is true."""

from types import SimpleNamespace

from lightrag_wrapper.tracing import (
    apply_litellm_tracing,
    langfuse_tracing_enabled,
    langgraph_callbacks,
)


def test_tracing_is_off_unless_the_flag_is_true(monkeypatch):
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)
    assert langfuse_tracing_enabled() is False
    for value in ("true", "TRUE", "1", "yes", "on"):
        monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", value)
        assert langfuse_tracing_enabled() is True
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    assert langfuse_tracing_enabled() is False


def test_litellm_callback_follows_the_flag(monkeypatch):
    litellm = SimpleNamespace(callbacks=["other"])
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    apply_litellm_tracing(litellm)
    assert litellm.callbacks == ["other", "langfuse_otel"]
    apply_litellm_tracing(litellm)
    assert litellm.callbacks == ["other", "langfuse_otel"]

    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    apply_litellm_tracing(litellm)
    assert litellm.callbacks == ["other"]


def test_langgraph_callbacks_are_empty_when_tracing_is_off(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    assert langgraph_callbacks() == []


def test_ingest_suppression_skips_langfuse_only_when_tracing_is_on(monkeypatch):
    from lightrag_wrapper.tracing import apply_call_tracing, without_langfuse

    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    ordinary: dict = {}
    apply_call_tracing(ordinary)
    assert "no-log" not in ordinary

    with without_langfuse():
        ingest_call: dict = {}
        apply_call_tracing(ingest_call)
    assert ingest_call["no-log"] is True
    after: dict = {}
    apply_call_tracing(after)
    assert "no-log" not in after

    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    with without_langfuse():
        disabled: dict = {}
        apply_call_tracing(disabled)
    assert "no-log" not in disabled
