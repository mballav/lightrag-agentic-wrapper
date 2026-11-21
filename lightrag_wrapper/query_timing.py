"""Per-query timings for the rewriter, the decomposer, and the DAG."""

from __future__ import annotations

import os
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_TIMES_LOG = Path("/app/data/wrapper/logs/times.log")
_file_lock = threading.Lock()
_reported_llm_seconds: ContextVar[float | None] = ContextVar(
    "reported_llm_seconds",
    default=None,
)


def clear_reported_llm_seconds() -> None:
    """Drop any response time left by an earlier call on this task."""
    _reported_llm_seconds.set(None)


def note_reported_llm_seconds(response: Any) -> None:
    """Remember the response time attached to one LLM response."""
    _reported_llm_seconds.set(reported_llm_seconds(response))


def take_reported_llm_seconds() -> float | None:
    """Return and clear the response time recorded for the current task."""
    value = _reported_llm_seconds.get()
    _reported_llm_seconds.set(None)
    return value


def reported_llm_seconds(response: Any) -> float | None:
    """Return the response time on an LLM response, in seconds.

    LiteLLM stores that duration on ``_response_ms`` in milliseconds.
    A ``response_time`` field is already in seconds.
    """
    milliseconds = getattr(response, "_response_ms", None)
    if milliseconds is None and isinstance(response, dict):
        milliseconds = response.get("_response_ms")
    if isinstance(milliseconds, (int, float)):
        return float(milliseconds) / 1000.0

    reported = getattr(response, "response_time", None)
    if reported is None and isinstance(response, dict):
        reported = response.get("response_time")
    if isinstance(reported, (int, float)):
        return float(reported)
    return None


def times_log_path() -> Path:
    """Path of the query timing report. ``QUERY_TIMES_LOG`` overrides the default."""
    override = os.environ.get("QUERY_TIMES_LOG")
    if override:
        return Path(override)
    return _DEFAULT_TIMES_LOG


@dataclass(frozen=True)
class DagCall:
    """One timed LLM or retrieval call inside a plan node."""

    node_id: int
    node_type: str
    step: str
    call_seconds: float
    response_seconds: float | None
    detail: str | None = None


@dataclass
class QueryTiming:
    """Timings for one answered query. Parallel nodes share this object."""

    query: str
    rewriter_calls_s: list[float] = field(default_factory=list)
    decomposer_calls_s: list[float] = field(default_factory=list)
    dag_llm_call_s: float = 0.0
    dag_llm_calls: int = 0
    dag_llm_response_s: float = 0.0
    dag_llm_response_samples: int = 0
    dag_node_s: float = 0.0
    dag_nodes: int = 0
    dag_calls: list[DagCall] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_rewriter_call(self, call_seconds: float) -> None:
        with self._lock:
            self.rewriter_calls_s.append(call_seconds)

    def add_decomposer_call(self, call_seconds: float) -> None:
        with self._lock:
            self.decomposer_calls_s.append(call_seconds)

    def add_dag_llm(
        self,
        call_seconds: float,
        response_seconds: float | None,
        *,
        node_id: int | None = None,
        node_type: str | None = None,
        step: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Add one DAG LLM or retrieval call.

        ``call_seconds`` is the wall clock around the call.
        ``response_seconds`` is the ``response_time`` the call returned.
        ``node_id``, ``node_type``, and ``step`` name that call in the report.
        ``detail`` is extra text for that step, such as a resolved search string.
        """
        with self._lock:
            self.dag_llm_call_s += call_seconds
            self.dag_llm_calls += 1
            if response_seconds is not None:
                self.dag_llm_response_s += response_seconds
                self.dag_llm_response_samples += 1
            if node_id is not None and node_type is not None and step is not None:
                self.dag_calls.append(
                    DagCall(
                        node_id=node_id,
                        node_type=node_type,
                        step=step,
                        call_seconds=call_seconds,
                        response_seconds=response_seconds,
                        detail=detail,
                    )
                )

    def add_node(self, execution_seconds: float) -> None:
        with self._lock:
            self.dag_node_s += execution_seconds
            self.dag_nodes += 1


def render_query_times(
    timing: QueryTiming,
    *,
    dag_executed: bool,
    recorded_at: datetime | None = None,
) -> str:
    """Format one report segment. The query text is the first block."""
    moment = recorded_at or datetime.now(timezone.utc)
    stamp = moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "=" * 80,
        "Query",
        timing.query,
        "",
        stamp,
        "",
        "Query rewriter",
        _call_lines("LLM call time", timing.rewriter_calls_s),
        "",
        "Query decomposer",
        _call_lines("LLM call time", timing.decomposer_calls_s),
        "",
        "DAG (accumulated over all nodes)",
    ]
    if not dag_executed:
        lines.append("not executed")
    else:
        lines.extend(
            [
                (
                    f"LLM call time: {_seconds(timing.dag_llm_call_s)} "
                    f"({timing.dag_llm_calls} calls)"
                ),
                (
                    f"LLM response time: {_seconds(timing.dag_llm_response_s)} "
                    f"({timing.dag_llm_response_samples} of {timing.dag_llm_calls} "
                    "calls returned a response_time)"
                ),
                (
                    f"Node execution time: {_seconds(timing.dag_node_s)} "
                    f"({timing.dag_nodes} nodes)"
                ),
                *_dag_call_lines(timing.dag_calls),
            ]
        )
    lines.extend(["=" * 80, ""])
    return "\n".join(lines)


def write_query_times(timing: QueryTiming, *, dag_executed: bool) -> None:
    """Append one report segment to the timing log."""
    path = times_log_path()
    segment = render_query_times(timing, dag_executed=dag_executed)
    with _file_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(segment)


def _dag_call_lines(calls: list[DagCall]) -> list[str]:
    """Per-call lines, grouped by plan node and kept in call order."""
    if not calls:
        return []
    grouped: dict[tuple[int, str], list[DagCall]] = {}
    for call in calls:
        grouped.setdefault((call.node_id, call.node_type), []).append(call)
    lines = ["", "LLM calls"]
    for node_id, node_type in sorted(grouped):
        lines.append(f"Node {node_id} ({node_type})")
        for call in grouped[(node_id, node_type)]:
            if call.response_seconds is None:
                response = "none"
            else:
                response = _seconds(call.response_seconds)
            lines.append(
                f"  {call.step}: LLM call time {_seconds(call.call_seconds)}, "
                f"LLM response time {response}"
            )
            if call.step == "resolve" and call.detail:
                lines.append(f"    resolved query: {call.detail}")
    return lines


def _call_lines(label: str, calls: list[float]) -> str:
    if not calls:
        return f"{label}: no call"
    if len(calls) == 1:
        return f"{label}: {_seconds(calls[0])}"
    numbered = "\n".join(
        f"  {index}. {_seconds(seconds)}" for index, seconds in enumerate(calls, start=1)
    )
    return f"{label}:\n{numbered}\n  total: {_seconds(sum(calls))}"


def _seconds(value: float) -> str:
    return f"{value:.3f} s"
