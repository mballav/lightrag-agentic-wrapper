"""Win rates, latency percentiles, retrieval hits, and estimated cost."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from lightrag_wrapper.evaluation.datasets import EvalQuestion, Passage
from lightrag_wrapper.evaluation.judge import CRITERIA
from lightrag_wrapper.llm.usage import TokenUsage


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear percentile. ``fraction`` is between 0 and 1."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def win_rates(winners: list[str]) -> dict[str, float]:
    """Shares of agentic, native, and tie outcomes. Empty input is all zero."""
    total = len(winners)
    if total == 0:
        return {"agentic": 0.0, "native": 0.0, "tie": 0.0}
    return {
        "agentic": winners.count("agentic") / total,
        "native": winners.count("native") / total,
        "tie": winners.count("tie") / total,
    }


def category_win_rates(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Overall win rates grouped by dataset category."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        winner = (row.get("winners") or {}).get("Overall")
        if winner is None or row.get("error"):
            continue
        grouped[row.get("category") or "uncategorized"].append(winner)
    return {category: win_rates(values) for category, values in sorted(grouped.items())}


def criterion_rates(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Win rates for each criterion, using rows that have a judgment."""
    rates: dict[str, dict[str, float]] = {}
    for name in CRITERIA:
        winners = [
            row["winners"][name]
            for row in rows
            if not row.get("error") and row.get("winners") and name in row["winners"]
        ]
        rates[name] = win_rates(winners)
    return rates


def retrieval_hit(references: list[Any], supporting: tuple[Passage, ...]) -> bool | None:
    """True when a retrieved reference contains a supporting passage."""
    if not supporting:
        return None
    blob = _reference_blob(references).casefold()
    if not blob:
        return False
    for passage in supporting:
        title = passage.title.strip()
        snippet = " ".join(passage.text.split())[:120]
        if title and title.casefold() in blob:
            return True
        if len(snippet) >= 40 and snippet.casefold() in blob:
            return True
    return False


def estimate_cost(
    usages: list[TokenUsage],
    prices: dict[str, dict[str, float]],
) -> dict[str, float | int | None]:
    """Price tokens with the configured map. Actual cost is copied from the provider."""
    estimated = 0.0
    priced = 0
    actual = 0.0
    actual_calls = 0
    input_tokens = 0
    output_tokens = 0
    for usage in usages:
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        if usage.actual_cost is not None:
            actual += usage.actual_cost
            actual_calls += 1
        price = _price_for(usage.model, prices)
        if price is None:
            continue
        estimated += usage.input_tokens * price["input"] / 1_000_000
        estimated += usage.output_tokens * price["output"] / 1_000_000
        priced += 1
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "llm_calls": len(usages),
        "estimated_cost": estimated if priced else None,
        "priced_calls": priced,
        "unpriced_calls": len(usages) - priced,
        "actual_cost": actual if actual_calls else None,
    }


def summarize_latencies(values: list[float]) -> dict[str, float | None]:
    """Average, P50, and P95 for one system's latencies."""
    if not values:
        return {"average": None, "p50": None, "p95": None}
    return {
        "average": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def _reference_blob(references: list[Any]) -> str:
    parts: list[str] = []
    for reference in references:
        if isinstance(reference, dict):
            parts.append(str(reference.get("file_path") or ""))
            content = reference.get("content") or []
        else:
            parts.append(str(getattr(reference, "file_path", "") or ""))
            content = getattr(reference, "content", None) or []
        if isinstance(content, list):
            parts.extend(str(item) for item in content)
        elif content:
            parts.append(str(content))
    return "\n".join(parts)


def _price_for(model: str | None, prices: dict[str, dict[str, float]]) -> dict[str, float] | None:
    if not model:
        return None
    if model in prices:
        return prices[model]
    for name, price in prices.items():
        if model.endswith(name) or name.endswith(model):
            return price
    return None


def question_documents(question: EvalQuestion) -> tuple[Passage, ...]:
    """Passages that should be ingested for one question."""
    return question.documents or question.supporting
