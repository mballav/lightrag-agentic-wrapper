"""JSON, CSV, and a short markdown summary for one evaluation run."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightrag_wrapper.evaluation.judge import CRITERIA
from lightrag_wrapper.evaluation.metrics import (
    category_win_rates,
    criterion_rates,
    estimate_cost,
    retrieval_hit,
    summarize_latencies,
)
from lightrag_wrapper.evaluation.runner import QuestionComparison


def comparison_row(
    comparison: QuestionComparison,
    prices: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """One question-level record."""
    native = comparison.native
    agentic = comparison.agentic
    native_cost = estimate_cost(native.usages, prices) if native else _empty_cost()
    agentic_cost = estimate_cost(agentic.usages, prices) if agentic else _empty_cost()
    judge_cost = estimate_cost(comparison.judge_usages, prices)
    winners = comparison.judgment.winners if comparison.judgment else None
    return {
        "id": comparison.question.id,
        "category": comparison.question.category,
        "question": comparison.question.question,
        "reference_answer": comparison.question.reference_answer,
        "native_answer": native.answer if native else "",
        "agentic_answer": agentic.answer if agentic else "",
        "winners": winners,
        "native_latency_s": native.latency_s if native else None,
        "agentic_latency_s": agentic.latency_s if agentic else None,
        "native_retrievals": native.retrievals if native else 0,
        "agentic_retrievals": agentic.retrievals if agentic else 0,
        "native_calls_by_component": native.calls_by_component if native else {},
        "agentic_calls_by_component": agentic.calls_by_component if agentic else {},
        "native_usage": native_cost,
        "agentic_usage": agentic_cost,
        "judge_usage": judge_cost,
        "native_retrieval_hit": retrieval_hit(native.references, comparison.question.supporting) if native else None,
        "agentic_retrieval_hit": retrieval_hit(agentic.references, comparison.question.supporting) if agentic else None,
        "judgments": _judgments(comparison),
        "error": comparison.error,
    }


def write_run(
    comparisons: list[QuestionComparison],
    *,
    output_dir: Path,
    dataset: str,
    prices: dict[str, dict[str, float]],
) -> Path:
    """Write results.json, results.csv, and report.md. Return the directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [comparison_row(item, prices) for item in comparisons]
    (output_dir / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(output_dir / "results.csv", rows)
    (output_dir / "report.md").write_text(
        render_report(rows, dataset=dataset),
        encoding="utf-8",
    )
    return output_dir


def render_report(rows: list[dict[str, Any]], *, dataset: str) -> str:
    """Human-readable summary of one run."""
    answered = [row for row in rows if not row.get("error")]
    rates = criterion_rates(answered)
    categories = category_win_rates(answered)
    native_latency = summarize_latencies(_numbers(answered, "native_latency_s"))
    agentic_latency = summarize_latencies(_numbers(answered, "agentic_latency_s"))
    native_hits = _hits(answered, "native_retrieval_hit")
    agentic_hits = _hits(answered, "agentic_retrieval_hit")
    lines = [
        "# Evaluation report",
        "",
        f"Recorded {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}.",
        "",
        f"Dataset: {dataset}",
        f"Questions evaluated: {len(rows)}",
        f"Questions with an error: {len(rows) - len(answered)}",
        "",
        "## Answer quality",
        "",
        _rate_block(rates),
        "",
        "## Retrieval",
        "",
        f"Native supporting-passage hit rate: {_rate(native_hits)}",
        f"Agentic supporting-passage hit rate: {_rate(agentic_hits)}",
        "",
        "## Latency (seconds)",
        "",
        _latency_line("Native", native_latency),
        _latency_line("Agentic", agentic_latency),
        "",
        "## Tokens and estimated cost",
        "",
        _usage_line("Native", answered, "native_usage"),
        _usage_line("Agentic", answered, "agentic_usage"),
        _usage_line("Judge", answered, "judge_usage"),
        "",
        "Estimated cost uses the configured price map. Actual cost is present only when the provider returned one.",
        "",
        "## Category win rates (overall)",
        "",
    ]
    if not categories:
        lines.append("No category labels were available.")
    else:
        for category, values in categories.items():
            lines.append(
                f"- {category}: agentic {_percent(values['agentic'])}, "
                f"native {_percent(values['native'])}, tie {_percent(values['tie'])}"
            )
    lines.extend(["", "## Examples", ""])
    lines.extend(_examples(answered))
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "category",
        "question",
        "reference_answer",
        "native_answer",
        "agentic_answer",
        "overall",
        "comprehensiveness",
        "diversity",
        "empowerment",
        "native_latency_s",
        "agentic_latency_s",
        "native_retrievals",
        "agentic_retrievals",
        "native_input_tokens",
        "native_output_tokens",
        "agentic_input_tokens",
        "agentic_output_tokens",
        "native_estimated_cost",
        "agentic_estimated_cost",
        "native_actual_cost",
        "agentic_actual_cost",
        "native_retrieval_hit",
        "agentic_retrieval_hit",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(fieldnames=fieldnames, f=handle)
        writer.writeheader()
        for row in rows:
            winners = row.get("winners") or {}
            native_usage = row.get("native_usage") or {}
            agentic_usage = row.get("agentic_usage") or {}
            writer.writerow(
                {
                    "id": row["id"],
                    "category": row.get("category") or "",
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "native_answer": row["native_answer"],
                    "agentic_answer": row["agentic_answer"],
                    "overall": winners.get("Overall", ""),
                    "comprehensiveness": winners.get("Comprehensiveness", ""),
                    "diversity": winners.get("Diversity", ""),
                    "empowerment": winners.get("Empowerment", ""),
                    "native_latency_s": row.get("native_latency_s"),
                    "agentic_latency_s": row.get("agentic_latency_s"),
                    "native_retrievals": row.get("native_retrievals"),
                    "agentic_retrievals": row.get("agentic_retrievals"),
                    "native_input_tokens": native_usage.get("input_tokens"),
                    "native_output_tokens": native_usage.get("output_tokens"),
                    "agentic_input_tokens": agentic_usage.get("input_tokens"),
                    "agentic_output_tokens": agentic_usage.get("output_tokens"),
                    "native_estimated_cost": native_usage.get("estimated_cost"),
                    "agentic_estimated_cost": agentic_usage.get("estimated_cost"),
                    "native_actual_cost": native_usage.get("actual_cost"),
                    "agentic_actual_cost": agentic_usage.get("actual_cost"),
                    "native_retrieval_hit": row.get("native_retrieval_hit"),
                    "agentic_retrieval_hit": row.get("agentic_retrieval_hit"),
                    "error": row.get("error") or "",
                }
            )


def _judgments(comparison: QuestionComparison) -> list[dict[str, Any]]:
    if comparison.judgment is None:
        return []
    exported: list[dict[str, Any]] = []
    for judgment in comparison.judgment.judgments:
        exported.append(
            {
                "agentic_first": judgment.agentic_first,
                "criteria": {
                    name: asdict(item) for name, item in judgment.criteria.items()
                },
            }
        )
    return exported


def _empty_cost() -> dict[str, float | int | None]:
    return estimate_cost([], {})


def _numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def _hits(rows: list[dict[str, Any]], key: str) -> list[bool]:
    return [bool(row[key]) for row in rows if isinstance(row.get(key), bool)]


def _rate(values: list[bool]) -> str:
    if not values:
        return "not available"
    return _percent(sum(1 for value in values if value) / len(values))


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _rate_block(rates: dict[str, dict[str, float]]) -> str:
    lines = []
    for name in CRITERIA:
        values = rates.get(name) or {"agentic": 0.0, "native": 0.0, "tie": 0.0}
        label = "Overall" if name == "Overall" else name
        lines.append(
            f"- {label}: agentic {_percent(values['agentic'])}, "
            f"native {_percent(values['native'])}, tie {_percent(values['tie'])}"
        )
    return "\n".join(lines)


def _latency_line(label: str, summary: dict[str, float | None]) -> str:
    return (
        f"- {label}: average {_seconds(summary['average'])}, "
        f"P50 {_seconds(summary['p50'])}, P95 {_seconds(summary['p95'])}"
    )


def _seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _usage_line(label: str, rows: list[dict[str, Any]], key: str) -> str:
    input_tokens = 0
    output_tokens = 0
    estimated = 0.0
    estimated_known = False
    actual = 0.0
    actual_known = False
    calls = 0
    for row in rows:
        usage = row.get(key) or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        calls += int(usage.get("llm_calls") or 0)
        if usage.get("estimated_cost") is not None:
            estimated += float(usage["estimated_cost"])
            estimated_known = True
        if usage.get("actual_cost") is not None:
            actual += float(usage["actual_cost"])
            actual_known = True
    count = len(rows) or 1
    estimated_text = f"{estimated:.6f}" if estimated_known else "not estimated"
    per_question = f"{estimated / count:.6f}" if estimated_known else "not estimated"
    actual_text = f"{actual:.6f}" if actual_known else "not reported"
    return (
        f"- {label}: calls {calls}, input tokens {input_tokens}, output tokens {output_tokens}, "
        f"total tokens {input_tokens + output_tokens}, estimated total {estimated_text}, "
        f"estimated per question {per_question}, actual {actual_text}"
    )


def _examples(rows: list[dict[str, Any]]) -> list[str]:
    groups = {
        "Agentic win": [row for row in rows if (row.get("winners") or {}).get("Overall") == "agentic"],
        "Native win": [row for row in rows if (row.get("winners") or {}).get("Overall") == "native"],
        "Poor answers": [row for row in rows if _poor(row)],
    }
    lines: list[str] = []
    for title, chosen in groups.items():
        lines.append(f"### {title}")
        lines.append("")
        if not chosen:
            lines.append("None in this run.")
            lines.append("")
            continue
        for row in chosen[:2]:
            lines.append(f"- {row['id']}: {row['question']}")
        lines.append("")
    return lines


def _poor(row: dict[str, Any]) -> bool:
    native = (row.get("native_answer") or "").strip()
    agentic = (row.get("agentic_answer") or "").strip()
    return len(native) < 40 and len(agentic) < 40
