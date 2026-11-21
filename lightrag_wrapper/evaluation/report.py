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
        "plan": agentic.plan if agentic else None,
        "judgments": _judgments(comparison),
        "error": comparison.error,
        "repeat": comparison.repeat,
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
    groups = _repeat_groups(answered)
    repeat_count = len({int(row.get("repeat") or 1) for row in rows})
    lines = [
        "# Evaluation report",
        "",
        f"Recorded {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}.",
        "",
        f"Dataset: {dataset}",
        f"Questions evaluated: {len({row['id'] for row in rows})}",
        f"Questions with an error: {len(rows) - len(answered)}",
    ]
    if repeat_count > 1:
        lines.extend(
            [
                f"Repeats: {repeat_count}",
                f"Question runs: {len(rows)}",
                "",
                "## Aggregated",
                "",
                "Pooled across every repeat.",
                "",
            ]
        )
        lines.extend(_metric_sections(answered, heading="###"))
        lines.extend(
            [
                "## Average",
                "",
                "Mean of each figure across repeats.",
                "",
            ]
        )
        lines.extend(_average_sections(groups))
    else:
        lines.append("")
        lines.extend(_metric_sections(answered, heading="##"))
    lines.extend(
        [
            "Estimated cost uses the configured price map. Actual cost is present only when the provider returned one.",
            "",
            "## Examples",
            "",
        ]
    )
    lines.extend(_examples(answered))
    lines.append("")
    return "\n".join(lines)


def _metric_sections(rows: list[dict[str, Any]], *, heading: str) -> list[str]:
    """Answer quality, retrieval, latency, and cost for one set of rows."""
    native_latency = summarize_latencies(_numbers(rows, "native_latency_s"))
    agentic_latency = summarize_latencies(_numbers(rows, "agentic_latency_s"))
    return [
        f"{heading} Answer quality",
        "",
        _quality_table(rows),
        "",
        f"{heading} Retrieval",
        "",
        f"Native supporting-passage hit rate: {_rate(_hits(rows, 'native_retrieval_hit'))}",
        f"Agentic supporting-passage hit rate: {_rate(_hits(rows, 'agentic_retrieval_hit'))}",
        "",
        f"{heading} Latency (seconds)",
        "",
        _latency_line("Native", native_latency),
        _latency_line("Agentic", agentic_latency),
        "",
        f"{heading} Tokens and estimated cost",
        "",
        _usage_line("Native", rows, "native_usage"),
        _usage_line("Agentic", rows, "agentic_usage"),
        _usage_line("Judge", rows, "judge_usage"),
        "",
    ]


def _repeat_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("repeat") or 1), []).append(row)
    return [grouped[key] for key in sorted(grouped)]


def _average_sections(groups: list[list[dict[str, Any]]]) -> list[str]:
    """Mean of the per-repeat summaries."""
    present = [group for group in groups if group]
    if not present:
        return ["No completed repeats.", ""]
    return [
        "### Answer quality",
        "",
        _averaged_quality_table(present),
        "",
        "### Retrieval",
        "",
        f"Native supporting-passage hit rate: {_mean_rate(present, 'native_retrieval_hit')}",
        f"Agentic supporting-passage hit rate: {_mean_rate(present, 'agentic_retrieval_hit')}",
        "",
        "### Latency (seconds)",
        "",
        _latency_line("Native", _mean_latency(present, "native_latency_s")),
        _latency_line("Agentic", _mean_latency(present, "agentic_latency_s")),
        "",
        "### Tokens and estimated cost",
        "",
        _mean_usage_line("Native", present, "native_usage"),
        _mean_usage_line("Agentic", present, "agentic_usage"),
        _mean_usage_line("Judge", present, "judge_usage"),
        "",
    ]


def _averaged_quality_table(groups: list[list[dict[str, Any]]]) -> str:
    """Win rates averaged across repeats, then one block per category."""
    header = "| | " + " | ".join(CRITERIA) + " |"
    separator = "| --- | " + " | ".join("---" for _ in CRITERIA) + " |"
    lines = [header, separator]
    labels = ["Overall"]
    labels.extend(sorted({str(row["category"]) for group in groups for row in group if row.get("category")}))
    for label in labels:
        subsets = []
        for group in groups:
            subset = group if label == "Overall" else [row for row in group if row.get("category") == label]
            if subset:
                subsets.append(subset)
        if not subsets:
            continue
        heading = f"{label} ({_count_label([len(subset) for subset in subsets])})"
        lines.append("| **" + _cell(heading) + "** | " + " | ".join("" for _ in CRITERIA) + " |")
        rates = [criterion_rates(subset) for subset in subsets]
        for outcome in ("agentic", "native", "tie"):
            values = [_percent(_mean([item[name][outcome] for item in rates])) for name in CRITERIA]
            lines.append("| " + outcome + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _count_label(sizes: list[int]) -> str:
    if not sizes:
        return "0"
    if len(set(sizes)) == 1:
        return str(sizes[0])
    return f"avg {sum(sizes) / len(sizes):.1f}"


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_rate(groups: list[list[dict[str, Any]]], key: str) -> str:
    rates: list[float] = []
    for group in groups:
        values = _hits(group, key)
        if values:
            rates.append(sum(1 for value in values if value) / len(values))
    if not rates:
        return "not available"
    return _percent(_mean(rates))


def _mean_latency(groups: list[list[dict[str, Any]]], key: str) -> dict[str, float | None]:
    summaries = [summarize_latencies(_numbers(group, key)) for group in groups]
    averaged: dict[str, float | None] = {}
    for name in ("average", "p50", "p95"):
        values = [summary[name] for summary in summaries if isinstance(summary[name], (int, float))]
        averaged[name] = _mean([float(value) for value in values]) if values else None
    return averaged


def _mean_usage_line(label: str, groups: list[list[dict[str, Any]]], key: str) -> str:
    totals = [_usage_totals(group, key) for group in groups]
    calls = _mean([item["calls"] for item in totals])
    input_tokens = _mean([item["input_tokens"] for item in totals])
    output_tokens = _mean([item["output_tokens"] for item in totals])
    estimated_values = [item["estimated"] for item in totals if item["estimated_known"]]
    actual_values = [item["actual"] for item in totals if item["actual_known"]]
    per_question_values = [
        item["estimated"] / item["count"]
        for item in totals
        if item["estimated_known"] and item["count"]
    ]
    estimated_text = f"{_mean(estimated_values):.6f}" if estimated_values else "not estimated"
    per_question = f"{_mean(per_question_values):.6f}" if per_question_values else "not estimated"
    actual_text = f"{_mean(actual_values):.6f}" if actual_values else "not reported"
    return (
        f"- {label}: calls {calls:.3f}, input tokens {input_tokens:.3f}, output tokens {output_tokens:.3f}, "
        f"total tokens {input_tokens + output_tokens:.3f}, estimated total {estimated_text}, "
        f"estimated per question {per_question}, actual {actual_text}"
    )


def _usage_totals(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | bool]:
    estimated = 0.0
    estimated_known = False
    actual = 0.0
    actual_known = False
    calls = 0
    input_tokens = 0
    output_tokens = 0
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
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated": estimated,
        "estimated_known": estimated_known,
        "actual": actual,
        "actual_known": actual_known,
        "count": len(rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "repeat",
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
                    "repeat": row.get("repeat") or 1,
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


def _quality_table(rows: list[dict[str, Any]]) -> str:
    """Win rates for the whole run, then one block per category."""
    header = "| | " + " | ".join(CRITERIA) + " |"
    separator = "| --- | " + " | ".join("---" for _ in CRITERIA) + " |"
    lines = [header, separator]
    groups: list[tuple[str, list[dict[str, Any]]]] = [("Overall", rows)]
    labels = sorted({str(row["category"]) for row in rows if row.get("category")})
    for label in labels:
        groups.append((label, [row for row in rows if row.get("category") == label]))
    for label, group in groups:
        heading = f"{label} ({len(group)})"
        lines.append("| **" + _cell(heading) + "** | " + " | ".join("" for _ in CRITERIA) + " |")
        rates = criterion_rates(group)
        for outcome in ("agentic", "native", "tie"):
            values = [_percent(rates[name][outcome]) for name in CRITERIA]
            lines.append("| " + outcome + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


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
