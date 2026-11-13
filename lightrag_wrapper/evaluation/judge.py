"""Pairwise answer comparison using LightRAG's three criteria."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

CRITERIA = ("Comprehensiveness", "Diversity", "Empowerment", "Overall")

JUDGE_SYSTEM_PROMPT = """
You compare two answers to the same question. You do not know which system wrote either answer.

Score each criterion independently, then choose an overall winner.

Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, the winner is "Answer 1", "Answer 2", or "Tie".
Use "Tie" when the answers are equally strong on that criterion.

Return one JSON object and nothing else:

{
  "Comprehensiveness": {"Winner": "Answer 1 | Answer 2 | Tie", "Explanation": "..."},
  "Diversity": {"Winner": "Answer 1 | Answer 2 | Tie", "Explanation": "..."},
  "Empowerment": {"Winner": "Answer 1 | Answer 2 | Tie", "Explanation": "..."},
  "Overall": {"Winner": "Answer 1 | Answer 2 | Tie", "Explanation": "..."}
}
"""


@dataclass(frozen=True)
class CriterionJudgment:
    """One criterion in one presentation order."""

    winner: str
    explanation: str


@dataclass(frozen=True)
class Judgment:
    """One blinded comparison. ``agentic_first`` records which side was Answer 1."""

    agentic_first: bool
    criteria: dict[str, CriterionJudgment]


@dataclass(frozen=True)
class CombinedJudgment:
    """A criterion win counts only when both presentation orders agree."""

    winners: dict[str, str]
    judgments: tuple[Judgment, Judgment]


def render_judge_prompt(question: str, answer_1: str, answer_2: str) -> str:
    """User message for one presentation order."""
    return (
        f"Question:\n{question}\n\n"
        f"Answer 1:\n{answer_1}\n\n"
        f"Answer 2:\n{answer_2}"
    )


def parse_judgment(raw: str, *, agentic_first: bool) -> Judgment | None:
    """Parse one judge JSON object. Return None when a criterion is missing."""
    payload = _json_object(raw)
    if payload is None:
        return None
    criteria: dict[str, CriterionJudgment] = {}
    for name in CRITERIA:
        block = _block(payload, name)
        if block is None:
            return None
        winner = _winner(block.get("Winner"))
        if winner is None:
            return None
        explanation = block.get("Explanation")
        criteria[name] = CriterionJudgment(
            winner=winner,
            explanation=explanation.strip() if isinstance(explanation, str) else "",
        )
    return Judgment(agentic_first=agentic_first, criteria=criteria)


def combine_judgments(first: Judgment, second: Judgment) -> CombinedJudgment:
    """Map both orders onto native, agentic, or tie."""
    winners: dict[str, str] = {}
    for name in CRITERIA:
        left = _system(first.criteria[name].winner, agentic_first=first.agentic_first)
        right = _system(second.criteria[name].winner, agentic_first=second.agentic_first)
        winners[name] = left if left == right else "tie"
    return CombinedJudgment(winners=winners, judgments=(first, second))


def _system(winner: str, *, agentic_first: bool) -> str:
    if winner == "tie":
        return "tie"
    agentic_is_answer_1 = winner == "answer 1"
    if agentic_first:
        return "agentic" if agentic_is_answer_1 else "native"
    return "native" if agentic_is_answer_1 else "agentic"


def _winner(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    folded = value.strip().casefold()
    if folded in {"answer 1", "1"}:
        return "answer 1"
    if folded in {"answer 2", "2"}:
        return "answer 2"
    if folded == "tie":
        return "tie"
    return None


def _block(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    folded_name = name.casefold()
    aliases = {folded_name, f"{folded_name} winner"}
    for key, value in payload.items():
        folded = str(key).replace("_", " ").casefold().strip()
        if folded in aliases and isinstance(value, dict):
            return value
    return None


def _json_object(raw: str) -> dict[str, Any] | None:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None
