"""Judge one LightRAG retrieval and decide whether to search again."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from lightrag_wrapper.answer_evaluator.evaluator_prompt import SYSTEM_PROMPT
from lightrag_wrapper.exceptions import LLMServiceError
from lightrag_wrapper.llm.service import LLMService
from lightrag_wrapper.log import get_logger
from lightrag_wrapper.models import QueryMode, Reference

logger = get_logger(__name__)

_DECISIONS = frozenset({"SUFFICIENT", "RETRIEVE_MORE", "CANNOT_ANSWER"})
_RETRY_MODES = frozenset(
    {
        QueryMode.NAIVE,
        QueryMode.LOCAL,
        QueryMode.GLOBAL,
        QueryMode.HYBRID,
        QueryMode.MIX,
    }
)


@dataclass(frozen=True)
class UpstreamOutput:
    """A parent node's output. It is context for the subquery, not evidence."""

    node_id: int
    node_type: str
    output: str


@dataclass(frozen=True)
class RetrievalAttempt:
    """One earlier search for this same node."""

    query: str
    mode: str
    outcome: str


@dataclass(frozen=True)
class Evaluation:
    """A usable evaluator decision. ``next_query`` is set only for another search."""

    decision: str
    next_query: str | None
    next_mode: QueryMode | None
    missing_information: list[str]


class AnswerEvaluator:
    """Ask the LLM whether the context retrieved for one node is enough."""

    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    async def evaluate(
        self,
        *,
        subquery: str,
        context: str,
        references: list[Reference] | None,
        parents: list[UpstreamOutput],
        attempts: list[RetrievalAttempt],
    ) -> Evaluation | None:
        """Return a decision, or None when the model output cannot be followed."""
        prompt = _render_prompt(
            subquery=subquery,
            context=context,
            references=references or [],
            parents=parents,
            attempts=attempts,
        )
        try:
            raw = await self._llm.complete_prompt(
                prompt,
                system_prompt=SYSTEM_PROMPT,
                response_format={"type": "json_object"},
            )
        except LLMServiceError as exc:
            logger.warning("Retrieval evaluator LLM call failed", error=str(exc))
            return None

        evaluation = parse_evaluation(raw if isinstance(raw, str) else str(raw))
        if evaluation is None:
            logger.warning("Retrieval evaluator returned an unusable decision", payload=raw)
            return None
        logger.info(
            "Retrieval evaluator decided",
            decision=evaluation.decision,
            subquery=subquery[:120],
        )
        return evaluation


def parse_evaluation(raw: str) -> Evaluation | None:
    """Parse one evaluator JSON object. Return None when it cannot be followed."""
    payload = _extract_json_object(raw)
    if payload is None:
        return None

    decision = _token(payload.get("decision"))
    if decision not in _DECISIONS:
        return None
    action = payload.get("next_action")
    if not isinstance(action, dict) or _token(action.get("type")) != decision:
        return None

    missing = _missing_information(payload.get("missing_information"))
    if decision != "RETRIEVE_MORE":
        return Evaluation(
            decision=decision,
            next_query=None,
            next_mode=None,
            missing_information=missing,
        )

    query = _search_query(action.get("query"))
    mode = _retry_mode(action.get("mode"))
    if query is None or mode is None:
        return None
    return Evaluation(
        decision=decision,
        next_query=query,
        next_mode=mode,
        missing_information=missing,
    )


def _render_prompt(
    *,
    subquery: str,
    context: str,
    references: list[Reference],
    parents: list[UpstreamOutput],
    attempts: list[RetrievalAttempt],
) -> str:
    return (
        f"SUBQUERY:\n{subquery}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        f"RETRIEVED EVIDENCE:\n{_format_evidence(references, context)}\n\n"
        f"PARENT OUTPUTS:\n{_format_parents(parents)}\n\n"
        f"PREVIOUS ATTEMPTS:\n{_format_attempts(attempts)}"
    )


def _format_evidence(references: list[Reference], context: str) -> str:
    items: list[str] = []
    for reference in references:
        source = reference.file_path or reference.reference_id or "retrieved context"
        for chunk in reference.content or []:
            text = chunk.strip()
            if not text:
                continue
            items.append(f'[E{len(items) + 1}] Source: {source}. Content: "{text}"')
    if items:
        return "\n".join(items)
    return f'[E1] Source: retrieved context. Content: "{context}"'


def _format_parents(parents: list[UpstreamOutput]) -> str:
    if not parents:
        return "None."
    return "\n\n".join(
        f"[Upstream node {parent.node_id}, {parent.node_type}] {parent.output}"
        for parent in parents
    )


def _format_attempts(attempts: list[RetrievalAttempt]) -> str:
    if not attempts:
        return "None."
    lines: list[str] = []
    for index, attempt in enumerate(attempts, start=1):
        lines.append(
            f'{index}. Query: "{attempt.query}" Mode: {attempt.mode}. Outcome: {attempt.outcome}'
        )
    return "\n".join(lines)


def _missing_information(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _search_query(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    query = value.strip().strip('"')
    return query or None


def _retry_mode(value: object) -> QueryMode | None:
    if not isinstance(value, str):
        return None
    try:
        mode = QueryMode(value.strip().lower())
    except ValueError:
        return None
    if mode not in _RETRY_MODES:
        return None
    return mode


def _token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip().upper() or None


def _extract_json_object(text: str) -> dict | None:
    cleaned = text.strip()
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
