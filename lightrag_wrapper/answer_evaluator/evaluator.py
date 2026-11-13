"""Judge one LightRAG retrieval and decide whether to search again."""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import dataclass

from lightrag_wrapper.answer_evaluator.evaluator_prompt import SYSTEM_PROMPT
from lightrag_wrapper.exceptions import LLMServiceError
from lightrag_wrapper.llm.service import LLMService
from lightrag_wrapper.log import get_logger
from lightrag_wrapper.models import QueryMode, Reference

logger = get_logger(__name__)

_DECISIONS = frozenset({"SUFFICIENT", "RETRIEVE_MORE", "CANNOT_ANSWER"})
_MODE_RANK = {
    QueryMode.NAIVE: 1,
    QueryMode.LOCAL: 2,
    QueryMode.HYBRID: 3,
    QueryMode.MIX: 4,
}
_CLAIM_STATUSES = frozenset({"SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"})
_CONFLICTS = frozenset({"resolved", "unresolved"})
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

    mode: str
    outcome: str


@dataclass(frozen=True)
class ClaimAssessment:
    """One material claim and how this call's evidence treats it."""

    claim: str
    status: str
    evidence_ids: tuple[str, ...]
    conflict: str | None


@dataclass(frozen=True)
class Evaluation:
    """A usable evaluator decision."""

    decision: str
    next_mode: QueryMode | None
    missing_information: list[str]
    claims: tuple[ClaimAssessment, ...]


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
        previous_mode: QueryMode,
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
            previous_mode=previous_mode,
        )
        try:
            raw = await self._llm.complete_prompt(
                prompt,
                system_prompt=SYSTEM_PROMPT,
                response_format={"type": "json_object"},
                model=self._llm.model_cheap,
                reasoning_effort="none",
            )
        except LLMServiceError as exc:
            logger.warning("Retrieval evaluator LLM call failed", error=str(exc))
            return None

        used_modes = {previous_mode.value, *(attempt.mode for attempt in attempts)}
        evaluation = parse_evaluation(
            raw if isinstance(raw, str) else str(raw),
            used_modes=used_modes,
        )
        if evaluation is None:
            logger.warning("Retrieval evaluator returned an unusable decision", payload=raw)
            return None
        logger.info(
            "Retrieval evaluator decided",
            decision=evaluation.decision,
            subquery=subquery[:120],
        )
        return evaluation


def parse_evaluation(
    raw: str,
    *,
    used_modes: Collection[str] | None = None,
) -> Evaluation | None:
    """Parse one evaluator JSON object. Return None when it cannot be followed.

    The decision is the one the claims require. A covered subquery is
    ``SUFFICIENT``. An uncovered subquery cannot stay ``SUFFICIENT``.
    """
    payload = _extract_json_object(raw)
    if payload is None:
        return None

    stated = _token(payload.get("decision"))
    if stated not in _DECISIONS:
        return None
    action = payload.get("next_action")
    if not isinstance(action, dict) or _token(action.get("type")) != stated:
        return None
    claims = _claims(payload.get("claims"))
    if claims is None:
        return None

    missing = _missing_information(payload.get("missing_information"))
    mode = _retry_mode(action.get("mode"))
    decided = _decision_from_claims(
        stated,
        claims,
        mode=mode,
        used_modes=frozenset(used_modes or ()),
    )
    if decided is None:
        return None
    decision, next_mode = decided
    return Evaluation(
        decision=decision,
        next_mode=next_mode,
        missing_information=missing,
        claims=claims,
    )


def _render_prompt(
    *,
    subquery: str,
    context: str,
    references: list[Reference],
    parents: list[UpstreamOutput],
    attempts: list[RetrievalAttempt],
    previous_mode: QueryMode,
) -> str:
    return (
        f"SUBQUERY:\n{subquery}\n\n"
        f"RETRIEVED EVIDENCE:\n{_format_evidence(references, context)}\n\n"
        f"PREVIOUS RETRIEVAL MODE:\n{previous_mode}\n\n"
        f"PARENT OUTPUTS:\n{_format_parents(parents)}\n\n"
        f"PREVIOUS ATTEMPTS:\n{_format_attempts(attempts)}\n\n"
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
            f'{index}. Outcome: {attempt.outcome}, Mode: {attempt.mode}'
        )
    return "\n".join(lines)


def _claims(value: object) -> tuple[ClaimAssessment, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    claims: list[ClaimAssessment] = []
    for item in value:
        claim = _claim(item)
        if claim is None:
            return None
        claims.append(claim)
    return tuple(claims)


def _claim(value: object) -> ClaimAssessment | None:
    if not isinstance(value, dict):
        return None
    text = value.get("claim")
    if not isinstance(text, str) or not text.strip():
        return None
    status = _token(value.get("status"))
    if status not in _CLAIM_STATUSES:
        return None
    evidence_ids = _evidence_ids(value.get("evidence_ids"))
    if evidence_ids is None:
        return None
    conflict = _conflict(value.get("conflict"))
    if status == "CONTRADICTED" and conflict is None:
        conflict = "unresolved"
    if status != "CONTRADICTED":
        conflict = None
    return ClaimAssessment(
        claim=text.strip(),
        status=status,
        evidence_ids=evidence_ids,
        conflict=conflict,
    )


def _evidence_ids(value: object) -> tuple[str, ...] | None:
    if value is None:
        return ()
    if not isinstance(value, list):
        return None
    ids: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        ids.append(item.strip())
    return tuple(ids)


def _conflict(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    conflict = value.strip().lower()
    if conflict not in _CONFLICTS:
        return None
    return conflict


def _claim_is_covered(claim: ClaimAssessment) -> bool:
    if claim.status == "SUPPORTED" and claim.evidence_ids:
        return True
    return claim.status == "CONTRADICTED" and claim.conflict == "resolved"


def _decision_from_claims(
    stated: str,
    claims: tuple[ClaimAssessment, ...],
    *,
    mode: QueryMode | None,
    used_modes: frozenset[str],
) -> tuple[str, QueryMode | None] | None:
    """Return the decision the claims allow, or None when a retry cannot be followed."""
    if all(_claim_is_covered(claim) for claim in claims):
        return "SUFFICIENT", None
    if stated != "RETRIEVE_MORE":
        return "CANNOT_ANSWER", None
    if mode is None:
        return None
    if mode.value in used_modes or _lower_than_a_used_mode(mode, used_modes):
        return "CANNOT_ANSWER", None
    return "RETRIEVE_MORE", mode


def _lower_than_a_used_mode(mode: QueryMode, used_modes: frozenset[str]) -> bool:
    """A lower ladder mode after a higher one is not another search. ``global`` is off the ladder."""
    rank = _MODE_RANK.get(mode)
    if rank is None:
        return False
    for used in used_modes:
        try:
            used_mode = QueryMode(used)
        except ValueError:
            continue
        used_rank = _MODE_RANK.get(used_mode)
        if used_rank is not None and used_rank > rank:
            return True
    return False


def _missing_information(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


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
