"""LLM-backed rewriter that turns a user turn into a retrieval query and keywords."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from lightrag_wrapper.exceptions import LLMServiceError
from lightrag_wrapper.llm.service import LLMService
from lightrag_wrapper.query_rewriter.rewrite_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# LightRAG /query rejects longer lists and longer individual keywords.
_MAX_KEYWORDS = 64
_MAX_KEYWORD_CHARS = 512


@dataclass(frozen=True)
class RewrittenQuery:
    """JSON payload produced by the query rewriter."""

    rewritten_query: str
    high_level_keywords: list[str] = field(default_factory=list)
    low_level_keywords: list[str] = field(default_factory=list)
    is_context_dependent: bool = False
    ambiguities: list[str] = field(default_factory=list)
    confidence: float = 0.0


class QueryRewriter:
    """Rewrite a user query and extract retrieval keywords via an LLM call."""

    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    async def rewrite(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> RewrittenQuery:
        """Return a retrieval query and keyword lists for ``query``."""
        cleaned = query.strip()
        if not cleaned:
            return _fallback(query)

        user_prompt = _render_input(cleaned, conversation_history)
        try:
            raw = await self._llm.complete_prompt(
                user_prompt,
                system_prompt=SYSTEM_PROMPT,
                response_format={"type": "json_object"},
                temperature=0,
            )
        except LLMServiceError as exc:
            logger.warning(
                "Query rewriter LLM call failed (%s); using the original query",
                exc,
            )
            return _fallback(cleaned)

        parsed = _parse_rewrite(raw if isinstance(raw, str) else str(raw), cleaned)
        if parsed is None:
            logger.warning(
                "Query rewriter returned unusable payload %r; using the original query",
                raw,
            )
            return _fallback(cleaned)

        logger.info(
            "Query rewriter produced %r (hl=%s ll=%s)",
            parsed.rewritten_query[:120],
            parsed.high_level_keywords,
            parsed.low_level_keywords,
        )
        return parsed


def _render_input(
    query: str,
    conversation_history: list[dict[str, str]] | None,
) -> str:
    """Inject the current query and conversation history into the user prompt."""
    if conversation_history:
        lines: list[str] = []
        for message in conversation_history:
            role = str(message.get("role") or "user").strip() or "user"
            content = str(message.get("content") or "").strip()
            lines.append(f"{role}: {content}")
        history = "\n".join(lines) if lines else "(none)"
    else:
        history = "(none)"
    return f"Conversation history:\n{history}\n\nCurrent user query:\n{query}"


def _fallback(query: str) -> RewrittenQuery:
    return RewrittenQuery(rewritten_query=query.strip())


def _parse_rewrite(raw: str, original_query: str) -> RewrittenQuery | None:
    payload = _extract_json_object(raw)
    if payload is None:
        return None

    rewritten = payload.get("rewritten_query")
    if not isinstance(rewritten, str) or not rewritten.strip():
        rewritten_query = original_query
    else:
        rewritten_query = rewritten.strip()

    return RewrittenQuery(
        rewritten_query=rewritten_query,
        high_level_keywords=_keyword_list(payload.get("high_level_keywords")),
        low_level_keywords=_keyword_list(payload.get("low_level_keywords")),
        is_context_dependent=bool(payload.get("is_context_dependent")),
        ambiguities=_string_list(payload.get("ambiguities")),
        confidence=_confidence(payload.get("confidence")),
    )


def _keyword_list(value: Any) -> list[str]:
    keywords = _string_list(value)
    trimmed = [item[:_MAX_KEYWORD_CHARS] for item in keywords if item]
    # Preserve order while dropping duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for item in trimmed:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:_MAX_KEYWORDS]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [part.strip() for part in re.split(r"[\n,;]+", value)]
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            items.append(text)
    return items


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
