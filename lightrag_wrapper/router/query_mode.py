"""LLM-backed router that selects a LightRAG query mode from the user question."""

from __future__ import annotations

import json
import re
from typing import Any

from lightrag_wrapper.exceptions import LLMServiceError
from lightrag_wrapper.log import get_logger
from lightrag_wrapper.llm.service import LLMService
from lightrag_wrapper.models import QueryMode

logger = get_logger(__name__)

# Modes the router may choose. ``bypass`` is intentionally excluded.
_ROUTABLE_MODES = frozenset(
    {
        QueryMode.LOCAL,
        QueryMode.GLOBAL,
        QueryMode.HYBRID,
        QueryMode.NAIVE,
        QueryMode.MIX,
    }
)

_DEFAULT_MODE = QueryMode.MIX

_SYSTEM_PROMPT = """\
You are a retrieval-mode router for LightRAG, a graph-based RAG system.
Given a user question, choose exactly one query mode.

Mode guide:
- local: specific factual lookup — concrete facts, attributes, or details about a particular entity or item (entity-centric / context-dependent retrieval).
- global: broad thematic question — themes, trends, overviews, or high-level summaries across the knowledge base (relationship-centric / global knowledge).
- hybrid: entity relationship question — how entities connect, interact, cause, or relate to each other (combines local and global retrieval).
- mix: comprehensive question — needs both knowledge-graph structure and vector text evidence for a complete answer.
- naive: simple text lookup — straightforward keyword / passage search without needing graph reasoning.

Rules:
- Reply with a JSON object only, no markdown.
- Schema: {"mode": "<local|global|hybrid|mix|naive>", "reason": "<short reason>"}
- Pick exactly one mode from that list.
- Prefer the most specific matching mode; use mix when the question clearly needs both graph and text evidence.
"""


class QueryModeRouter:
    """Decide a LightRAG ``QueryMode`` from the user query via an LLM call."""

    def __init__(
        self,
        llm: LLMService,
        *,
        default_mode: QueryMode = _DEFAULT_MODE,
    ) -> None:
        self._llm = llm
        self._default_mode = default_mode

    async def route(self, query: str) -> QueryMode:
        """Return the best LightRAG mode for ``query``."""
        cleaned = query.strip()
        if not cleaned:
            return self._default_mode

        try:
            raw = await self._llm.complete_prompt(
                cleaned,
                system_prompt=_SYSTEM_PROMPT,
                response_format={"type": "json_object"},
            )
        except LLMServiceError as exc:
            logger.warning(
                "Query mode routing LLM call failed; using the default mode",
                error=str(exc),
                default_mode=self._default_mode.value,
            )
            return self._default_mode

        mode = self._parse_mode(raw if isinstance(raw, str) else str(raw))
        if mode is None:
            logger.warning(
                "Query mode router returned an unusable payload; using the default mode",
                payload=raw,
                default_mode=self._default_mode.value,
            )
            return self._default_mode

        logger.info(
            "Query mode router selected a mode",
            mode=mode.value,
            query=cleaned[:120],
        )
        return mode

    def _parse_mode(self, raw: str) -> QueryMode | None:
        payload = _extract_json_object(raw)
        if payload is None:
            # Fall back to a bare mode token if the model ignored JSON.
            token = raw.strip().lower().strip("\"'`")
            for candidate in _ROUTABLE_MODES:
                if token == candidate.value:
                    return candidate
            return None

        value = payload.get("mode")
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        try:
            mode = QueryMode(normalized)
        except ValueError:
            return None
        if mode not in _ROUTABLE_MODES:
            return None
        return mode


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
