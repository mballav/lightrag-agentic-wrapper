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
The input is one retrieval subquery from a query plan. It is often worded "Retrieve the passages..." and it often names a file. Choose exactly one query mode for that subquery.

Words such as "passages", "citations", "source", and "file_path" mean the retrieved text must be kept. They do not select a mode. Every mode can return source text.

Mode guide:
* 'local': The subquery asks for an attribute or fact of a named entity: a person, company, court, report, product, date, or amount. Retrieve that entity and its directly connected attributes. Use this for a single named document when the fact should be attached to that entity.
* 'global': The subquery asks for a theme, trend, policy, or relationship chain that is not an attribute of one entity. Use this for how a process works, what a set of practices amounts to, or a pattern across the graph.
* 'hybrid': The subquery needs both a named entity and the relationship chain around it. Use this when the answer depends on who or what the entity is and on how it connects to several other facts. Hybrid uses the graph only.
* 'mix': The subquery needs the entity or relationship context and the original wording. Use this when the answer must follow, quote, or summarize a source passage and the graph alone may drop a sentence, figure, or condition. Mix is the right mode for "retrieve the passages describing..." a named entity or document.
* 'naive': The subquery has no usable entity or relationship, and the evidence is a raw text span: an exact phrase, a label, a table cell, or a short string the graph is unlikely to contain. Naive searches chunks only.

Rules:
- Reply with a JSON object only, no markdown.
- Schema: {"mode": "<local|global|hybrid|mix|naive>", "reason": "<short reason>"}
- Pick exactly one mode from that list.
- Do not treat a request for passages as naive.
- Choose hybrid when both the entity and its relationships matter and the original wording is not required.
- Choose mix when both the graph and the original wording matter.
- Choose global when the subquery is about a theme or relationship pattern rather than one entity's attribute.
- Choose naive only when the graph would not help.
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
                f"\nSubquery: {cleaned}",
                system_prompt=_SYSTEM_PROMPT,
                response_format={"type": "json_object"},
                model=self._llm.model_cheap,
                reasoning_effort="none",
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
