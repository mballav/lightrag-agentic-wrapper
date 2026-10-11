"""LLM-backed decomposer that turns a user query into a retrieval DAG."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from lightrag_wrapper.adapter._catalog import unavailable_catalog
from lightrag_wrapper.adapter.base import LightRAGAdapter
from lightrag_wrapper.exceptions import LightRAGAdapterError, LLMServiceError
from lightrag_wrapper.log import get_logger
from lightrag_wrapper.llm.service import LLMService
from lightrag_wrapper.models import QueryPlanNode
from lightrag_wrapper.query_decomposer.decomposer_prompt import query_decomposer_prompt

logger = get_logger(__name__)

_NODE_TYPES = frozenset(
    {
        "retrieve",
        "filter",
        "transform",
        "aggregate",
        "compare",
        "assemble",
        "evaluate",
        "synthesize",
    }
)

_SYSTEM_PROMPT = (
    "You are Query Decomposer. Follow the instructions in the user message "
    "exactly and return only the JSON array of DAG nodes."
)


@dataclass(frozen=True)
class Decomposition:
    """Parsed DAG plus the catalog text that was shown to the model."""

    nodes: list[QueryPlanNode]
    knowledge_base_metadata: str


class QueryDecomposer:
    """Break a user query into a dependency-aware retrieval plan."""

    def __init__(self, llm: LLMService, adapter: LightRAGAdapter) -> None:
        self._llm = llm
        self._adapter = adapter

    async def decompose(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> Decomposition | None:
        """Return a DAG for ``query``, or None when the model output is unusable."""
        cleaned = query.strip()
        if not cleaned:
            return None

        metadata = await self._knowledge_base_metadata()
        prompt = render_decomposer_prompt(
            cleaned,
            conversation_history,
            metadata,
        )
        try:
            raw = await self._llm.complete_prompt(
                prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0,
            )
        except LLMServiceError as exc:
            logger.warning("Query decomposer LLM call failed", error=str(exc))
            return None

        nodes = parse_decomposition(raw if isinstance(raw, str) else str(raw))
        if nodes is None:
            logger.warning("Query decomposer returned an unusable plan", payload=raw)
            return None

        logger.info(
            "Query decomposer produced a plan",
            node_count=len(nodes),
            query=cleaned[:120],
        )
        return Decomposition(nodes=nodes, knowledge_base_metadata=metadata)

    async def _knowledge_base_metadata(self) -> str:
        try:
            catalog = await self._adapter.list_document_catalog()
        except LightRAGAdapterError as exc:
            logger.warning("Document catalog unavailable", error=str(exc))
            catalog = unavailable_catalog(str(exc))
        return json.dumps(catalog, indent=2, ensure_ascii=False)


def render_decomposer_prompt(
    user_query: str,
    previous_messages: list[dict[str, str]] | None,
    knowledge_base_metadata: str,
) -> str:
    """Fill the decomposer template. Placeholders are replaced literally."""
    history = json.dumps(previous_messages or [], indent=2, ensure_ascii=False)
    return (
        query_decomposer_prompt.replace("{previous_messages}", history)
        .replace("{knowledge_base_metadata}", knowledge_base_metadata)
        .replace("{user_query}", user_query)
    )


def initial_retrieve_nodes(nodes: list[QueryPlanNode]) -> list[QueryPlanNode]:
    """Retrieve nodes that do not need another node's output."""
    return [
        node
        for node in nodes
        if node.node_type == "retrieve" and not node.depends_on
    ]


def parse_decomposition(raw: str) -> list[QueryPlanNode] | None:
    """Parse a JSON array of DAG nodes. Returns None when the plan is invalid."""
    payload = _extract_json_array(raw)
    if payload is None:
        return None

    nodes: list[QueryPlanNode] = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        try:
            node = QueryPlanNode.model_validate(item)
        except ValueError:
            return None
        if node.node_type not in _NODE_TYPES or not node.subquery.strip():
            return None
        nodes.append(node)

    if not nodes:
        return None
    if not _dependencies_are_valid(nodes) or not _plan_structure_is_valid(nodes):
        return None
    return nodes


def _plan_structure_is_valid(nodes: list[QueryPlanNode]) -> bool:
    """Require a result, result leaves, and dependencies on every non-retrieve node."""
    if not any(node.is_result_node for node in nodes):
        return False
    depended_on = {dependency for node in nodes for dependency in node.depends_on}
    for node in nodes:
        if node.id not in depended_on and not node.is_result_node:
            return False
        if node.node_type != "retrieve" and not node.depends_on:
            return False
    return True


def _dependencies_are_valid(nodes: list[QueryPlanNode]) -> bool:
    ids = [node.id for node in nodes]
    if len(ids) != len(set(ids)):
        return False
    known = set(ids)
    graph = {node.id: list(node.depends_on) for node in nodes}
    for dependencies in graph.values():
        if any(dependency not in known for dependency in dependencies):
            return False

    visiting: set[int] = set()
    visited: set[int] = set()

    def walk(node_id: int) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for dependency in graph[node_id]:
            if walk(dependency):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return not any(walk(node_id) for node_id in ids)


def _extract_json_array(text: str) -> list | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None
