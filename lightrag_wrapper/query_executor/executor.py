"""Execute a decomposer plan as a LangGraph DAG."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from lightrag_wrapper.adapter.base import LightRAGAdapter
from lightrag_wrapper.llm.service import LLMService
from lightrag_wrapper.log import get_logger
from lightrag_wrapper.models import QueryMode, QueryPlanNode, Reference
from lightrag_wrapper.router.query_mode import QueryModeRouter

logger = get_logger(__name__)

_RESOLVE_SYSTEM_PROMPT = (
    "You rewrite a dependent retrieval subquery into one standalone search string. "
    "Use only the parent outputs. Do not add facts that are not in those outputs. "
    "Return only the search string."
)

_REASON_SYSTEM_PROMPT = (
    "You perform one operation in a retrieval plan. "
    "Use only the parent outputs and the user query. Do not retrieve and do not invent sources. "
    "Return only the operation result."
)


@dataclass
class NodeArtifact:
    """Output of one plan node, stored for its dependents."""

    node_id: int
    node_type: str
    subquery: str
    output: str
    references: list[Reference]
    is_result_node: bool
    mode: QueryMode | None = None
    response_time: float | None = None


def _merge_artifacts(
    left: dict[int, NodeArtifact] | None,
    right: dict[int, NodeArtifact] | None,
) -> dict[int, NodeArtifact]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class PlanState(TypedDict):
    """Shared graph state. Parallel nodes merge ``artifacts`` by node id."""

    artifacts: Annotated[dict[int, NodeArtifact], _merge_artifacts]


@dataclass
class _CallContext:
    user_query: str
    conversation_history: list[dict[str, str]] | None = None
    high_level_keywords: list[str] | None = None
    low_level_keywords: list[str] | None = None


class PlanExecutor:
    """Run a parsed retrieval DAG. Only ``retrieve`` nodes call LightRAG."""

    def __init__(
        self,
        llm: LLMService,
        adapter: LightRAGAdapter,
        router: QueryModeRouter,
    ) -> None:
        self._llm = llm
        self._adapter = adapter
        self._router = router

    async def stream(
        self,
        nodes: list[QueryPlanNode],
        *,
        user_query: str,
        conversation_history: list[dict[str, str]] | None = None,
        high_level_keywords: list[str] | None = None,
        low_level_keywords: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        """Yield each result node as soon as it finishes. Do not sort or join them."""
        context = _CallContext(
            user_query=user_query,
            conversation_history=conversation_history,
            high_level_keywords=high_level_keywords,
            low_level_keywords=low_level_keywords,
        )
        compiled = self._compile(nodes, context)
        async for update in compiled.astream({"artifacts": {}}, stream_mode="updates"):
            for payload in _result_payloads(update):
                yield payload
        logger.info(
            "Plan execution finished",
            node_count=len(nodes),
            result_count=_result_count(nodes),
        )

    def _compile(self, nodes: list[QueryPlanNode], context: _CallContext):
        depended_on = {dependency for node in nodes for dependency in node.depends_on}
        graph = StateGraph(PlanState)
        for node in nodes:
            graph.add_node(str(node.id), self._runner(node, context))
        for node in nodes:
            dependencies = _dependency_ids(node)
            if not dependencies:
                graph.add_edge(START, str(node.id))
            else:
                for dependency in dependencies:
                    graph.add_edge(str(dependency), str(node.id))
            if node.id not in depended_on:
                graph.add_edge(str(node.id), END)
        return graph.compile()

    def _runner(self, node: QueryPlanNode, context: _CallContext):
        async def run(state: PlanState) -> dict[str, dict[int, NodeArtifact]]:
            artifact = await self._run_node(node, state["artifacts"], context)
            return {"artifacts": {node.id: artifact}}

        return run

    async def _run_node(
        self,
        node: QueryPlanNode,
        artifacts: dict[int, NodeArtifact],
        context: _CallContext,
    ) -> NodeArtifact:
        parents = [_require_parent(artifacts, dependency) for dependency in _dependency_ids(node)]
        if node.node_type == "retrieve":
            return await self._retrieve(node, parents, context)
        return await self._reason(node, parents, context)

    async def _retrieve(
        self,
        node: QueryPlanNode,
        parents: list[NodeArtifact],
        context: _CallContext,
    ) -> NodeArtifact:
        search = node.subquery
        if parents:
            search = await self._resolve_search(node, parents)
        mode = await self._router.route(search)
        keywords: dict[str, list[str] | None] = {
            "high_level_keywords": None,
            "low_level_keywords": None,
        }
        if not parents:
            keywords = {
                "high_level_keywords": context.high_level_keywords,
                "low_level_keywords": context.low_level_keywords,
            }
        result = await self._adapter.query(
            search,
            mode=mode,
            only_need_context=not node.is_result_node,
            user_prompt=None,
            conversation_history=context.conversation_history,
            include_references=True,
            **keywords,
        )
        return NodeArtifact(
            node_id=node.id,
            node_type=node.node_type,
            subquery=node.subquery,
            output=result.response,
            references=_union_references([parent.references for parent in parents] + [result.references or []]),
            is_result_node=node.is_result_node,
            mode=result.mode,
            response_time=result.response_time,
        )

    async def _resolve_search(self, node: QueryPlanNode, parents: list[NodeArtifact]) -> str:
        prompt = (
            f"Subquery:\n{node.subquery}\n\n"
            f"Parent outputs:\n{_format_parents(parents)}"
        )
        raw = await self._llm.complete_prompt(
            prompt,
            system_prompt=_RESOLVE_SYSTEM_PROMPT,
            temperature=0,
        )
        resolved = _as_text(raw).strip().strip('"')
        return resolved or node.subquery

    async def _reason(
        self,
        node: QueryPlanNode,
        parents: list[NodeArtifact],
        context: _CallContext,
    ) -> NodeArtifact:
        prompt = (
            f"User query:\n{context.user_query}\n\n"
            f"Operation ({node.node_type}):\n{node.subquery}\n\n"
            f"Parent outputs:\n{_format_parents(parents)}"
        )
        raw = await self._llm.complete_prompt(
            prompt,
            system_prompt=_REASON_SYSTEM_PROMPT,
            temperature=0,
        )
        return NodeArtifact(
            node_id=node.id,
            node_type=node.node_type,
            subquery=node.subquery,
            output=_as_text(raw).strip(),
            references=_union_references([parent.references for parent in parents]),
            is_result_node=node.is_result_node,
        )


def _dependency_ids(node: QueryPlanNode) -> list[int]:
    seen: list[int] = []
    for dependency in node.depends_on:
        if dependency not in seen:
            seen.append(dependency)
    return seen


def _require_parent(artifacts: dict[int, NodeArtifact], dependency: int) -> NodeArtifact:
    try:
        return artifacts[int(dependency)]
    except KeyError as exc:
        raise RuntimeError(f"Plan node is missing the output of node {dependency}") from exc


def _format_parents(parents: list[NodeArtifact]) -> str:
    blocks: list[str] = []
    for parent in parents:
        citations = "; ".join(_format_citation(ref) for ref in parent.references) or "none"
        blocks.append(
            f"Node {parent.node_id} ({parent.node_type})\n"
            f"Subquery: {parent.subquery}\n"
            f"Output:\n{parent.output}\n"
            f"Citations: {citations}"
        )
    return "\n\n".join(blocks)


def _format_citation(ref: Reference) -> str:
    source = ref.file_path or ref.reference_id or "unknown"
    if not ref.content:
        return source
    return f"{source}: {' '.join(ref.content)}"


def _union_references(groups: list[list[Reference]]) -> list[Reference]:
    seen: set[tuple[str | None, str | None, tuple[str, ...]]] = set()
    merged: list[Reference] = []
    for group in groups:
        for ref in group:
            key = (ref.reference_id, ref.file_path, tuple(ref.content or []))
            if key in seen:
                continue
            seen.add(key)
            merged.append(ref)
    return merged


def _result_count(nodes: list[QueryPlanNode]) -> int:
    return sum(1 for node in nodes if node.is_result_node)


def _result_payloads(update: object) -> list[dict]:
    if not isinstance(update, dict):
        return []
    payloads: list[dict] = []
    for node_update in update.values():
        if not isinstance(node_update, dict):
            continue
        for artifact in (node_update.get("artifacts") or {}).values():
            if isinstance(artifact, NodeArtifact) and artifact.is_result_node:
                payloads.append(
                    {
                        "id": artifact.node_id,
                        "response": artifact.output,
                        "references": [ref.model_dump() for ref in artifact.references],
                    }
                )
    return payloads


def _as_text(raw: object) -> str:
    return raw if isinstance(raw, str) else str(raw)
