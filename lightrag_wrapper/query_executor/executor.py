"""Execute a decomposer plan as a LangGraph DAG."""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from lightrag_wrapper.adapter.base import LightRAGAdapter
from lightrag_wrapper.answer_evaluator.evaluator import (
    AnswerEvaluator,
    Evaluation,
    RetrievalAttempt,
    UpstreamOutput,
)
from lightrag_wrapper.llm.service import LLMService
from lightrag_wrapper.log import get_logger
from lightrag_wrapper.models import QueryMode, QueryPlanNode, QueryResult, Reference
from lightrag_wrapper.query_timing import (
    QueryTiming,
    clear_reported_llm_seconds,
    take_reported_llm_seconds,
    write_query_times,
)
from lightrag_wrapper.router.query_mode import QueryModeRouter
from lightrag_wrapper.tracing import langgraph_callbacks

logger = get_logger(__name__)

_RESOLVE_SYSTEM_PROMPT = """You rewrite a dependent retrieval subquery into one standalone search string and a grand search string.

A retriever runs the grand search string next. It does not see the entire execution plan.

The subquery asks for a fact that is still unknown. It points at something an earlier node already retrieved, such as "the person identified in Node 1". The parent outputs are retrieved passages. Use them only to learn what that reference names.

The grand search string is a search string that is more specific than the standalone search string. The grand search string is the current subquery combined with the parent subqueries and outputs. The grand search string is the question, not the answer.

Do this for the standalone search string:
- Take the name, title, or value in the parent outputs that the subquery's reference points to.
- Rewrite the subquery so it asks for the same unknown fact, with that reference replaced by the concrete value.
- Leave the requested fact open. The standalone search string is the question, not the answer.

Do this for the grand search string:
- Combine the current subquery with the parent subqueries and outputs.
- Leave the requested fact open. The grand search string is the question, not the answer.
- **Do not** leave out any details from the parent subqueries. The final grand search string should be a consolidated question.

For both the standalone search string and the grand search string, do not:
- Answer the subquery, or place the sought fact in the string, even when a parent passage states it or you know it.
- Copy passages, citation numbers, source titles, or extra parent facts into the string.
- Add a name, date, place, or conclusion that the parent outputs do not state.
- Mention nodes, the plan, "retrieve", "passages", "citations", or "sources".
- Return a keyword list.

If the parent outputs never identify the referenced thing, keep the subquery's own description of it. Do not invent a name.

Return JSON containing standalone search string and grand search string in the following format:
{
    "standalone_search": "In which year was Gene Kelly born?",
    "grand_search": "In which year was Gene Kelly who performed the ballet "Swan Lake" born?"
}

Examples:

Example 1:
Current Subquery: Retrieve the birth year of the choreographer identified in Node 1.
Parent subqueries: 
Node 1: Subquery: Retrieve the name of the choreographer who performed the ballet "Swan Lake".

Parent output identifies the choreographer as Gene Kelly.

Output:
{
    "standalone_search": "In which year was Gene Kelly born?",
    "grand_search": "In which year was Gene Kelly who performed the ballet "Swan Lake" born?"
}

Example 2:
Current Subquery: Retrieve a county that shares a border with the county containing the city identified in Node 2, in the state identified in Node 1.
Parent subqueries:
Node 1: Subquery: Retrieve the state where Washington State Prison is located.
Node 2: Subquery: Retrieve the most populous city in the state identified in Node 1 and the county containing that city.

Parent outputs identify the state as Georgia, the most populous city as Atlanta, and Atlanta's county as Fulton County. They do not name a bordering county.

Output:
{
    "standalone_search": "Which county in Georgia shares a border with Fulton County?",
    "grand_search": "Washington State Prison is located in Georgia, and Atlanta is the most populous city in Georgia. Atlanta is in Fulton County. Which county in Georgia shares a border with Fulton County?"
}
"""

_REASON_OPERATIONS = {
    "filter": "Keep only the parent evidence that meets the stated condition, and drop the rest.",
    "transform": "Rewrite the parent evidence into the structure the operation requests, without new facts.",
    "aggregate": (
        "Perform the requested count, sum, difference, range, or other calculation, and show the work."
    ),
    "compare": (
        "Set the parents side by side on the points the operation names. "
        "Analyze where they agree, where they differ, and where they conflict. "
        "Finally, merge them into one undifferentiated summary while answering the subquery presented in the current node with all the details."
    ),
    "assemble": (
        "Combine every parent into one artifact a later node can use without seeing those parents again. "
        "Do not drop a parent."
    ),
    "evaluate": (
        "Judge sufficiency, relevance, consistency, contradiction, or support from the parent outputs, "
        "and state the judgment together with the evidence it uses."
    ),
    "synthesize": "Write the answer the operation asks for from the parent outputs.",
}


def _reason_system_prompt(node_type: str) -> str:
    """Instructions for this operation only. The shared rules stay the same."""
    operation = _REASON_OPERATIONS.get(
        node_type,
        "Write the output the operation asks for from the parent outputs.",
    )
    return (
        "You execute exactly one operation in a retrieval plan. "
        f"The operation is {node_type}. "
        "Do not retrieve, search, or call a tool. Do not add, skip, or replace the operation.\n\n"
        "You may use only three inputs: the user query, the operation text, and the parent outputs. "
        "The parent outputs are the only evidence. "
        "Do not add outside knowledge, and do not invent numbers, names, dates, citations, or sources. "
        "Repeat a citation only when that parent lists it. "
        "Keep each fact with the parent it came from.\n\n"
        "If a parent output is empty or does not contain a fact the operation needs, "
        "say that the fact is not in the parent outputs and name what is missing. Do not guess it. "
        "When the operation requires arithmetic, use only numbers written in the parent outputs. "
        "Show the inputs, each step, and the result in the output, and state any assumption. "
        "If a required number is absent, do not produce a total.\n\n"
        f"Write this operation's output: {operation}\n\n"
        "When the node is a result, write the answer the user will read, in plain prose. "
        "Lead with the answer, then the supporting detail. Do not open with a label such as \"Here is\".\n"
        "When the node is not a result, write an intermediate artifact for the next node. "
        "Carry forward the facts, figures, constraints, and citations that later step needs. "
        "Do not address the user.\n\n"
        "The message content is the node output. Write the complete output there. "
        "Do not leave the message empty, and do not stop after planning the answer. "
        "Do not return a status, an apology, a question, or a request for more input. "
        "Do not wrap the output in code fences or in a JSON envelope unless the operation asks for that format. "
        "Return only the node output."
    )

_ANSWER_SYSTEM_PROMPT = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer the user query accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Those retrievals are the ones this plan already fetched and assembled. Do not search again.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information (from any outside sources).
  - Track the reference id of the document chunk which directly supports the facts presented in the response. Correlate that id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.
  - A note that an earlier step could not find a fact is part of the context. Keep that limit. Do not fill the gap.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in Multiple Paragraphs.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: 
   - Use only the knowledge graph data, retrieved passages, and assembled parent outputs in the Context. 
   - Do not add facts from outside those retrievals. 
   - You may add relevant and additional facts taken from the knowledge graph data, retrieved passages, and parent outputs if they will make the answer more complete and informative.
   - You **must** repeat phrases or information from the user query to make the answer more complete and informative.
   - If the context does not contain the information, you **must** state that you do not have enough information to answer. Please also justify logically why the answer cannot be deduced from the context. In such a case, if part of the question can be answered, make sure to include that answer as well.
"""


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
    timing: QueryTiming | None = None


class PlanExecutor:
    """Run a parsed retrieval DAG. Only ``retrieve`` nodes call LightRAG."""

    def __init__(
        self,
        llm: LLMService,
        adapter: LightRAGAdapter,
        router: QueryModeRouter,
        *,
        max_extra_retrievals: int = 2,
    ) -> None:
        self._llm = llm
        self._adapter = adapter
        self._router = router
        self._evaluator = AnswerEvaluator(llm)
        self._max_extra_retrievals = max(0, max_extra_retrievals)

    async def stream(
        self,
        nodes: list[QueryPlanNode],
        *,
        user_query: str,
        conversation_history: list[dict[str, str]] | None = None,
        high_level_keywords: list[str] | None = None,
        low_level_keywords: list[str] | None = None,
        timing: QueryTiming | None = None,
    ) -> AsyncIterator[dict]:
        """Yield each result node as soon as it finishes. Do not sort or join them."""
        context = _CallContext(
            user_query=user_query,
            conversation_history=conversation_history,
            high_level_keywords=high_level_keywords,
            low_level_keywords=low_level_keywords,
            timing=timing,
        )
        compiled = self._compile(nodes, context)
        try:
            graph_config: dict = {}
            callbacks = langgraph_callbacks()
            if callbacks:
                graph_config["callbacks"] = callbacks
            async for update in compiled.astream(
                {"artifacts": {}},
                stream_mode="updates",
                config=graph_config,
            ):
                for payload in _result_payloads(update):
                    yield payload
            logger.info(
                "Plan execution finished",
                node_count=len(nodes),
                result_count=_result_count(nodes),
            )
        finally:
            if context.timing is not None:
                write_query_times(context.timing, dag_executed=True)

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
                # A list of start nodes waits until every parent has finished.
                # Separate edges would start this node after the first parent.
                graph.add_edge(
                    [str(dependency) for dependency in dependencies],
                    str(node.id),
                )
            if node.id not in depended_on:
                graph.add_edge(str(node.id), END)
        return graph.compile()

    def _runner(self, node: QueryPlanNode, context: _CallContext):
        async def run(state: PlanState) -> dict[str, dict[int, NodeArtifact]]:
            started = time.perf_counter()
            try:
                artifact = await self._run_node(node, state["artifacts"], context)
                return {"artifacts": {node.id: artifact}}
            finally:
                if context.timing is not None:
                    context.timing.add_node(time.perf_counter() - started)

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
        grand_search = None
        if parents:
            standalone_search, grand_search = await self._timed_llm(
                context,
                self._resolve_search(node, parents),
                node,
                "resolve",
            )
            search = standalone_search

        logger.info(f"Search: {search} | Grand search: {grand_search}")
        # mode = await self._timed_llm(context, self._router.route(search), node, "route")
        mode = QueryMode.MIX
        attempts: list[RetrievalAttempt] = []
        retrieved = ""
        last_references: list[Reference] = []
        result: QueryResult | None = None
        extra_used = 0
        while True:
            attempt = len(attempts) + 1
            keywords = _search_keywords(context, root=not parents, first_search=not attempts)
            result = await self._timed_retrieval(
                context,
                self._adapter.query(
                    grand_search or search,
                    mode=mode,
                    only_need_context=True,
                    user_prompt=None,
                    conversation_history=context.conversation_history,
                    include_references=True,
                    **keywords,
                ),
                node,
                f"retrieve {attempt}",
            )
            retrieved = result.response
            last_references = list(result.references or [])
            if extra_used >= self._max_extra_retrievals:
                break
            evaluation = await self._timed_llm(
                context,
                self._evaluator.evaluate(
                    subquery=grand_search or search,
                    context=retrieved,
                    references=result.references,
                    parents=_upstream(parents),
                    attempts=attempts,
                    previous_mode=mode,
                ),
                node,
                f"evaluate {attempt}",
            )
            if evaluation is None:
                break
            attempts.append(
                RetrievalAttempt(
                    mode=mode.value,
                    outcome=_attempt_outcome(evaluation),
                )
            )
            if evaluation.decision != "RETRIEVE_MORE":
                break
            if evaluation.next_mode is None:
                break
            mode = evaluation.next_mode
            extra_used += 1

        assert result is not None
        output = retrieved
        if node.is_result_node:
            output = await self._timed_llm(
                context,
                self._synthesize(parents, retrieved, search, grand_search),
                node,
                "synthesize",
            )
        node.subquery = grand_search or search
        return NodeArtifact(
            node_id=node.id,
            node_type=node.node_type,
            subquery=grand_search or search,
            output=output,
            references=_union_references(
                [parent.references for parent in parents] + [last_references]
            ),
            is_result_node=node.is_result_node,
            mode=result.mode,
            response_time=result.response_time,
        )

    async def _resolve_search(
        self,
        node: QueryPlanNode,
        parents: list[NodeArtifact],
    ) -> tuple[str, str]:
        prompt = (
            f"Subquery:\n{node.subquery}\n\n"
            f"Parent subqueries:\n{_format_subqueries(parents)}\n\n"
            f"Parent outputs:\n{_format_parents(parents)}\n\n"
            "Return the JSON object with standalone_search and grand_search."
        )
        raw = await self._llm.complete_prompt(
            prompt,
            system_prompt=_RESOLVE_SYSTEM_PROMPT,
            model=self._llm.model_cheap,
            reasoning_effort="none",
            response_format={"type": "json_object"},
        )
        standalone_search, grand_search = _resolved_searches(_as_text(raw), node.subquery)
        logger.info(f"Resolved standalone: {standalone_search}")
        logger.info(f"Resolved grand: {grand_search}")
        return standalone_search, grand_search

    async def _synthesize(
        self,
        parents: list[NodeArtifact],
        retrieved: str,
        search: str,
        grand_search: str | None,
    ) -> str:
        context = retrieved.strip() or "None."
        if parents:
            context = f"{context}\n\n{_answer_context(parents)}"
        raw = await self._llm.complete_prompt(
            _answer_prompt(grand_search or search, context),
            system_prompt=_ANSWER_SYSTEM_PROMPT,
            model=self._llm.model_cheap,
            reasoning_effort="none",
        )
        return _as_text(raw).strip()

    async def _reason(
        self,
        node: QueryPlanNode,
        parents: list[NodeArtifact],
        context: _CallContext,
    ) -> NodeArtifact:
        search = node.subquery
        grand_search = None
        if parents:
            standalone_search, grand_search = await self._timed_llm(
                context,
                self._resolve_search(node, parents),
                node,
                "resolve",
            )
            search = standalone_search

        logger.info(f"Search: {search} | Grand search: {grand_search}")
        if node.is_result_node and node.node_type == "synthesize":
            prompt = _answer_prompt(grand_search or search, _answer_context(parents))
            system_prompt = _ANSWER_SYSTEM_PROMPT
            completion_kwargs = {
                "reasoning_effort": "none",
                "timeout": 180,
                "num_retries": 2,
            }
        else:
            parents_text = _format_reason_parents(parents) if parents else "None."
            audience = (
                "This node is a result. Write the answer the user will read."
                if node.is_result_node
                else "This node is not a result. Write the intermediate artifact for the next node."
            )
            prompt = (
                f"User query:\n{grand_search or search}\n\n"
                f"Node {node.id}\n"
                f"Operation type: {node.node_type}\n"
                f"{audience}\n"
                f"Operation:\n{node.subquery}\n\n"
                f"Parent outputs:\n{parents_text}\n\n"
                "Write the complete node output now. "
                "Use only the parent outputs. "
                "If a required fact is absent, name what is missing instead of returning an empty reply."
            )
            system_prompt = _reason_system_prompt(node.node_type)
            completion_kwargs = {
                "reasoning_effort": "none",
                "max_completion_tokens": 10000,
                "timeout": 180,
                "num_retries": 2,
            }

        try:
            raw = await self._timed_llm(
                context,
                self._llm.complete_prompt(
                    prompt,
                    system_prompt=system_prompt,
                    model=self._llm.model_cheap,
                    **completion_kwargs,
                ),
                node,
                "reason",
            )
        except Exception:
            logger.exception(
                "Reasoning call failed",
                node_id=node.id,
                node_type=node.node_type,
            )
            raise

        node.subquery = grand_search or search
        return NodeArtifact(
            node_id=node.id,
            node_type=node.node_type,
            subquery=node.subquery,
            output=_as_text(raw).strip(),
            references=_union_references([parent.references for parent in parents]),
            is_result_node=node.is_result_node,
        )

    async def _timed_llm(self, context: _CallContext, awaitable, node: QueryPlanNode, step: str):
        """Time one wrapper LLM call, including network wait."""
        clear_reported_llm_seconds()
        started = time.perf_counter()
        result = None
        detail = None
        try:
            result = await awaitable
            if step == "route":
                step = f"{step} ({result})\n({node.subquery}):\n"
            elif step == "resolve" and isinstance(result, tuple) and len(result) == 2:
                standalone_search, grand_search = result
                detail = (
                    f"standalone: {' '.join(str(standalone_search).split())} | "
                    f"grand: {' '.join(str(grand_search).split())}"
                )
            elif step.startswith("evaluate") and result is not None:
                step = f"{step} ({result.next_mode}):\n"
            return result
        finally:
            self._record_call(
                context,
                node,
                step,
                time.perf_counter() - started,
                take_reported_llm_seconds(),
                detail,
            )

    async def _timed_retrieval(
        self,
        context: _CallContext,
        awaitable,
        node: QueryPlanNode,
        step: str,
    ) -> QueryResult:
        """Time one LightRAG retrieval. Its response_time is the returned value."""
        started = time.perf_counter()
        reported: float | None = None
        try:
            result = await awaitable
            if isinstance(result.response_time, (int, float)):
                reported = float(result.response_time)
            return result
        finally:
            self._record_call(context, node, step, time.perf_counter() - started, reported)

    def _record_call(
        self,
        context: _CallContext,
        node: QueryPlanNode,
        step: str,
        call_seconds: float,
        response_seconds: float | None,
        detail: str | None = None,
    ) -> None:
        if context.timing is None:
            return
        context.timing.add_dag_llm(
            call_seconds,
            response_seconds,
            node_id=node.id,
            node_type=node.node_type,
            step=step,
            detail=detail,
        )


def _search_keywords(
    context: _CallContext,
    *,
    root: bool,
    first_search: bool,
) -> dict[str, list[str] | None]:
    if root and first_search:
        return {
            "high_level_keywords": context.high_level_keywords,
            "low_level_keywords": context.low_level_keywords,
        }
    return {"high_level_keywords": None, "low_level_keywords": None}


def _upstream(parents: list[NodeArtifact]) -> list[UpstreamOutput]:
    return [
        UpstreamOutput(node_id=parent.node_id, node_type=parent.node_type, output=parent.output)
        for parent in parents
    ]


def _answer_prompt(query: str, context: str) -> str:
    """User message for the LightRAG answer prompt. The query is the question to answer."""
    return f"---User Query---\n{query}\n\n---Context---\n{context}"


_KG_SECTION = re.compile(
    r"Knowledge Graph Data \((Entity|Relationship)\):\s*```(?:json)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def _knowledge_graph_text(output: str) -> tuple[str, str]:
    """Entity and relationship JSON from a LightRAG context string."""
    entities: list[str] = []
    relations: list[str] = []
    for match in _KG_SECTION.finditer(output):
        body = match.group(2).strip()
        if not body:
            continue
        target = entities if match.group(1).lower() == "entity" else relations
        if body not in target:
            target.append(body)
    return "\n".join(entities), "\n".join(relations)


def _answer_context(parents: list[NodeArtifact]) -> str:
    """Passages plus each parent's knowledge-graph blocks, in LightRAG's shape."""
    by_title: dict[str, list[str]] = {}
    order: list[str] = []
    extras: list[str] = []
    entity_blocks: list[str] = []
    relation_blocks: list[str] = []
    for parent in parents:
        passages = _passage_lines(parent.references)
        if passages:
            entities, relations = _knowledge_graph_text(parent.output)
            if entities and entities not in entity_blocks:
                entity_blocks.append(entities)
            if relations and relations not in relation_blocks:
                relation_blocks.append(relations)
            for line in passages:
                title, _, text = line.partition(": ")
                title = title or "unknown"
                if title not in by_title:
                    by_title[title] = []
                    order.append(title)
                if text and text not in by_title[title]:
                    by_title[title].append(text)
        elif parent.output.strip():
            extras.append(parent.output.strip())
    chunks: list[str] = []
    listing: list[str] = []
    for index, title in enumerate(order, start=1):
        chunks.append(f"[{index}] {title}\n" + "\n".join(by_title[title]))
        listing.append(f"[{index}] {title}")
    next_index = len(order) + 1
    for extra in extras:
        chunks.append(f"[{next_index}] Assembled from an earlier step\n{extra}")
        listing.append(f"[{next_index}] Assembled from an earlier step")
        next_index += 1
    if not chunks:
        chunk_section = "Document Chunks:\n\nNone.\n\nReference Document List:\n\nNone."
    else:
        chunk_section = (
            "Document Chunks (each [n] matches the Reference Document List):\n\n"
            + "\n\n".join(chunks)
            + "\n\nReference Document List:\n\n"
            + "\n".join(listing)
        )
    sections: list[str] = []
    if entity_blocks:
        sections.append(
            "Knowledge Graph Data (Entity):\n\n```json\n"
            + "\n".join(entity_blocks)
            + "\n```"
        )
    if relation_blocks:
        sections.append(
            "Knowledge Graph Data (Relationship):\n\n```json\n"
            + "\n".join(relation_blocks)
            + "\n```"
        )
    sections.append(chunk_section)
    return "\n\n".join(sections)


def _attempt_outcome(evaluation: Evaluation) -> str:
    missing = "; ".join(evaluation.missing_information)
    if missing:
        return f"{evaluation.decision}. Missing: {missing}"
    return evaluation.decision


def _repeats_search(query: str, attempts: list[RetrievalAttempt]) -> bool:
    folded = query.strip().casefold()
    return any(attempt.query.strip().casefold() == folded for attempt in attempts)


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


def _format_reason_parents(parents: list[NodeArtifact]) -> str:
    """Parent text for a reason call.

    A non-result retrieval stores the full LightRAG context. When that parent
    also has passage text, send the passages instead of the context dump.
    """
    blocks: list[str] = []
    for parent in parents:
        passages = _passage_lines(parent.references)
        if parent.node_type == "retrieve" and not parent.is_result_node and passages:
            body = "\n".join(passages)
            citations = "none"
        else:
            body = parent.output
            citations = "; ".join(_format_citation(ref) for ref in parent.references) or "none"
        blocks.append(
            f"Node {parent.node_id} ({parent.node_type})\n"
            f"Subquery: {parent.subquery}\n"
            f"Output:\n{body}\n"
            f"Citations: {citations}"
        )
    return "\n\n".join(blocks)


def _passage_lines(references: list[Reference]) -> list[str]:
    lines: list[str] = []
    for ref in references:
        source = ref.file_path or ref.reference_id or "unknown"
        for chunk in ref.content or []:
            text = chunk.strip()
            if text:
                lines.append(f"{source}: {text}")
    return lines


def _format_subqueries(parents: list[NodeArtifact]) -> str:
    blocks: list[str] = []
    for parent in parents:
        blocks.append(
            f"Node {parent.node_id}: Subquery: {parent.subquery}\n"
        )
    logger.info(f"Parent Subqueries (blocks): {blocks}")
    return "\n\n".join(blocks)

def _format_parents(parents: list[NodeArtifact]) -> str:
    blocks: list[str] = []
    for parent in parents:
        citations = "; ".join(_format_citation(ref) for ref in parent.references) or "none"
        blocks.append(
            f"Node {parent.node_id} ({parent.node_type})\n"
            f"Output:\n{parent.output}\n"
            f"Citations: {citations}"
        )
    # logger.info(f"Parents (blocks): {blocks}")
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


def _resolved_searches(raw: str, fallback: str) -> tuple[str, str]:
    """Read both search strings. A missing value falls back to the subquery."""
    data = _json_object(raw)
    standalone = _search_field(data, "standalone_search")
    grand = _search_field(data, "grand_search")
    standalone = standalone or fallback
    return standalone, grand or standalone


def _search_field(data: dict[str, object] | None, key: str) -> str:
    if not data:
        return ""
    value = data.get(key)
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _json_object(text: str) -> dict[str, object] | None:
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
