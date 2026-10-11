"""LangGraph execution of a decomposer plan."""

import asyncio

import pytest

from lightrag_wrapper.exceptions import LightRAGAdapterError
from lightrag_wrapper.models import QueryMode, QueryPlanNode, QueryResult, Reference
from lightrag_wrapper.query_executor.executor import PlanExecutor, _require_parent


def _node(node_id, node_type, subquery, depends_on, is_result):
    return QueryPlanNode(
        Id=node_id,
        nodeType=node_type,
        subquery=subquery,
        depends_on=depends_on,
        isResultNode=is_result,
    )


class _LLM:
    def __init__(self):
        self.prompts = []
        self.system_prompts = []
        self.resolved = "executives who approved Project Alpha"

    async def complete_prompt(self, prompt, **kwargs):
        self.prompts.append(prompt)
        system = kwargs.get("system_prompt") or ""
        self.system_prompts.append(system)
        if "standalone search string" in system:
            return self.resolved
        return "assembled answer"


class _Router:
    def __init__(self):
        self.queries = []

    async def route(self, query):
        self.queries.append(query)
        if "2026" in query:
            return QueryMode.GLOBAL
        return QueryMode.LOCAL


class _Adapter:
    def __init__(self):
        self.calls = []

    async def query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return QueryResult(
            response=f"evidence:{query}",
            mode=kwargs["mode"],
            references=[
                Reference(
                    file_path=f"{query}.pdf",
                    content=None if "2026" in query else ["passage"],
                )
            ],
            response_time=1.0,
        )


def _executor():
    llm = _LLM()
    adapter = _Adapter()
    router = _Router()
    return PlanExecutor(llm, adapter, router), llm, adapter, router


async def _events(executor, nodes, **kwargs):
    return [event async for event in executor.stream(nodes, **kwargs)]


async def test_root_result_retrieve_answers_in_one_call():
    executor, llm, adapter, router = _executor()
    nodes = [_node(1, "retrieve", "vacation policy", [], True)]

    events = await _events(
        executor,
        nodes,
        user_query="What is the vacation policy?",
        conversation_history=[{"role": "user", "content": "policy"}],
        high_level_keywords=["leave"],
        low_level_keywords=["vacation"],
    )

    assert llm.prompts == []
    assert router.queries == ["vacation policy"]
    assert len(adapter.calls) == 1
    query, kwargs = adapter.calls[0]
    assert query == "vacation policy"
    assert kwargs["only_need_context"] is False
    assert kwargs["high_level_keywords"] == ["leave"]
    assert kwargs["low_level_keywords"] == ["vacation"]
    assert kwargs["conversation_history"] == [{"role": "user", "content": "policy"}]
    assert events == [
        {
            "id": 1,
            "response": "evidence:vacation policy",
            "references": [
                {"reference_id": None, "file_path": "vacation policy.pdf", "content": ["passage"]}
            ],
        }
    ]


async def test_compare_uses_both_branches_and_does_not_search_again():
    executor, llm, adapter, _router = _executor()
    nodes = [
        _node(1, "retrieve", "2024 policy", [], False),
        _node(2, "retrieve", "2026 policy", [], False),
        _node(3, "compare", "Compare the two policies.", [1, 2], True),
    ]

    events = await _events(
        executor,
        nodes,
        user_query="Compare the 2024 and 2026 policies.",
        high_level_keywords=["policy"],
        low_level_keywords=["2024"],
    )

    assert {query for query, _kwargs in adapter.calls} == {"2024 policy", "2026 policy"}
    assert all(kwargs["only_need_context"] is True for _query, kwargs in adapter.calls)
    assert all(kwargs["high_level_keywords"] == ["policy"] for _query, kwargs in adapter.calls)
    assert len(llm.prompts) == 1
    assert "evidence:2024 policy" in llm.prompts[0]
    assert "evidence:2026 policy" in llm.prompts[0]
    assert "Compare the two policies." in llm.prompts[0]
    assert "Compare the 2024 and 2026 policies." in llm.prompts[0]
    assert len(events) == 1
    assert events[0]["id"] == 3
    assert events[0]["response"] == "assembled answer"
    assert [ref["file_path"] for ref in events[0]["references"]] == [
        "2024 policy.pdf",
        "2026 policy.pdf",
    ]


async def test_dependent_result_retrieve_resolves_then_answers_once():
    executor, llm, adapter, router = _executor()
    nodes = [
        _node(1, "retrieve", "Identify the project acquired by Company X.", [], False),
        _node(
            2,
            "retrieve",
            "Identify the executives who approved the project identified in Node 1.",
            [1],
            True,
        ),
    ]

    events = await _events(
        executor,
        nodes,
        user_query="Who approved the project acquired by Company X?",
        high_level_keywords=["acquisition"],
        low_level_keywords=["Company X"],
    )

    assert len(llm.prompts) == 1
    assert "standalone search string" in llm.system_prompts[0]
    assert "evidence:Identify the project acquired by Company X." in llm.prompts[0]
    assert [query for query, _kwargs in adapter.calls] == [
        "Identify the project acquired by Company X.",
        "executives who approved Project Alpha",
    ]
    assert adapter.calls[0][1]["only_need_context"] is True
    assert adapter.calls[0][1]["high_level_keywords"] == ["acquisition"]
    assert adapter.calls[1][1]["only_need_context"] is False
    assert adapter.calls[1][1]["high_level_keywords"] is None
    assert adapter.calls[1][1]["low_level_keywords"] is None
    assert router.queries[1] == "executives who approved Project Alpha"
    assert len(events) == 1
    assert events[0]["id"] == 2
    assert events[0]["response"] == "evidence:executives who approved Project Alpha"
    paths = [ref["file_path"] for ref in events[0]["references"]]
    assert "Identify the project acquired by Company X..pdf" in paths
    assert "executives who approved Project Alpha.pdf" in paths


async def test_result_nodes_are_yielded_as_separate_objects():
    executor, _llm, _adapter, _router = _executor()
    nodes = [
        _node(2, "retrieve", "Second fact", [], True),
        _node(1, "retrieve", "First fact", [], True),
    ]

    events = await _events(executor, nodes, user_query="Give me both facts.")

    assert [event["id"] for event in events] == [2, 1] or [event["id"] for event in events] == [1, 2]
    assert {event["response"] for event in events} == {
        "evidence:First fact",
        "evidence:Second fact",
    }
    assert all("\n\n" not in event["response"] for event in events)


async def test_a_result_node_still_feeds_its_children():
    executor, llm, adapter, _router = _executor()
    nodes = [
        _node(1, "retrieve", "Find the policy.", [], True),
        _node(2, "synthesize", "State the allowance.", [1], True),
    ]

    events = await _events(executor, nodes, user_query="What is the allowance?")

    assert len(adapter.calls) == 1
    assert adapter.calls[0][1]["only_need_context"] is False
    assert len(llm.prompts) == 1
    assert "synthesize" in llm.prompts[0]
    assert "evidence:Find the policy." in llm.prompts[0]
    assert [event["id"] for event in events] == [1, 2]
    assert events[0]["response"] == "evidence:Find the policy."
    assert events[1]["response"] == "assembled answer"


async def test_blank_resolution_keeps_the_subquery():
    executor, llm, adapter, _router = _executor()
    llm.resolved = "   "
    nodes = [
        _node(1, "retrieve", "Identify the project.", [], False),
        _node(2, "retrieve", "Find who approved it.", [1], True),
    ]

    await _events(executor, nodes, user_query="Who approved it?")

    assert adapter.calls[1][0] == "Find who approved it."


async def test_a_slow_result_does_not_delay_an_earlier_one():
    executor, _llm, adapter, _router = _executor()
    release = asyncio.Event()
    original = adapter.query

    async def query(query, **kwargs):
        if query == "slow fact":
            await release.wait()
        return await original(query, **kwargs)

    adapter.query = query
    nodes = [
        _node(1, "retrieve", "fast fact", [], True),
        _node(2, "retrieve", "slow fact", [], True),
    ]

    async def collect():
        events = []
        async for event in executor.stream(nodes, user_query="both"):
            events.append(event)
            if len(events) == 1:
                release.set()
        return events

    events = await asyncio.wait_for(collect(), timeout=2)
    assert [event["id"] for event in events] == [1, 2]
    assert events[0]["response"] == "evidence:fast fact"


async def test_a_later_failure_follows_the_result_already_yielded():
    executor, _llm, adapter, _router = _executor()
    original = adapter.query

    async def query(query, **kwargs):
        if query != "fast fact":
            raise LightRAGAdapterError("query down", status_code=500)
        return await original(query, **kwargs)

    adapter.query = query
    nodes = [
        _node(1, "retrieve", "fast fact", [], True),
        _node(2, "retrieve", "Find the follow-up.", [1], True),
    ]
    events = []
    with pytest.raises(LightRAGAdapterError, match="query down"):
        async for event in executor.stream(nodes, user_query="both"):
            events.append(event)
    assert events[0]["id"] == 1
    assert events[0]["response"] == "evidence:fast fact"


def test_missing_parent_output_is_an_error():
    with pytest.raises(RuntimeError, match="node 4"):
        _require_parent({}, 4)
