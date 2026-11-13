"""LangGraph execution of a decomposer plan."""

import asyncio
import json

import pytest

from lightrag_wrapper.exceptions import LLMServiceError, LightRAGAdapterError
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


def _decision(decision, query=None, mode=None, missing=None):
    claim_status = "SUPPORTED" if decision == "SUFFICIENT" else "UNSUPPORTED"
    return json.dumps(
        {
            "claims": [
                {
                    "claim": "the requested fact",
                    "status": claim_status,
                    "evidence_ids": ["E1"] if claim_status == "SUPPORTED" else [],
                    "conflict": None,
                }
            ],
            "decision": decision,
            "missing_information": missing or [],
            "next_action": {
                "type": decision,
                "purpose": None if decision != "RETRIEVE_MORE" else "close the gap",
                "query": query,
                "mode": mode,
            },
        }
    )


class _LLM:
    model_cheap = "test-cheap"
    model_mid_tier = "test-mid"

    def __init__(self):
        self.prompts = []
        self.system_prompts = []
        self.call_kwargs = []
        self.resolved = "executives who approved Project Alpha"
        self.evaluations = []
        self.fail_evaluator = False
        self.synthesis = None

    async def complete_prompt(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.call_kwargs.append(kwargs)
        system = kwargs.get("system_prompt") or ""
        self.system_prompts.append(system)
        if "standalone search string" in system:
            return self.resolved
        if "Evidence Evaluator" in system:
            if self.fail_evaluator:
                raise LLMServiceError("evaluator down")
            if self.evaluations:
                return self.evaluations.pop(0)
            return _decision("SUFFICIENT")
        if "provided **Context**" in system:
            if self.synthesis is not None:
                return self.synthesis
            query = prompt.split("---User Query---\n", 1)[1].split("\n", 1)[0]
            return f"answer:{query}"
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


def _executor(max_extra_retrievals=2):
    llm = _LLM()
    adapter = _Adapter()
    router = _Router()
    executor = PlanExecutor(
        llm,
        adapter,
        router,
        max_extra_retrievals=max_extra_retrievals,
    )
    return executor, llm, adapter, router


def _prompts_for(llm, marker):
    return [
        prompt
        for prompt, system in zip(llm.prompts, llm.system_prompts)
        if marker in system
    ]


async def _events(executor, nodes, **kwargs):
    return [event async for event in executor.stream(nodes, **kwargs)]


async def test_root_result_retrieve_yields_the_synthesized_answer():
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

    assert router.queries == ["vacation policy"]
    assert len(adapter.calls) == 1
    query, kwargs = adapter.calls[0]
    assert query == "vacation policy"
    assert kwargs["only_need_context"] is True
    assert kwargs["high_level_keywords"] == ["leave"]
    assert kwargs["low_level_keywords"] == ["vacation"]
    assert kwargs["conversation_history"] == [{"role": "user", "content": "policy"}]
    evaluation = _prompts_for(llm, "Evidence Evaluator")
    assert len(evaluation) == 1
    assert "vacation policy" in evaluation[0]
    assert "[E1]" in evaluation[0]
    assert "What is the vacation policy?" not in evaluation[0]
    synthesis = _prompts_for(llm, "provided **Context**")
    assert len(synthesis) == 1
    assert "What is the vacation policy?" not in synthesis[0]
    assert "evidence:vacation policy" in synthesis[0]
    assert events == [
        {
            "id": 1,
            "response": "answer:vacation policy",
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
    assert _prompts_for(llm, "provided **Context**") == []
    evaluations = _prompts_for(llm, "Evidence Evaluator")
    assert len(evaluations) == 2
    assert all("Compare the 2024 and 2026 policies." not in prompt for prompt in evaluations)
    compared = _prompts_for(llm, "one operation")
    assert len(compared) == 1
    assert "2024 policy.pdf: passage" in compared[0]
    assert "evidence:2024 policy" not in compared[0]
    assert "evidence:2026 policy" in compared[0]
    reason_systems = [system for system in llm.system_prompts if "one operation" in system]
    assert len(reason_systems) == 1
    assert "The operation is compare." in reason_systems[0]
    assert "filter:" not in reason_systems[0]
    reason_kwargs = [
        kwargs
        for kwargs, system in zip(llm.call_kwargs, llm.system_prompts)
        if "one operation" in system
    ]
    assert reason_kwargs[0]["reasoning_effort"] == "none"
    assert reason_kwargs[0]["max_completion_tokens"] == 10000
    assert "Compare the two policies." in compared[0]
    assert "Compare the 2024 and 2026 policies." in compared[0]
    assert len(events) == 1
    assert events[0]["id"] == 3
    assert events[0]["response"] == "assembled answer"
    assert [ref["file_path"] for ref in events[0]["references"]] == [
        "2024 policy.pdf",
        "2026 policy.pdf",
    ]


async def test_dependent_result_retrieve_resolves_then_retrieves_context_once():
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

    resolved = _prompts_for(llm, "standalone search string")
    assert len(resolved) == 1
    assert "evidence:Identify the project acquired by Company X." in resolved[0]
    assert [query for query, _kwargs in adapter.calls] == [
        "Identify the project acquired by Company X.",
        "executives who approved Project Alpha",
    ]
    assert adapter.calls[0][1]["only_need_context"] is True
    assert adapter.calls[0][1]["high_level_keywords"] == ["acquisition"]
    assert adapter.calls[1][1]["only_need_context"] is True
    assert adapter.calls[1][1]["high_level_keywords"] is None
    assert adapter.calls[1][1]["low_level_keywords"] is None
    assert router.queries == [
        "Identify the project acquired by Company X.",
        "executives who approved Project Alpha",
    ]
    synthesis = _prompts_for(llm, "provided **Context**")
    assert len(synthesis) == 1
    assert "Who approved the project acquired by Company X?" not in synthesis[0]
    assert "evidence:executives who approved Project Alpha" in synthesis[0]
    assert "Identify the project acquired by Company X..pdf" in synthesis[0]
    assert "passage" in synthesis[0]
    assert len(events) == 1
    assert events[0]["id"] == 2
    assert events[0]["response"] == (
        "answer:Identify the executives who approved the project identified in Node 1."
    )
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
        "answer:First fact",
        "answer:Second fact",
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
    assert adapter.calls[0][1]["only_need_context"] is True
    reasoned = _prompts_for(llm, "one operation")
    assert reasoned == []
    synthesized = _prompts_for(llm, "provided **Context**")
    assert len(synthesized) == 2
    assert "passage" in synthesized[1]
    assert "What is the allowance?" in synthesized[1]
    assert [event["id"] for event in events] == [1, 2]
    assert events[0]["response"] == "answer:Find the policy."
    assert events[1]["response"] == "answer:What is the allowance?"


async def test_a_join_waits_until_every_parent_has_finished():
    executor, llm, _adapter, _router = _executor()
    filter_started = asyncio.Event()
    release_filter = asyncio.Event()
    original = llm.complete_prompt

    async def complete_prompt(prompt, **kwargs):
        system = kwargs.get("system_prompt") or ""
        if "one operation" in system and "Operation type: filter" in prompt:
            filter_started.set()
            await release_filter.wait()
        return await original(prompt, **kwargs)

    llm.complete_prompt = complete_prompt
    nodes = [
        _node(1, "retrieve", "Find the policy.", [], False),
        _node(2, "filter", "Keep the allowance.", [1], False),
        _node(3, "synthesize", "State the allowance.", [1, 2], True),
    ]
    task = asyncio.create_task(
        _events(executor, nodes, user_query="What is the allowance?")
    )
    await asyncio.wait_for(filter_started.wait(), timeout=2)
    await asyncio.sleep(0.05)
    assert not task.done()
    assert not any("State the allowance." in prompt for prompt in llm.prompts)
    release_filter.set()
    events = await asyncio.wait_for(task, timeout=2)
    assert [event["id"] for event in events] == [3]


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
    assert events[0]["response"] == "answer:fast fact"


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
    assert events[0]["response"] == "answer:fast fact"


async def test_a_follow_up_search_uses_the_evaluator_query_and_mode():
    executor, llm, adapter, router = _executor()
    llm.evaluations = [
        _decision(
            "RETRIEVE_MORE",
            "What does the vacation policy grant?",
            "hybrid",
            ["the allowance"],
        ),
        _decision("SUFFICIENT"),
    ]
    nodes = [_node(1, "retrieve", "vacation policy", [], True)]

    events = await _events(
        executor,
        nodes,
        user_query="What is the vacation policy?",
        high_level_keywords=["leave"],
        low_level_keywords=["vacation"],
    )

    assert [query for query, _kwargs in adapter.calls] == [
        "vacation policy",
        "vacation policy",
    ]
    assert adapter.calls[0][1]["high_level_keywords"] == ["leave"]
    assert adapter.calls[1][1]["mode"] == QueryMode.HYBRID
    assert adapter.calls[1][1]["high_level_keywords"] is None
    assert adapter.calls[1][1]["low_level_keywords"] is None
    assert router.queries == ["vacation policy"]
    evaluations = _prompts_for(llm, "Evidence Evaluator")
    assert "RETRIEVE_MORE" in evaluations[1]
    assert "the allowance" in evaluations[1]
    assert events[0]["response"] == "answer:vacation policy"
    assert [ref["file_path"] for ref in events[0]["references"]] == [
        "vacation policy.pdf",
    ]


async def test_citations_keep_parents_and_only_the_last_attempt():
    executor, llm, adapter, _router = _executor()
    llm.resolved = "executives of the acquired project"
    llm.evaluations = [
        _decision("SUFFICIENT"),
        _decision("RETRIEVE_MORE", "Who approved Project Alpha?", "hybrid"),
        _decision("SUFFICIENT"),
    ]
    nodes = [
        _node(1, "retrieve", "Identify the project.", [], False),
        _node(2, "retrieve", "Find who approved it.", [1], True),
    ]

    events = await _events(executor, nodes, user_query="Who approved it?")

    assert [query for query, _kwargs in adapter.calls] == [
        "Identify the project.",
        "executives of the acquired project",
        "executives of the acquired project",
    ]
    assert [ref["file_path"] for ref in events[0]["references"]] == [
        "Identify the project..pdf",
        "executives of the acquired project.pdf",
    ]


async def test_extra_retrievals_stop_after_the_budget():
    executor, llm, adapter, router = _executor(max_extra_retrievals=1)
    llm.evaluations = [
        _decision("RETRIEVE_MORE", "second search", "global", ["more"]),
        _decision("RETRIEVE_MORE", "third search", "naive", ["still"]),
        _decision("RETRIEVE_MORE", "fourth search", "mix"),
    ]
    nodes = [_node(1, "retrieve", "first search", [], True)]

    events = await _events(executor, nodes, user_query="Find it.")

    assert [query for query, _kwargs in adapter.calls] == ["first search", "first search"]
    assert adapter.calls[1][1]["mode"] is QueryMode.GLOBAL
    assert router.queries == ["first search"]
    assert len(_prompts_for(llm, "Evidence Evaluator")) == 1
    assert events[0]["response"] == "answer:first search"


async def test_unusable_evaluation_does_not_search_again():
    executor, llm, adapter, _router = _executor()
    llm.evaluations = ["not json"]
    nodes = [_node(1, "retrieve", "vacation policy", [], True)]

    events = await _events(executor, nodes, user_query="What is the vacation policy?")

    assert len(adapter.calls) == 1
    assert events[0]["response"] == "answer:vacation policy"


async def test_a_repeated_follow_up_does_not_search_again():
    executor, llm, adapter, _router = _executor()
    llm.evaluations = [_decision("RETRIEVE_MORE", "vacation policy", "local")]
    nodes = [_node(1, "retrieve", "vacation policy", [], True)]

    await _events(executor, nodes, user_query="What is the vacation policy?")

    assert len(adapter.calls) == 1


async def test_an_evaluator_failure_still_synthesizes_the_result():
    executor, llm, adapter, _router = _executor()
    llm.fail_evaluator = True
    nodes = [_node(1, "retrieve", "vacation policy", [], True)]

    events = await _events(executor, nodes, user_query="What is the vacation policy?")

    assert len(adapter.calls) == 1
    assert events[0]["response"] == "answer:vacation policy"


async def test_cannot_answer_does_not_search_again():
    executor, llm, adapter, _router = _executor()
    llm.evaluations = [_decision("CANNOT_ANSWER", missing=["the date"])]
    nodes = [_node(1, "retrieve", "vacation policy", [], True)]

    events = await _events(executor, nodes, user_query="What is the vacation policy?")

    assert len(adapter.calls) == 1
    assert events[0]["response"] == "answer:vacation policy"


def test_missing_parent_output_is_an_error():
    with pytest.raises(RuntimeError, match="node 4"):
        _require_parent({}, 4)


async def test_stream_accumulates_dag_times_and_writes_the_query(tmp_path):
    from lightrag_wrapper.query_timing import QueryTiming

    executor, _llm, _adapter, _router = _executor()
    nodes = [
        _node(1, "retrieve", "vacation policy", [], False),
        _node(2, "assemble", "combine the evidence", [1], True),
    ]
    query = "What is the vacation policy?"
    timing = QueryTiming(query=query)

    events = await _events(executor, nodes, user_query=query, timing=timing)

    assert events[0]["response"] == "assembled answer"
    assert timing.dag_nodes == 2
    assert timing.dag_llm_calls == 4
    assert timing.dag_llm_response_samples == 1
    assert timing.dag_llm_response_s == pytest.approx(1.0)
    assert timing.dag_node_s + 1e-6 >= timing.dag_llm_call_s
    text = (tmp_path / "times.log").read_text(encoding="utf-8")
    head, _rewriter = text.split("Query rewriter", 1)
    assert query in head
    assert "LLM call time:" in text
    assert "LLM response time: 1.000 s (1 of 4 calls returned a response_time)" in text
    assert "Node execution time:" in text
    assert "(2 nodes)" in text
    node_1, node_2 = text.split("Node 2 (assemble)", 1)
    assert "Node 1 (retrieve)" in node_1
    assert node_1.index("route (") < node_1.index("retrieve 1:") < node_1.index("evaluate 1 (")
    assert "retrieve 1:" in node_1 and "LLM response time 1.000 s" in node_1
    assert "synthesize:" not in node_1
    assert node_2.strip().startswith("reason:")
    assert "LLM response time none" in node_2


async def test_stream_logs_resolve_follow_up_and_synthesize(tmp_path):
    from lightrag_wrapper.query_timing import QueryTiming

    executor, llm, _adapter, _router = _executor()
    llm.evaluations = [
        _decision("SUFFICIENT"),
        _decision(
            "RETRIEVE_MORE",
            "What does the vacation policy grant?",
            "hybrid",
            ["the allowance"],
        ),
        _decision("SUFFICIENT"),
    ]
    nodes = [
        _node(1, "retrieve", "vacation policy", [], False),
        _node(2, "retrieve", "Find who approved it.", [1], True),
    ]
    query = "Who approved the policy?"

    await _events(executor, nodes, user_query=query, timing=QueryTiming(query=query))

    text = (tmp_path / "times.log").read_text(encoding="utf-8")
    node_1, node_2 = text.split("Node 2 (retrieve)", 1)
    assert "Node 1 (retrieve)" in node_1
    assert "resolve:" not in node_1
    assert "synthesize:" not in node_1
    assert (
        node_2.index("resolve:")
        < node_2.index("route (")
        < node_2.index("retrieve 1:")
        < node_2.index("evaluate 1 (")
        < node_2.index("retrieve 2:")
        < node_2.index("evaluate 2 (")
        < node_2.index("synthesize:")
    )
    assert node_2.count("LLM response time 1.000 s") == 2


async def test_a_failed_retrieval_is_still_named_in_the_timing_log(tmp_path):
    from lightrag_wrapper.query_timing import QueryTiming

    executor, _llm, adapter, _router = _executor()

    async def query(query, **kwargs):
        raise LightRAGAdapterError("query down", status_code=500)

    adapter.query = query
    nodes = [_node(1, "retrieve", "vacation policy", [], True)]
    with pytest.raises(LightRAGAdapterError, match="query down"):
        await _events(
            executor,
            nodes,
            user_query="What is the vacation policy?",
            timing=QueryTiming(query="What is the vacation policy?"),
        )

    text = (tmp_path / "times.log").read_text(encoding="utf-8")
    retrieve = next(line for line in text.splitlines() if "retrieve 1:" in line)
    assert retrieve.endswith("LLM response time none")
    assert "evaluate 1:" not in text
    assert "synthesize:" not in text
