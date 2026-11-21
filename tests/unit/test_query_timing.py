"""Query timing totals and the times.log report."""

from types import SimpleNamespace

import pytest

from lightrag_wrapper.exceptions import LLMServiceError
from lightrag_wrapper.query_decomposer.query_decomposer import QueryDecomposer
from lightrag_wrapper.query_rewriter.query_rewriter import QueryRewriter
from lightrag_wrapper.query_timing import (
    QueryTiming,
    note_reported_llm_seconds,
    render_query_times,
    reported_llm_seconds,
    take_reported_llm_seconds,
    write_query_times,
)


def test_litellm_response_ms_is_converted_to_seconds():
    assert reported_llm_seconds(SimpleNamespace(_response_ms=2500)) == pytest.approx(2.5)
    assert reported_llm_seconds({"response_time": 1.25}) == pytest.approx(1.25)
    assert reported_llm_seconds(SimpleNamespace()) is None


def test_reported_time_is_taken_once():
    note_reported_llm_seconds(SimpleNamespace(_response_ms=1000))
    assert take_reported_llm_seconds() == pytest.approx(1.0)
    assert take_reported_llm_seconds() is None


def test_report_puts_the_query_first_and_lists_each_pre_dag_call():
    timing = QueryTiming(query="How many leave days\napplies in Berlin?")
    timing.add_rewriter_call(1.25)
    timing.add_decomposer_call(0.5)
    timing.add_decomposer_call(0.25)
    timing.add_dag_llm(3.0, 2.5)
    timing.add_dag_llm(1.0, None)
    timing.add_node(4.5)
    timing.add_node(1.5)

    text = render_query_times(timing, dag_executed=True)
    head, rest = text.split("Query rewriter", 1)
    assert "How many leave days\napplies in Berlin?" in head
    assert "LLM call time: 1.250 s" in rest
    assert "1. 0.500 s" in rest
    assert "2. 0.250 s" in rest
    assert "total: 0.750 s" in rest
    assert "LLM call time: 4.000 s (2 calls)" in rest
    assert "LLM response time: 2.500 s (1 of 2 calls returned a response_time)" in rest
    assert "Node execution time: 6.000 s (2 nodes)" in rest
    assert "LLM calls" not in rest


def test_report_lists_each_dag_call_under_its_node():
    timing = QueryTiming(query="How many leave days?")
    timing.add_dag_llm(
        0.4,
        0.3,
        node_id=2,
        node_type="assemble",
        step="resolve",
        detail="Where is the Harris Museum located?",
    )
    timing.add_dag_llm(0.2, None, node_id=1, node_type="retrieve", step="route")
    timing.add_dag_llm(1.5, 1.0, node_id=1, node_type="retrieve", step="retrieve 1")
    timing.add_dag_llm(0.3, 0.25, node_id=1, node_type="retrieve", step="evaluate 1")
    timing.add_dag_llm(0.8, None, node_id=1, node_type="retrieve", step="retrieve 2")
    timing.add_node(3.2)
    timing.add_node(0.4)

    text = render_query_times(timing, dag_executed=True)
    calls = text.split("LLM calls", 1)[1]
    assert calls.index("Node 1 (retrieve)") < calls.index("Node 2 (assemble)")
    node_1, node_2 = calls.split("Node 2 (assemble)", 1)
    assert node_1.index("route:") < node_1.index("retrieve 1:") < node_1.index("evaluate 1:")
    assert node_1.index("evaluate 1:") < node_1.index("retrieve 2:")
    assert "route: LLM call time 0.200 s, LLM response time none" in node_1
    assert "retrieve 1: LLM call time 1.500 s, LLM response time 1.000 s" in node_1
    assert "evaluate 1: LLM call time 0.300 s, LLM response time 0.250 s" in node_1
    assert "retrieve 2: LLM call time 0.800 s, LLM response time none" in node_1
    assert "resolve: LLM call time 0.400 s, LLM response time 0.300 s" in node_2
    assert "resolved query: Where is the Harris Museum located?" in node_2
    assert "LLM call time: 3.200 s (5 calls)" in text
    assert "LLM response time: 1.550 s (3 of 5 calls returned a response_time)" in text


def test_report_marks_a_query_that_never_reached_the_dag():
    text = render_query_times(QueryTiming(query="bare question"), dag_executed=False)
    assert text.split("Query rewriter", 1)[0].count("bare question") == 1
    assert "LLM call time: no call" in text
    assert "not executed" in text


def test_reports_append_one_segment_per_query(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "times.log"
    monkeypatch.setenv("QUERY_TIMES_LOG", str(path))
    write_query_times(QueryTiming(query="first question"), dag_executed=False)
    write_query_times(QueryTiming(query="second question"), dag_executed=False)
    text = path.read_text(encoding="utf-8")
    assert text.index("first question") < text.index("second question")
    assert text.count("=" * 80) == 4


class _RewriterLLM:
    model_mid_tier = "mid"

    def __init__(self, payload):
        self.payload = payload

    async def complete_prompt(self, prompt, **kwargs):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


async def test_rewriter_records_the_llm_call_wall_clock():
    timing = QueryTiming(query="Why was it delayed?")
    raw = (
        '{"rewritten_query": "Reasons for delays to Project Orion",'
        ' "high_level_keywords": [], "low_level_keywords": [],'
        ' "is_context_dependent": false, "ambiguities": [], "confidence": 0.5}'
    )
    result = await QueryRewriter(_RewriterLLM(raw)).rewrite("Why was it delayed?", timing=timing)
    assert result.rewritten_query.startswith("Reasons")
    assert len(timing.rewriter_calls_s) == 1
    assert timing.rewriter_calls_s[0] >= 0


async def test_rewriter_records_a_failed_llm_call():
    timing = QueryTiming(query="original")
    result = await QueryRewriter(_RewriterLLM(LLMServiceError("down"))).rewrite(
        "original",
        timing=timing,
    )
    assert result.rewritten_query == "original"
    assert len(timing.rewriter_calls_s) == 1


class _DecomposerLLM:
    model_high_end = "high"

    async def complete_prompt(self, prompt, **kwargs):
        raise LLMServiceError("down")


class _Catalog:
    async def list_document_catalog(self):
        return {"documents": []}


async def test_decomposer_records_the_llm_call_wall_clock():
    timing = QueryTiming(query="How many days?")
    plan = await QueryDecomposer(_DecomposerLLM(), _Catalog()).decompose(
        "How many days?",
        timing=timing,
    )
    assert plan is None
    assert len(timing.decomposer_calls_s) == 1
    assert timing.decomposer_calls_s[0] >= 0
