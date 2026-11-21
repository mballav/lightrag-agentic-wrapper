"""Loaders, judge, metrics, and report for the evaluation package."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lightrag_wrapper.config import LightRAGConfig
from lightrag_wrapper.evaluation.datasets import EvalQuestion, Passage, load_dataset, select_questions
from lightrag_wrapper.evaluation.judge import combine_judgments, parse_judgment
from lightrag_wrapper.evaluation.metrics import (
    estimate_cost,
    percentile,
    retrieval_hit,
    win_rates,
)
from lightrag_wrapper.evaluation.report import comparison_row, render_report, write_run
from lightrag_wrapper.evaluation.runner import (
    QuestionComparison,
    SystemRun,
    compare_question,
    isolated_config,
)
from lightrag_wrapper.llm.usage import TokenUsage, begin_token_usage, take_token_usage, token_usage_from_response
from lightrag_wrapper.models import QueryMode


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loaders_normalize_the_three_datasets(tmp_path: Path):
    dragon = _write(
        tmp_path / "dragon.json",
        [
            {
                "id": "d1",
                "question": "What is the fee?",
                "answer": "18",
                "domain": "finance",
                "doc": "The annual fee is 18 days.",
                "evidence": [{"title": "Fee clause", "content": "The annual fee is 18 days."}],
            },
            {
                "id": "d2",
                "question": "What is the cap?",
                "ground_truth": "50",
                "domain": "law",
                "documents": [{"title": "Cap", "content": "The cap is 50 questions per week."}],
            },
        ],
    )
    hotpot = _write(
        tmp_path / "hotpot.json",
        [
            {
                "_id": "h1",
                "question": "Who directed the film?",
                "answer": "Nolan",
                "type": "bridge",
                "supporting_facts": [["Film", 0]],
                "context": [["Film", ["Nolan directed the film."]], ["Other", ["Unrelated."]]],
            }
        ],
    )
    musique = tmp_path / "musique.jsonl"
    musique.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "2hop__1",
                        "question": "Where is the capital?",
                        "answer": "Paris",
                        "answerable": True,
                        "paragraphs": [
                            {"title": "City", "paragraph_text": "The capital is Paris.", "is_supporting": True},
                            {"title": "Other", "paragraph_text": "Noise.", "is_supporting": False},
                        ],
                    }
                ),
                json.dumps({"id": "2hop__2", "question": "Skip", "answer": "", "answerable": False, "paragraphs": []}),
            ]
        ),
        encoding="utf-8",
    )

    dragon_rows = load_dataset("dragonball", dragon)
    assert [row.id for row in dragon_rows] == ["d1", "d2"]
    assert dragon_rows[0].category == "finance"
    assert dragon_rows[0].supporting[0].title == "Fee clause"
    hotpot_rows = load_dataset("hotpotqa", hotpot)
    assert hotpot_rows[0].category == "bridge"
    assert [item.title for item in hotpot_rows[0].supporting] == ["Film"]
    musique_rows = load_dataset("musique", musique)
    assert [row.id for row in musique_rows] == ["2hop__1"]
    assert musique_rows[0].category == "2hop"
    assert musique_rows[0].supporting[0].text == "The capital is Paris."


def test_sampling_is_stable_for_a_seed():
    questions = [
        EvalQuestion(str(index), f"Q{index}", "a", "finance" if index % 2 == 0 else "law", (), ())
        for index in range(6)
    ]
    first = select_questions(questions, category="finance", limit=2, seed=3)
    second = select_questions(questions, category="finance", limit=2, seed=3)
    assert [item.id for item in first] == [item.id for item in second]
    assert len(first) == 2
    assert {item.category for item in first} == {"finance"}
    only = select_questions(questions, question_id="1", category="finance", limit=0, seed=9)
    assert [item.id for item in only] == ["1"]
    listed = select_questions(questions, question_id="5, missing, 1, 5")
    assert [item.id for item in listed] == ["5", "1"]
    assert select_questions(questions, question_id="missing, also") == []
    assert select_questions(questions, question_id="  ") == select_questions(questions, limit=None)


def test_judge_requires_agreement_across_both_orders():
    raw = """
    {
      "Comprehensiveness": {"Winner": "Answer 1", "Explanation": "more detail"},
      "Diversity": {"Winner": "Tie", "Explanation": "same"},
      "Empowerment": {"Winner": "Answer 2", "Explanation": "clearer"},
      "Overall": {"Winner": "Answer 1", "Explanation": "stronger"}
    }
    """
    first = parse_judgment(raw, agentic_first=True)
    swapped = """
    {
      "Comprehensiveness": {"Winner": "Answer 2", "Explanation": "more detail"},
      "Diversity": {"Winner": "Tie", "Explanation": "same"},
      "Empowerment": {"Winner": "Answer 1", "Explanation": "clearer"},
      "Overall": {"Winner": "Answer 1", "Explanation": "the other side"}
    }
    """
    second = parse_judgment(swapped, agentic_first=False)
    assert first is not None and second is not None
    combined = combine_judgments(first, second)
    assert combined.winners["Comprehensiveness"] == "agentic"
    assert combined.winners["Diversity"] == "tie"
    assert combined.winners["Empowerment"] == "native"
    assert combined.winners["Overall"] == "tie"
    assert parse_judgment("not json", agentic_first=True) is None


def test_metrics_cover_rates_hits_and_estimated_cost():
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert win_rates(["agentic", "native", "tie", "agentic"]) == {
        "agentic": 0.5,
        "native": 0.25,
        "tie": 0.25,
    }
    hit = retrieval_hit(
        [{"file_path": "Fee clause.pdf", "content": ["The annual fee is 18 days of leave for staff."]}],
        (Passage("Fee clause", "The annual fee is 18 days of leave for staff."),),
    )
    assert hit is True
    assert retrieval_hit([], (Passage("Fee", "The annual fee is 18 days of leave for staff."),)) is False
    cost = estimate_cost(
        [TokenUsage(model="gpt-test", input_tokens=1_000_000, output_tokens=500_000, total_tokens=1_500_000)],
        {"gpt-test": {"input": 2.0, "output": 4.0}},
    )
    assert cost["estimated_cost"] == pytest.approx(4.0)
    assert cost["actual_cost"] is None


def test_token_usage_is_read_from_the_provider_response():
    response = SimpleNamespace(
        model="gpt-test",
        usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "cost": 0.01},
    )
    usage = token_usage_from_response(response)
    assert usage is not None
    assert usage.input_tokens == 3
    assert usage.output_tokens == 5
    assert usage.actual_cost == pytest.approx(0.01)
    begin_token_usage()
    from lightrag_wrapper.llm.usage import note_token_usage

    note_token_usage(response)
    recorded = take_token_usage()
    assert recorded == [usage]
    assert take_token_usage() == []


def test_isolated_config_does_not_keep_the_production_workspace(tmp_path: Path):
    config = LightRAGConfig(lightrag_mode="server", working_dir="./data/embedded/rag_storage", workspace=None)
    isolated = isolated_config(config, tmp_path / "rag_storage")
    assert isolated.lightrag_mode == "embedded"
    assert isolated.workspace == "evaluation"
    assert isolated.working_dir == str(tmp_path / "rag_storage")
    assert isolated.save_dag_visual is False
    assert isolated.enable_rerank is True
    disabled = isolated_config(config, tmp_path / "rag_storage", enable_rerank=False)
    assert disabled.enable_rerank is False


async def test_a_question_failure_does_not_raise():
    question = EvalQuestion("q1", "What is the fee?", "18", "finance", (), ())

    async def _fail(item):
        raise RuntimeError("down")

    comparison = await compare_question(question, native=_fail, agentic=_fail, judge=_fail)
    assert comparison.error == "down"
    assert comparison.judgment is None


def test_report_writes_json_csv_and_summary(tmp_path: Path):
    question = EvalQuestion("q1", "What is the fee?", "18", "finance", (Passage("Fee", "The fee is 18."),), ())
    native = SystemRun("short", [], 1.0, 1, [], error=None)
    agentic = SystemRun(
        "The fee clause sets the allowance at 18 and explains who it covers in detail.",
        [{"file_path": "Fee", "content": ["The fee is 18."]}],
        2.5,
        2,
        [TokenUsage("gpt-test", 10, 20, 30)],
    )
    from lightrag_wrapper.evaluation.judge import CriterionJudgment, Judgment

    criteria = {
        name: CriterionJudgment("answer 1", "because")
        for name in ("Comprehensiveness", "Diversity", "Empowerment", "Overall")
    }
    first = Judgment(True, criteria)
    second = Judgment(
        False,
        {name: CriterionJudgment("answer 2", "because") for name in criteria},
    )
    from lightrag_wrapper.evaluation.judge import combine_judgments

    comparison = QuestionComparison(
        question=question,
        native=native,
        agentic=agentic,
        judgment=combine_judgments(first, second),
        judge_usages=[],
    )
    output = write_run([comparison], output_dir=tmp_path / "run", dataset="dragonball", prices={})
    assert (output / "results.json").is_file()
    assert (output / "results.csv").is_file()
    summary = (output / "report.md").read_text(encoding="utf-8")
    assert "Dataset: dragonball" in summary
    assert "Answer quality" in summary
    assert "q1" in summary
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert payload[0]["winners"]["Overall"] == "agentic"
    assert payload[0]["plan"] is None
    text = render_report(payload, dataset="dragonball")
    assert "| **Overall (1)** |" in text
    assert "| **finance (1)** |" in text
    assert "| agentic | 100.0% | 100.0% | 100.0% | 100.0% |" in text


def test_overall_winner_is_accepted_as_overall():
    raw = """
    {
      "Comprehensiveness": {"Winner": "Answer 1", "Explanation": "a"},
      "Diversity": {"Winner": "Tie", "Explanation": "b"},
      "Empowerment": {"Winner": "Answer 2", "Explanation": "c"},
      "Overall Winner": {"Winner": "Answer 1", "Explanation": "d"}
    }
    """
    parsed = parse_judgment(raw, agentic_first=True)
    assert parsed is not None
    assert parsed.criteria["Overall"].winner == "answer 1"


def test_prices_skip_incomplete_entries(tmp_path: Path):
    from lightrag_wrapper.evaluation.pricing import load_prices

    path = tmp_path / "prices.json"
    path.write_text('{"gpt-test": {"input": 1.5, "output": 2.5}, "bad": "no"}', encoding="utf-8")
    assert load_prices(path)["gpt-test"] == {"input": 1.5, "output": 2.5}
    assert "bad" not in load_prices(path)
    bundled = load_prices(None)
    assert bundled["gpt-6-luna"] == {"input": 0.10, "output": 0.50}
    assert bundled["gpt-6.1-terra"]["output"] == 5.00
    assert bundled["gpt-6.2-sol"]["input"] == 2.00
    assert bundled["text-embedding-3-small"] == {"input": 0.02, "output": 0.0}
    luna = estimate_cost(
        [TokenUsage("openai/gpt-6-luna", 1_000_000, 1_000_000, 2_000_000)],
        bundled,
    )
    assert luna["estimated_cost"] == pytest.approx(0.60)
    assert load_prices(tmp_path / "missing.json") == {}


def test_dragonball_queries_are_joined_to_their_documents():
    from lightrag_wrapper.evaluation.datasets import _combine_dragonball

    combined = _combine_dragonball(
        [
            {
                "domain": "Finance",
                "language": "en",
                "query": {"query_id": 7, "content": "When was the CEO appointed?"},
                "ground_truth": {
                    "doc_ids": [4],
                    "content": "January 2021",
                    "references": ["The first sub-event was the appointment of a new CEO in January 2021."],
                },
            }
        ],
        [
            {
                "domain": "Finance",
                "language": "en",
                "doc_id": 4,
                "company_name": "Green Fields",
                "content": "A new CEO was appointed in January 2021.",
            }
        ],
    )
    assert combined[0]["id"] == "en-Finance-7"
    assert combined[0]["domain"] == "Finance"
    assert combined[0]["documents"][0]["title"] == "Green Fields"
    assert combined[0]["answer"] == "January 2021"


def test_download_dragonball_writes_the_response(tmp_path: Path, monkeypatch):
    from lightrag_wrapper.evaluation.datasets import download_dragonball

    class _Response:
        def read(self):
            return b"[]"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "lightrag_wrapper.evaluation.datasets.urlopen",
        lambda *args, **kwargs: _Response(),
    )
    path = download_dragonball(tmp_path / "nested" / "dragon.json")
    assert path.read_bytes() == b"[]"


async def test_dataset_path_uses_a_local_file_and_requires_data(tmp_path: Path):
    from lightrag_wrapper.evaluation.__main__ import _dataset_path

    data = tmp_path / "hotpot.json"
    data.write_text("[]", encoding="utf-8")
    assert await _dataset_path("hotpotqa", str(data)) == data
    with pytest.raises(SystemExit):
        await _dataset_path("hotpotqa", None)
    with pytest.raises(SystemExit):
        await _dataset_path("dragonball", str(tmp_path / "missing.json"))


async def test_native_and_agentic_runs_record_usage_and_retrievals():
    from lightrag_wrapper.evaluation.judge import JUDGE_SYSTEM_PROMPT
    from lightrag_wrapper.evaluation.runner import RetrievalCounter, judge_pair, run_agentic, run_native
    from lightrag_wrapper.models import QueryResult, Reference
    from lightrag_wrapper.query_decomposer.query_decomposer import Decomposition
    from lightrag_wrapper.query_timing import DagCall

    question = EvalQuestion("q1", "What is the fee?", "18", "finance", (), ())

    class _Inner:
        async def query(self, text, mode, only_need_context=False):
            return QueryResult(
                response=f"native {mode.value}",
                mode=mode,
                references=[Reference(file_path="Fee", content=["The fee is 18."])],
            )

    native = await run_native(RetrievalCounter(_Inner()), question, QueryMode.MIX)
    assert native.answer == "native mix"
    assert native.retrievals == 1
    assert native.calls_by_component == {"query": 1}

    class _Decomposer:
        async def decompose(self, query, history, timing=None):
            timing.decomposer_calls_s.append(0.2)
            return None

    class _Router:
        async def route(self, query):
            return QueryMode.NAIVE

    class _Executor:
        def __init__(self):
            self._adapter = RetrievalCounter(_Inner())

    fallback = await run_agentic(
        question,
        decomposer=_Decomposer(),
        executor=_Executor(),
        router=_Router(),
    )
    assert fallback.answer == "native naive"
    assert fallback.plan is None
    assert fallback.calls_by_component["decomposer"] == 1
    assert fallback.retrievals == 1

    class _Planned:
        async def decompose(self, query, history, timing=None):
            timing.decomposer_calls_s.append(0.2)
            from lightrag_wrapper.models import QueryPlanNode

            return Decomposition(
                nodes=[
                    QueryPlanNode(
                        id=1,
                        node_type="retrieve",
                        subquery="Find the fee.",
                        depends_on=[],
                        is_result_node=False,
                    ),
                    QueryPlanNode(
                        id=2,
                        node_type="synthesize",
                        subquery="State the fee.",
                        depends_on=[1],
                        is_result_node=True,
                    ),
                ],
                knowledge_base_metadata="",
            )

    class _Streaming:
        def __init__(self):
            self._adapter = None

        async def stream(self, nodes, *, user_query, timing=None):
            timing.dag_calls.append(DagCall(1, "synthesize", "reason", 0.4, 0.3))
            nodes[1].subquery = "What is the fee?"
            yield {
                "response": "assembled answer",
                "references": [{"file_path": "Fee", "content": ["The fee is 18."]}],
            }

    planned = await run_agentic(
        question,
        decomposer=_Planned(),
        executor=_Streaming(),
        router=_Router(),
    )
    assert planned.answer == "assembled answer"
    assert planned.plan == [
        {
            "Id": 1,
            "nodeType": "retrieve",
            "subquery": "Find the fee.",
            "depends_on": [],
            "isResultNode": False,
            "original_subquery": "Find the fee.",
        },
        {
            "Id": 2,
            "nodeType": "synthesize",
            "subquery": "What is the fee?",
            "depends_on": [1],
            "isResultNode": True,
            "original_subquery": "State the fee.",
        },
    ]
    assert planned.calls_by_component["reason"] == 1
    assert planned.calls_by_component["decomposer"] == 1
    assert planned.references[0]["file_path"] == "Fee"

    class _Judge:
        async def complete_prompt(self, prompt, **kwargs):
            assert kwargs["system_prompt"] == JUDGE_SYSTEM_PROMPT
            winner = "Answer 1" if "assembled answer" in prompt.split("Answer 2:", 1)[0] else "Answer 2"
            return json.dumps(
                {
                    name: {"Winner": winner, "Explanation": "detail"}
                    for name in ("Comprehensiveness", "Diversity", "Empowerment", "Overall")
                }
            )

    judgment, usages = await judge_pair(_Judge(), question, native, planned, model="judge-model")
    assert judgment is not None
    assert judgment.winners["Overall"] == "agentic"
    assert usages == []


async def test_ingest_deduplicates_passages():
    from lightrag_wrapper.evaluation.runner import _ingest_questions

    shared = Passage("Fee", "The fee is 18.")
    questions = [
        EvalQuestion("a", "Q", "18", None, (shared,), (shared,)),
        EvalQuestion("b", "Q2", "18", None, (shared,), (shared, Passage("Cap", "The cap is 50."))),
    ]

    class _Adapter:
        def __init__(self):
            self.texts = None
            self.sources = None

        async def ingest_texts(self, texts, sources=None, wait=True):
            self.texts = texts
            self.sources = sources

    adapter = _Adapter()
    await _ingest_questions(adapter, questions)
    assert adapter.texts == ["The fee is 18.", "The cap is 50."]
    assert adapter.sources == ["Fee", "Cap"]


def test_small_metric_and_parser_edges():
    from lightrag_wrapper.evaluation.metrics import summarize_latencies
    from lightrag_wrapper.llm.usage import note_token_usage

    assert percentile([], 0.5) is None
    assert win_rates([]) == {"agentic": 0.0, "native": 0.0, "tie": 0.0}
    assert retrieval_hit([], ()) is None
    assert summarize_latencies([])["average"] is None
    assert estimate_cost(
        [TokenUsage("gpt-test", 10, 0, 10, actual_cost=1.25)],
        {"vendor/gpt-test": {"input": 1.0, "output": 1.0}},
    )["actual_cost"] == pytest.approx(1.25)
    reference = SimpleNamespace(file_path="Fee", content="The fee is 18.")
    assert retrieval_hit([reference], (Passage("Fee", "x" * 50),)) is True
    assert token_usage_from_response(SimpleNamespace(usage=None)) is None
    assert token_usage_from_response(SimpleNamespace(usage={"prompt_tokens": 0, "completion_tokens": 0})) is None
    hidden = SimpleNamespace(
        model=None,
        usage=SimpleNamespace(prompt_tokens=True, input_tokens=2, completion_tokens=0, total_tokens=0),
        _hidden_params={"response_cost": "nope"},
    )
    usage = token_usage_from_response(hidden)
    assert usage is not None
    assert usage.input_tokens == 2
    assert usage.actual_cost is None
    note_token_usage(hidden)
    fenced = """```json
    {"Comprehensiveness": {"Winner": 1, "Explanation": "a"}}
    ```"""
    assert parse_judgment(fenced, agentic_first=True) is None
    prose = 'note {"Comprehensiveness": {"Winner": "Answer 1", "Explanation": "a"}} trailing'
    assert parse_judgment(prose, agentic_first=True) is None
    assert parse_judgment('{"Comprehensiveness": {"Winner": "Answer 1"}}', agentic_first=True) is None


def test_dataset_shapes_and_unknown_name(tmp_path: Path):
    payload = {
        "data": [
            {
                "id": "doc",
                "domain": "finance",
                "documents": {"title": "Policy", "text": "The fee is 18."},
                "qa_pairs": [
                    {"question": "What is the fee?", "answers": ["18"]},
                    {"query": "", "answer": ""},
                ],
            }
        ]
    }
    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = load_dataset("dragonball", path)
    assert len(rows) == 1
    assert rows[0].reference_answer == "18"
    assert rows[0].documents[0].title == "Policy"
    assert select_questions(rows, limit=0) == []
    assert select_questions(rows, limit=None) == rows
    with pytest.raises(ValueError):
        load_dataset("other", path)
    musique = tmp_path / "odd.jsonl"
    musique.write_text(
        json.dumps(
            {
                "id": "3hop__9",
                "question": "Which city?",
                "answer": ["Paris", "Lyon"],
                "paragraphs": ["skip", {"title": "City", "paragraph_text": "Paris is the capital city of France."}],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_dataset("musique", musique)
    assert loaded[0].reference_answer == "Paris"
    hotpot = tmp_path / "hotpot.json"
    hotpot.write_text(
        json.dumps(
            [
                {
                    "question": "Who?",
                    "answer": "Nolan",
                    "context": ["plain sentence", ["Film", "Nolan directed it."]],
                    "supporting_facts": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    assert load_dataset("hotpotqa", hotpot)[0].documents[1].title == "Film"


def test_report_covers_missing_judgment_and_actual_cost():
    question = EvalQuestion("q2", "What?", "x", None, (), ())
    native = SystemRun("no", [], 1.0, 0, [TokenUsage("m", 1, 1, 2, actual_cost=0.2)])
    agentic = SystemRun("no", [], 1.0, 0, [TokenUsage("m", 1, 1, 2, actual_cost=0.2)])
    comparison = QuestionComparison(question, native, agentic, None, [], error=None)
    text = render_report([comparison_row(comparison, {})], dataset="dragonball")
    assert "| **Overall (1)** |" in text
    assert "**uncategorized**" not in text
    assert "actual 0.200000" in text


def test_repeated_report_shows_aggregated_and_average_metrics():
    rows = [
        {
            "id": "q",
            "category": "2hop",
            "question": "What?",
            "repeat": 1,
            "winners": {name: "agentic" for name in ("Comprehensiveness", "Diversity", "Empowerment", "Overall")},
            "native_latency_s": 1.0,
            "agentic_latency_s": 2.0,
            "native_retrieval_hit": True,
            "agentic_retrieval_hit": True,
            "native_usage": {"input_tokens": 10, "output_tokens": 1, "llm_calls": 1, "estimated_cost": 0.1, "actual_cost": 0.1},
            "agentic_usage": {"input_tokens": 20, "output_tokens": 2, "llm_calls": 2, "estimated_cost": 0.2, "actual_cost": 0.2},
            "judge_usage": {"input_tokens": 5, "output_tokens": 1, "llm_calls": 2, "estimated_cost": 0.05, "actual_cost": 0.05},
            "native_answer": "a native answer long enough to count",
            "agentic_answer": "an agentic answer long enough to count",
        },
        {
            "id": "q",
            "category": "2hop",
            "question": "What?",
            "repeat": 2,
            "winners": {name: "tie" for name in ("Comprehensiveness", "Diversity", "Empowerment", "Overall")},
            "native_latency_s": 3.0,
            "agentic_latency_s": 4.0,
            "native_retrieval_hit": False,
            "agentic_retrieval_hit": True,
            "native_usage": {"input_tokens": 30, "output_tokens": 3, "llm_calls": 3, "estimated_cost": 0.3, "actual_cost": 0.3},
            "agentic_usage": {"input_tokens": 40, "output_tokens": 4, "llm_calls": 4, "estimated_cost": 0.4, "actual_cost": 0.4},
            "judge_usage": {"input_tokens": 5, "output_tokens": 1, "llm_calls": 2, "estimated_cost": 0.05, "actual_cost": 0.05},
            "native_answer": "a native answer long enough to count",
            "agentic_answer": "an agentic answer long enough to count",
        },
    ]
    text = render_report(rows, dataset="musique")
    assert "Repeats: 2" in text
    assert "Question runs: 2" in text
    assert "Questions evaluated: 1" in text
    aggregated, average = text.split("## Average", 1)
    assert "calls 4" in aggregated
    assert "calls 2.000" in average
    assert "| agentic | 50.0% | 50.0% | 50.0% | 50.0% |" in aggregated
    assert "| agentic | 50.0% | 50.0% | 50.0% | 50.0% |" in average


async def test_question_errors_and_a_bad_judge_are_kept():
    question = EvalQuestion("q3", "What?", "x", "law", (), ())

    async def _native(item):
        return SystemRun("", [], 0.1, 0, [], error="native: down")

    async def _agentic(item):
        return SystemRun("ok", [], 0.1, 0, [])

    async def _unused(*args):
        raise AssertionError("judge should not run")

    failed = await compare_question(question, native=_native, agentic=_agentic, judge=_unused)
    assert failed.error == "native: down"

    async def _ok(item):
        return SystemRun("answer", [], 0.1, 0, [])

    async def _bad_judge(item, native_run, agentic_run):
        return None, []

    judged = await compare_question(question, native=_ok, agentic=_ok, judge=_bad_judge)
    assert judged.error == "The judge returned an unusable comparison."


async def test_measured_failure_and_compare_questions(monkeypatch, tmp_path: Path):
    from lightrag_wrapper.evaluation import runner
    from lightrag_wrapper.exceptions import LightRAGAdapterError

    question = EvalQuestion("q4", "What?", "x", None, (Passage("", "text"),), ())

    class _Broken:
        async def query(self, *args, **kwargs):
            raise LightRAGAdapterError("offline")

    failed = await runner.run_native(_Broken(), question, QueryMode.LOCAL)
    assert failed.error.startswith("native:")
    assert runner._take_retrievals() == 0

    class _Adapter:
        def __init__(self):
            self.closed = False

        async def initialize(self):
            return None

        async def aclose(self):
            self.closed = True

        async def ingest_texts(self, texts, sources=None, wait=True):
            self.texts = texts

        def extra(self):
            return "kept"

    adapter = _Adapter()
    monkeypatch.setattr(runner, "create_adapter", lambda config: adapter)
    monkeypatch.setattr(runner, "LLMService", lambda config: object())
    monkeypatch.setattr(runner, "QueryDecomposer", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "QueryModeRouter", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "PlanExecutor", lambda *args, **kwargs: object())

    class _PlanCache:
        async def aclose(self):
            return None

    async def _open_plan_cache(config):
        return _PlanCache()

    monkeypatch.setattr(runner, "open_plan_cache", _open_plan_cache)

    async def _fake(question, **kwargs):
        return QuestionComparison(question, None, None, None, [], error="stored")

    monkeypatch.setattr(runner, "compare_question", _fake)
    config = LightRAGConfig()
    rows = await runner.compare_questions([question], config=config, judge_model="judge")
    assert rows[0].error == "stored"
    assert adapter.closed is True
    assert adapter.texts == ["text"]
    assert runner.RetrievalCounter(adapter).extra() == "kept"

    adapter.ingest_calls = 0
    original_ingest = adapter.ingest_texts

    async def _counting_ingest(texts, sources=None, wait=True):
        adapter.ingest_calls += 1
        return await original_ingest(texts, sources=sources, wait=wait)

    adapter.ingest_texts = _counting_ingest
    repeated = await runner.compare_questions(
        [question],
        config=config,
        judge_model="judge",
        repeats=2,
    )
    assert adapter.ingest_calls == 1
    assert [item.repeat for item in repeated] == [1, 2]
    with pytest.raises(ValueError):
        await runner.compare_questions([question], config=config, judge_model="judge", repeats=0)


async def test_cli_main_writes_the_three_files(tmp_path: Path, monkeypatch):
    from lightrag_wrapper.evaluation import __main__ as cli
    from lightrag_wrapper.evaluation.runner import QuestionComparison

    data = tmp_path / "dragon.json"
    data.write_text(
        json.dumps([{"id": "d1", "question": "What?", "answer": "18", "domain": "finance", "doc": "The fee is 18."}]),
        encoding="utf-8",
    )
    question = EvalQuestion("d1", "What?", "18", "finance", (), ())

    async def _compare(questions, **kwargs):
        return [QuestionComparison(question, SystemRun("n", [], 0.2, 1, []), SystemRun("a", [], 0.3, 1, []), None, [])]

    async def _config(path, workspace="evaluation", enable_rerank=None):
        return LightRAGConfig()

    monkeypatch.setattr(cli, "compare_questions", _compare)
    monkeypatch.setattr(cli, "load_isolated_config", _config)
    monkeypatch.setattr(cli, "configure_logging", lambda level: None)
    args = cli.build_parser().parse_args(
        ["--dataset", "dragonball", "--data", str(data), "--limit", "1", "--output-dir", str(tmp_path / "runs")]
    )
    await cli._main(args)
    reports = list((tmp_path / "runs").glob("*/report.md"))
    assert len(reports) == 1
    assert "Dataset: dragonball" in reports[0].read_text(encoding="utf-8")


async def test_cli_question_id_runs_that_question_only(tmp_path: Path, monkeypatch, capsys):
    from lightrag_wrapper.evaluation import __main__ as cli
    from lightrag_wrapper.evaluation.runner import QuestionComparison

    data = tmp_path / "dragon.json"
    data.write_text(
        json.dumps(
            [
                {"id": "d1", "question": "What?", "answer": "18", "domain": "finance", "doc": "The fee is 18."},
                {"id": "d2", "question": "Who?", "answer": "Ann", "domain": "law", "doc": "Ann signed it."},
            ]
        ),
        encoding="utf-8",
    )
    seen: list[str] = []

    async def _compare(questions, **kwargs):
        seen.extend(item.id for item in questions)
        return [
            QuestionComparison(item, SystemRun("n", [], 0.2, 1, []), SystemRun("a", [], 0.3, 1, []), None, [])
            for item in questions
        ]

    async def _config(path, workspace="evaluation", enable_rerank=None):
        return LightRAGConfig()

    monkeypatch.setattr(cli, "compare_questions", _compare)
    monkeypatch.setattr(cli, "load_isolated_config", _config)
    monkeypatch.setattr(cli, "configure_logging", lambda level: None)
    args = cli.build_parser().parse_args(
        [
            "--dataset",
            "dragonball",
            "--data",
            str(data),
            "--question-id",
            "d2, absent, d1",
            "--limit",
            "0",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    await cli._main(args)
    assert seen == ["d2", "d1"]
    assert "absent" in capsys.readouterr().out
    missing = cli.build_parser().parse_args(
        ["--dataset", "dragonball", "--data", str(data), "--question-id", "absent,gone", "--output-dir", str(tmp_path / "runs")]
    )
    with pytest.raises(SystemExit, match="No questions from the id list"):
        await cli._main(missing)


async def test_cli_ingest_only_skips_evaluation(tmp_path: Path, monkeypatch, capsys):
    from lightrag_wrapper.evaluation import __main__ as cli

    data = tmp_path / "dragon.json"
    data.write_text(
        json.dumps(
            [
                {"id": "d1", "question": "What?", "answer": "18", "domain": "finance", "doc": "The fee is 18."},
                {"id": "d2", "question": "Who?", "answer": "Ann", "domain": "law", "doc": "Ann signed it."},
            ]
        ),
        encoding="utf-8",
    )
    seen: list[str] = []

    async def _ingest(questions, *, config):
        seen.extend(item.id for item in questions)
        return 3

    async def _compare(*args, **kwargs):
        raise AssertionError("evaluation should not run")

    async def _config(path, workspace="evaluation", enable_rerank=None):
        return LightRAGConfig()

    monkeypatch.setattr(cli, "ingest_questions", _ingest)
    monkeypatch.setattr(cli, "compare_questions", _compare)
    monkeypatch.setattr(cli, "load_isolated_config", _config)
    monkeypatch.setattr(cli, "configure_logging", lambda level: None)
    args = cli.build_parser().parse_args(
        [
            "--dataset",
            "dragonball",
            "--data",
            str(data),
            "--limit",
            "1",
            "--seed",
            "0",
            "--ingest-only",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    await cli._main(args)
    assert seen == ["d2"]
    assert "Ingested 3 unique passages for 1 questions." in capsys.readouterr().out
    assert list((tmp_path / "runs").glob("*/report.md")) == []


async def test_ingest_questions_sends_unique_passages(monkeypatch):
    from lightrag_wrapper.evaluation import runner

    shared = Passage("Fee", "The fee is 18.")
    questions = [
        EvalQuestion("a", "Q", "18", None, (shared,), (shared,)),
        EvalQuestion("b", "Q2", "18", None, (shared,), (shared, Passage("Cap", "The cap is 50."))),
    ]

    class _Adapter:
        def __init__(self):
            self.texts = None
            self.closed = False

        async def initialize(self):
            return None

        async def ingest_texts(self, texts, sources=None, wait=True):
            self.texts = list(texts)

        async def aclose(self):
            self.closed = True

    adapter = _Adapter()
    monkeypatch.setattr(runner, "create_adapter", lambda config: adapter)
    count = await runner.ingest_questions(questions, config=LightRAGConfig())
    assert count == 2
    assert adapter.texts == ["The fee is 18.", "The cap is 50."]
    assert adapter.closed is True


def test_cli_int_env(monkeypatch):
    from lightrag_wrapper.evaluation.__main__ import _int_env

    monkeypatch.delenv("EVAL_LIMIT", raising=False)
    assert _int_env("EVAL_LIMIT", 15) == 15
    monkeypatch.setenv("EVAL_LIMIT", "4")
    assert _int_env("EVAL_LIMIT", 15) == 4


