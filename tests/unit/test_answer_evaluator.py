"""Parsing and prompting for one retrieval evaluation."""

import json

import pytest

from lightrag_wrapper.answer_evaluator.evaluator import (
    AnswerEvaluator,
    RetrievalAttempt,
    UpstreamOutput,
    parse_evaluation,
)
from lightrag_wrapper.exceptions import LLMServiceError
from lightrag_wrapper.models import QueryMode, Reference


def _payload(**overrides):
    body = {
        "decision": "SUFFICIENT",
        "missing_information": [],
        "next_action": {"type": "SUFFICIENT", "purpose": None, "query": None, "mode": None},
    }
    body.update(overrides)
    return json.dumps(body)


def test_parse_evaluation_accepts_a_fenced_sufficient_decision():
    raw = "```json\n" + _payload() + "\n```"
    evaluation = parse_evaluation(raw)
    assert evaluation is not None
    assert evaluation.decision == "SUFFICIENT"
    assert evaluation.next_query is None
    assert evaluation.next_mode is None


def test_parse_evaluation_ignores_a_missing_information_value_that_is_not_a_list():
    evaluation = parse_evaluation(_payload(missing_information="the date"))
    assert evaluation is not None
    assert evaluation.missing_information == []


def test_parse_evaluation_keeps_a_follow_up_query_and_mode():
    evaluation = parse_evaluation(
        _payload(
            decision="RETRIEVE_MORE",
            missing_information=["the 2024 policy", "  "],
            next_action={
                "type": "RETRIEVE_MORE",
                "purpose": "find the missing version",
                "query": " What does the 2024 policy say? ",
                "mode": "Naive",
            },
        )
    )
    assert evaluation is not None
    assert evaluation.decision == "RETRIEVE_MORE"
    assert evaluation.next_query == "What does the 2024 policy say?"
    assert evaluation.next_mode is QueryMode.NAIVE
    assert evaluation.missing_information == ["the 2024 policy"]


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        _payload(decision="REPLAN"),
        _payload(next_action={"type": "STOP", "query": None, "mode": None}),
        _payload(
            decision="RETRIEVE_MORE",
            next_action={"type": "RETRIEVE_MORE", "query": "  ", "mode": "local"},
        ),
        _payload(
            decision="RETRIEVE_MORE",
            next_action={"type": "RETRIEVE_MORE", "query": "Where is it?", "mode": "bypass"},
        ),
        _payload(
            decision="RETRIEVE_MORE",
            next_action={"type": "RETRIEVE_MORE", "query": 1, "mode": "local"},
        ),
        _payload(
            decision="RETRIEVE_MORE",
            next_action={"type": "RETRIEVE_MORE", "query": "Where is it?", "mode": 1},
        ),
        _payload(
            decision="RETRIEVE_MORE",
            next_action={"type": "RETRIEVE_MORE", "query": "Where is it?", "mode": "sideways"},
        ),
        _payload(decision=1),
        "see {not json}",
        "[1, 2]",
    ],
)
def test_parse_evaluation_rejects_a_decision_that_cannot_be_followed(raw):
    assert parse_evaluation(raw) is None


class _LLM:
    def __init__(self, result):
        self.result = result
        self.prompts = []

    async def complete_prompt(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_evaluate_numbers_chunk_text_and_omits_parent_text_from_evidence():
    llm = _LLM(_payload())
    evaluator = AnswerEvaluator(llm)
    evaluation = await evaluator.evaluate(
        subquery="Determine the launch date.",
        context="ignored when chunks exist",
        references=[
            Reference(file_path="launch.pdf", content=["", "Product X launched on March 12."]),
            Reference(reference_id="2", content=["A second passage."]),
        ],
        parents=[UpstreamOutput(node_id=1, node_type="retrieve", output="upstream fact")],
        attempts=[RetrievalAttempt(query="earlier", mode="local", outcome="RETRIEVE_MORE")],
    )

    assert evaluation is not None
    assert evaluation.decision == "SUFFICIENT"
    prompt = llm.prompts[0]
    assert "[E1] Source: launch.pdf. Content: \"Product X launched on March 12.\"" in prompt
    assert "[E2] Source: 2. Content: \"A second passage.\"" in prompt
    evidence, parents = prompt.split("RETRIEVED EVIDENCE:\n", 1)[1].split("PARENT OUTPUTS:\n", 1)
    assert "upstream fact" in parents
    assert "upstream fact" not in evidence
    assert 'Query: "earlier" Mode: local. Outcome: RETRIEVE_MORE' in prompt


async def test_evaluate_uses_the_context_when_no_chunk_text_is_present():
    llm = _LLM(
        _payload(
            decision="CANNOT_ANSWER",
            next_action={"type": "CANNOT_ANSWER", "purpose": None, "query": None, "mode": None},
        )
    )
    evaluator = AnswerEvaluator(llm)
    evaluation = await evaluator.evaluate(
        subquery="Find the date.",
        context="No passages mention it.",
        references=[Reference(file_path="empty.pdf")],
        parents=[],
        attempts=[],
    )

    assert evaluation is not None
    assert evaluation.decision == "CANNOT_ANSWER"
    assert '[E1] Source: retrieved context. Content: "No passages mention it."' in llm.prompts[0]
    assert "PARENT OUTPUTS:\nNone." in llm.prompts[0]
    assert "PREVIOUS ATTEMPTS:\nNone." in llm.prompts[0]


async def test_evaluate_stops_when_the_llm_call_fails():
    llm = _LLM(LLMServiceError("down"))
    evaluator = AnswerEvaluator(llm)
    evaluation = await evaluator.evaluate(
        subquery="Find the date.",
        context="context",
        references=[],
        parents=[],
        attempts=[],
    )
    assert evaluation is None
