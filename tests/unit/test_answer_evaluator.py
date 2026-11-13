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


def _claim(status="SUPPORTED", evidence_ids=None, conflict=None):
    if evidence_ids is None:
        evidence_ids = ["E1"] if status == "SUPPORTED" else []
    return {
        "claim": "the requested fact",
        "status": status,
        "evidence_ids": evidence_ids,
        "conflict": conflict,
    }


def _payload(**overrides):
    decision = overrides.get("decision", "SUFFICIENT")
    body = {
        "claims": [_claim() if decision == "SUFFICIENT" else _claim("UNSUPPORTED")],
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
    assert evaluation.next_mode is None
    assert evaluation.claims[0].status == "SUPPORTED"
    assert evaluation.claims[0].evidence_ids == ("E1",)


def test_parse_evaluation_ignores_a_missing_information_value_that_is_not_a_list():
    evaluation = parse_evaluation(_payload(missing_information="the date"))
    assert evaluation is not None
    assert evaluation.missing_information == []


def test_parse_evaluation_keeps_a_retry_mode_when_a_claim_is_uncovered():
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
    assert evaluation.next_mode is QueryMode.NAIVE
    assert evaluation.missing_information == ["the 2024 policy"]


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        _payload(decision="REPLAN"),
        _payload(next_action={"type": "STOP", "query": None, "mode": None}),
        _payload(claims=[]),
        _payload(claims=[{"claim": " ", "status": "SUPPORTED", "evidence_ids": ["E1"]}]),
        _payload(claims=[{"claim": "a fact", "status": "MAYBE", "evidence_ids": ["E1"]}]),
        _payload(
            decision="RETRIEVE_MORE",
            next_action={"type": "RETRIEVE_MORE", "query": "Where is it?", "mode": "bypass"},
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
    model_cheap = "test-cheap"

    def __init__(self, result):
        self.result = result
        self.prompts = []

    async def complete_prompt(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_parse_evaluation_follows_the_claims_when_the_label_disagrees():
    covered = parse_evaluation(
        _payload(
            decision="RETRIEVE_MORE",
            claims=[_claim("SUPPORTED"), _claim("CONTRADICTED", ["E1", "E2"], "resolved")],
            next_action={"type": "RETRIEVE_MORE", "purpose": "again", "mode": "hybrid"},
        )
    )
    assert covered is not None
    assert covered.decision == "SUFFICIENT"
    assert covered.next_mode is None

    open_claim = parse_evaluation(
        _payload(
            claims=[_claim("PARTIAL", ["E1"])],
            next_action={"type": "SUFFICIENT", "purpose": None, "mode": None},
        )
    )
    assert open_claim is not None
    assert open_claim.decision == "CANNOT_ANSWER"

    repeated_mode = parse_evaluation(
        _payload(
            decision="RETRIEVE_MORE",
            claims=[_claim("UNSUPPORTED")],
            next_action={"type": "RETRIEVE_MORE", "purpose": "the gap", "mode": "local"},
        ),
        used_modes=["local"],
    )
    assert repeated_mode is not None
    assert repeated_mode.decision == "CANNOT_ANSWER"
    assert repeated_mode.next_mode is None

    lower_mode = parse_evaluation(
        _payload(
            decision="RETRIEVE_MORE",
            claims=[_claim("UNSUPPORTED")],
            next_action={"type": "RETRIEVE_MORE", "purpose": "the gap", "mode": "naive"},
        ),
        used_modes=["hybrid"],
    )
    assert lower_mode is not None
    assert lower_mode.decision == "CANNOT_ANSWER"

    thematic = parse_evaluation(
        _payload(
            decision="RETRIEVE_MORE",
            claims=[_claim("UNSUPPORTED")],
            next_action={"type": "RETRIEVE_MORE", "purpose": "the overview", "mode": "global"},
        ),
        used_modes=["mix"],
    )
    assert thematic is not None
    assert thematic.decision == "RETRIEVE_MORE"
    assert thematic.next_mode is QueryMode.GLOBAL


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
        attempts=[RetrievalAttempt(mode="local", outcome="RETRIEVE_MORE")],
        previous_mode=QueryMode.LOCAL,
    )

    assert evaluation is not None
    assert evaluation.decision == "SUFFICIENT"
    prompt = llm.prompts[0]
    assert "RETRIEVED CONTEXT:" not in prompt
    assert llm.kwargs["reasoning_effort"] == "low"
    assert "[E1] Source: launch.pdf. Content: \"Product X launched on March 12.\"" in prompt
    assert "[E2] Source: 2. Content: \"A second passage.\"" in prompt
    evidence, parents = prompt.split("RETRIEVED EVIDENCE:\n", 1)[1].split("PARENT OUTPUTS:\n", 1)
    assert "upstream fact" in parents
    assert "upstream fact" not in evidence
    assert "1. Outcome: RETRIEVE_MORE, Mode: local" in prompt


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
        previous_mode=QueryMode.LOCAL,
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
        previous_mode=QueryMode.LOCAL,
    )
    assert evaluation is None
