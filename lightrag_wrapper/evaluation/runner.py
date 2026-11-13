"""Run native LightRAG and the agentic pipeline on an isolated index."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lightrag_wrapper.adapter import create_adapter
from lightrag_wrapper.config import LightRAGConfig, get_config
from lightrag_wrapper.evaluation.datasets import EvalQuestion
from lightrag_wrapper.evaluation.judge import (
    JUDGE_SYSTEM_PROMPT,
    CombinedJudgment,
    Judgment,
    combine_judgments,
    parse_judgment,
    render_judge_prompt,
)
from lightrag_wrapper.evaluation.metrics import question_documents
from lightrag_wrapper.exceptions import LightRAGAdapterError, LLMServiceError
from lightrag_wrapper.llm.service import LLMService
from lightrag_wrapper.llm.usage import TokenUsage, begin_token_usage, take_token_usage
from lightrag_wrapper.models import QueryMode, QueryResult
from lightrag_wrapper.query_decomposer import QueryDecomposer
from lightrag_wrapper.query_executor import PlanExecutor
from lightrag_wrapper.query_timing import QueryTiming
from lightrag_wrapper.router import QueryModeRouter

_retrieval_count: ContextVar[list[int] | None] = ContextVar("eval_retrieval_count", default=None)


@dataclass
class SystemRun:
    """One system's answer and the measurements around it."""

    answer: str
    references: list[Any]
    latency_s: float
    retrievals: int
    usages: list[TokenUsage]
    calls_by_component: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass
class QuestionComparison:
    """Both answers, both judge orders, and an error when the question could not finish."""

    question: EvalQuestion
    native: SystemRun | None
    agentic: SystemRun | None
    judgment: CombinedJudgment | None
    judge_usages: list[TokenUsage]
    error: str | None = None


class RetrievalCounter:
    """Count ``query`` calls without changing the adapter's other methods."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def query(self, *args: Any, **kwargs: Any) -> Any:
        bucket = _retrieval_count.get()
        if bucket is not None:
            bucket[0] += 1
        return await self._inner.query(*args, **kwargs)


def isolated_config(base: LightRAGConfig, working_dir: Path) -> LightRAGConfig:
    """Embedded settings that do not use the production working directory or workspace."""
    return base.model_copy(
        update={
            "lightrag_mode": "embedded",
            "working_dir": str(working_dir),
            "workspace": "evaluation",
            "save_dag_visual": False,
        }
    )


async def compare_questions(
    questions: list[EvalQuestion],
    *,
    config: LightRAGConfig,
    judge_model: str,
    native_mode: QueryMode = QueryMode.MIX,
    concurrency: int = 2,
) -> list[QuestionComparison]:
    """Ingest the sampled passages once, then compare each question."""
    adapter = RetrievalCounter(create_adapter(config))
    llm = LLMService(config)
    await adapter.initialize()
    try:
        await _ingest_questions(adapter, questions)
        decomposer = QueryDecomposer(llm, adapter, save_dag_visual=False)
        router = QueryModeRouter(llm)
        executor = PlanExecutor(llm, adapter, router)
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _one(question: EvalQuestion) -> QuestionComparison:
            async with semaphore:
                return await compare_question(
                    question,
                    native=lambda item: run_native(adapter, item, native_mode),
                    agentic=lambda item: run_agentic(
                        item,
                        decomposer=decomposer,
                        executor=executor,
                        router=router,
                    ),
                    judge=lambda item, left, right: judge_pair(
                        llm,
                        item,
                        left,
                        right,
                        model=judge_model,
                    ),
                )

        return list(await asyncio.gather(*[_one(question) for question in questions]))
    finally:
        await adapter.aclose()


async def compare_question(
    question: EvalQuestion,
    *,
    native: Callable[[EvalQuestion], Awaitable[SystemRun]],
    agentic: Callable[[EvalQuestion], Awaitable[SystemRun]],
    judge: Callable[[EvalQuestion, SystemRun, SystemRun], Awaitable[tuple[CombinedJudgment | None, list[TokenUsage]]]],
) -> QuestionComparison:
    """Compare one question. A failure is stored on the row."""
    native_run: SystemRun | None = None
    agentic_run: SystemRun | None = None
    try:
        native_run = await native(question)
        agentic_run = await agentic(question)
        if native_run.error or agentic_run.error:
            return QuestionComparison(
                question=question,
                native=native_run,
                agentic=agentic_run,
                judgment=None,
                judge_usages=[],
                error=native_run.error or agentic_run.error,
            )
        judgment, judge_usages = await judge(question, native_run, agentic_run)
        error = None if judgment is not None else "The judge returned an unusable comparison."
        return QuestionComparison(
            question=question,
            native=native_run,
            agentic=agentic_run,
            judgment=judgment,
            judge_usages=judge_usages,
            error=error,
        )
    except (LightRAGAdapterError, LLMServiceError, RuntimeError, OSError) as exc:
        return QuestionComparison(
            question=question,
            native=native_run,
            agentic=agentic_run,
            judgment=None,
            judge_usages=[],
            error=str(exc),
        )


async def run_native(adapter: Any, question: EvalQuestion, mode: QueryMode) -> SystemRun:
    """One native LightRAG answer."""
    return await _measured(
        "native",
        lambda: adapter.query(question.question, mode=mode, only_need_context=False),
    )


async def run_agentic(
    question: EvalQuestion,
    *,
    decomposer: QueryDecomposer,
    executor: PlanExecutor,
    router: QueryModeRouter,
) -> SystemRun:
    """The same pipeline the query API uses when there is no conversation history."""
    timing = QueryTiming(query=question.question)

    async def _answer() -> QueryResult:
        plan = await decomposer.decompose(question.question, None, timing=timing)
        if plan is None:
            mode = await router.route(question.question)
            result = await executor._adapter.query(
                question.question,
                mode=mode,
                only_need_context=True,
            )
            references = list(result.references or [])
            raw = dict(result.raw) if isinstance(result.raw, dict) else {}
            raw["evaluation_calls"] = _component_calls(timing)
            raw["evaluation_references"] = [item.model_dump() if hasattr(item, "model_dump") else item for item in references]
            return result.model_copy(update={"raw": raw})
        parts: list[str] = []
        references: list[Any] = []
        async for event in executor.stream(
            plan.nodes,
            user_query=question.question,
            timing=timing,
        ):
            if event.get("response"):
                parts.append(str(event["response"]))
            references.extend(event.get("references") or [])
        return QueryResult(
            response="\n\n".join(parts),
            mode=QueryMode.MIX,
            references=[],
            raw={
                "evaluation_references": references,
                "evaluation_calls": _component_calls(timing),
            },
        )

    return await _measured("agentic", _answer)


async def judge_pair(
    llm: LLMService,
    question: EvalQuestion,
    native: SystemRun,
    agentic: SystemRun,
    *,
    model: str,
) -> tuple[CombinedJudgment | None, list[TokenUsage]]:
    """Judge both presentation orders with the same model and prompt."""
    begin_token_usage()
    try:
        first = await _one_judgment(
            llm,
            question.question,
            agentic.answer,
            native.answer,
            agentic_first=True,
            model=model,
        )
        second = await _one_judgment(
            llm,
            question.question,
            native.answer,
            agentic.answer,
            agentic_first=False,
            model=model,
        )
    finally:
        usages = take_token_usage()
    if first is None or second is None:
        return None, usages
    return combine_judgments(first, second), usages


async def _one_judgment(
    llm: LLMService,
    question: str,
    answer_1: str,
    answer_2: str,
    *,
    agentic_first: bool,
    model: str,
) -> Judgment | None:
    raw = await llm.complete_prompt(
        render_judge_prompt(question, answer_1, answer_2),
        system_prompt=JUDGE_SYSTEM_PROMPT,
        model=model,
        response_format={"type": "json_object"},
        reasoning_effort="none",
    )
    if not isinstance(raw, str):
        return None
    return parse_judgment(raw, agentic_first=agentic_first)


async def _measured(label: str, call: Callable[[], Awaitable[Any]]) -> SystemRun:
    begin_token_usage()
    _retrieval_count.set([0])
    started = time.perf_counter()
    try:
        result = await call()
    except (LightRAGAdapterError, LLMServiceError, RuntimeError, OSError) as exc:
        return SystemRun(
            answer="",
            references=[],
            latency_s=time.perf_counter() - started,
            retrievals=_take_retrievals(),
            usages=take_token_usage(),
            error=f"{label}: {exc}",
        )
    latency = time.perf_counter() - started
    usages = take_token_usage()
    retrievals = _take_retrievals()
    references, calls = _references_and_calls(result)
    return SystemRun(
        answer=getattr(result, "response", "") or "",
        references=references,
        latency_s=latency,
        retrievals=retrievals,
        usages=usages,
        calls_by_component=calls,
    )


def _references_and_calls(result: Any) -> tuple[list[Any], dict[str, int]]:
    raw = getattr(result, "raw", None)
    if isinstance(raw, dict) and "evaluation_calls" in raw:
        return list(raw.get("evaluation_references") or []), dict(raw["evaluation_calls"])
    references = getattr(result, "references", None) or []
    return list(references), {"query": 1}


def _component_calls(timing: QueryTiming) -> dict[str, int]:
    counts: dict[str, int] = {}
    if timing.rewriter_calls_s:
        counts["rewriter"] = len(timing.rewriter_calls_s)
    if timing.decomposer_calls_s:
        counts["decomposer"] = len(timing.decomposer_calls_s)
    for call in timing.dag_calls:
        name = call.step.split(" ", 1)[0].split("(", 1)[0].strip() or call.node_type
        counts[name] = counts.get(name, 0) + 1
    return counts


def _take_retrievals() -> int:
    bucket = _retrieval_count.get()
    _retrieval_count.set(None)
    if not bucket:
        return 0
    return bucket[0]


async def _ingest_questions(adapter: Any, questions: list[EvalQuestion]) -> None:
    texts: list[str] = []
    sources: list[str] = []
    seen: set[str] = set()
    for question in questions:
        for index, passage in enumerate(question_documents(question), start=1):
            key = f"{passage.title}\n{passage.text}"
            if key in seen:
                continue
            seen.add(key)
            texts.append(passage.text)
            title = passage.title or f"{question.id}-{index}"
            sources.append(title)
    if texts:
        await adapter.ingest_texts(texts, sources=sources, wait=True)


async def load_isolated_config(working_dir: Path) -> LightRAGConfig:
    """Process configuration with the evaluation working directory and workspace."""
    return isolated_config(await get_config(), working_dir)
