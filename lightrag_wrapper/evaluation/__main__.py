"""Run an evaluation: `python -m lightrag_wrapper.evaluation`."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from lightrag_wrapper.evaluation.datasets import (
    download_dragonball,
    load_dataset,
    parse_question_ids,
    select_questions,
)
from lightrag_wrapper.evaluation.pricing import DEFAULT_PRICES_PATH, load_prices
from lightrag_wrapper.evaluation.report import write_run
from lightrag_wrapper.evaluation.runner import compare_questions, ingest_questions, load_isolated_config
from lightrag_wrapper.log import configure_logging
from lightrag_wrapper.models import QueryMode
from lightrag_wrapper.query_timing import times_log_path


def main() -> None:
    """CLI entry point."""
    asyncio.run(_main(build_parser().parse_args()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare native LightRAG with the agentic pipeline.")
    parser.add_argument("--dataset", default=os.getenv("EVAL_DATASET", "dragonball"))
    parser.add_argument("--data", default=os.getenv("EVAL_DATA"))
    parser.add_argument("--limit", type=int, default=_int_env("EVAL_LIMIT", 15))
    parser.add_argument("--seed", type=int, default=_int_env("EVAL_SEED", 0))
    parser.add_argument(
        "--repeat",
        type=int,
        default=_int_env("EVAL_REPEAT", 1),
        help="Run every selected question this many times after ingesting the passages once.",
    )
    parser.add_argument("--category", default=os.getenv("EVAL_CATEGORY"))
    parser.add_argument(
        "--question-id",
        default=os.getenv("EVAL_QUESTION_ID"),
        help="Comma-separated question ids to evaluate. Ignores --limit, --seed, and --category. Unknown ids are skipped.",
    )
    parser.add_argument("--concurrency", type=int, default=_int_env("EVAL_CONCURRENCY", 2))
    parser.add_argument("--native-mode", default=os.getenv("EVAL_NATIVE_MODE", "mix"))
    parser.add_argument("--judge-model", default=os.getenv("EVAL_JUDGE_MODEL", "gpt-6-luna"))
    parser.add_argument(
        "--prices",
        default=os.getenv("EVAL_PRICES", str(DEFAULT_PRICES_PATH)),
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("EVAL_OUTPUT_DIR", "./data/eval/runs"),
    )
    parser.add_argument(
        "--working-dir",
        default=os.getenv("EVAL_WORKING_DIR", "./data/eval/rag_storage"),
    )
    parser.add_argument("--workspace", default=os.getenv("EVAL_WORKSPACE", "evaluation"))
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        default=_bool_env("EVAL_DISABLE_RERANK", False),
        help="Query LightRAG with enable_rerank=False.",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        default=_bool_env("EVAL_INGEST_ONLY", False),
        help="Ingest every passage for the selected questions and skip evaluation.",
    )
    return parser


async def _main(args: argparse.Namespace) -> None:
    data_path = await _dataset_path(args.dataset, args.data)
    loaded = load_dataset(args.dataset, data_path)
    requested = parse_question_ids(args.question_id)
    questions = select_questions(
        loaded,
        category=args.category,
        limit=args.limit,
        seed=args.seed,
        question_id=args.question_id,
    )
    if requested:
        found = {item.id for item in questions}
        missing = [item for item in requested if item not in found]
        if missing:
            print(
                f"Skipped question ids not found in {args.dataset}: {', '.join(missing)}"
            )
    if not questions:
        if requested:
            raise SystemExit(f"No questions from the id list were found in {args.dataset}.")
        raise SystemExit("No questions matched the dataset, category, and limit.")
    config = await load_isolated_config(
        Path(args.working_dir),
        args.workspace,
        enable_rerank=False if args.disable_rerank else None,
    )
    configure_logging(config.log_level)
    if args.ingest_only:
        ingested = await ingest_questions(questions, config=config)
        print(f"Ingested {ingested} unique passages for {len(questions)} questions.")
        return
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QUERY_TIMES_LOG"] = str(output_dir / "times.log")
    comparisons = await compare_questions(
        questions,
        config=config,
        judge_model=args.judge_model,
        native_mode=QueryMode(args.native_mode),
        concurrency=args.concurrency,
        repeats=args.repeat,
    )
    output = write_run(
        comparisons,
        output_dir=output_dir,
        dataset=args.dataset,
        prices=load_prices(args.prices),
    )
    print(f"Wrote {output / 'report.md'}")
    print(f"Timing log: {times_log_path()}")


async def _dataset_path(dataset: str, data: str | None) -> Path:
    if data:
        path = Path(data)
        if not path.is_file():
            raise SystemExit(f"Dataset file not found: {path}")
        return path
    if dataset.strip().lower() != "dragonball":
        raise SystemExit("--data is required for this dataset.")
    destination = Path("./data/eval/datasets/dragonball_en.json")
    if destination.is_file():
        return destination
    try:
        return download_dragonball(destination)
    except Exception as exc:
        raise SystemExit(
            "DRAGONBall is not available. Pass --data or check the download source. "
            f"({exc.__class__.__name__})"
        ) from exc


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


if __name__ == "__main__":
    main()
