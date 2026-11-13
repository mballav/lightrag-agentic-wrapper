"""Run an evaluation: `python -m lightrag_wrapper.evaluation`."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from lightrag_wrapper.evaluation.datasets import download_dragonball, load_dataset, select_questions
from lightrag_wrapper.evaluation.pricing import DEFAULT_PRICES_PATH, load_prices
from lightrag_wrapper.evaluation.report import write_run
from lightrag_wrapper.evaluation.runner import compare_questions, load_isolated_config
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
    parser.add_argument("--category", default=os.getenv("EVAL_CATEGORY"))
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
    return parser


async def _main(args: argparse.Namespace) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["QUERY_TIMES_LOG"] = str(output_dir / "times.log")
    data_path = await _dataset_path(args.dataset, args.data)
    questions = select_questions(
        load_dataset(args.dataset, data_path),
        category=args.category,
        limit=args.limit,
        seed=args.seed,
    )
    if not questions:
        raise SystemExit("No questions matched the dataset, category, and limit.")
    config = await load_isolated_config(Path(args.working_dir))
    configure_logging(config.log_level)
    comparisons = await compare_questions(
        questions,
        config=config,
        judge_model=args.judge_model,
        native_mode=QueryMode(args.native_mode),
        concurrency=args.concurrency,
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


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


if __name__ == "__main__":
    main()
