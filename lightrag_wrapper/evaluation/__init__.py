"""Compare native LightRAG with the agentic pipeline."""

from lightrag_wrapper.evaluation.datasets import EvalQuestion, load_dataset, select_questions
from lightrag_wrapper.evaluation.runner import compare_questions

__all__ = [
    "EvalQuestion",
    "compare_questions",
    "load_dataset",
    "select_questions",
]
