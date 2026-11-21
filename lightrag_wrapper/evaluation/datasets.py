"""Load evaluation questions into one shape."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

DATASETS = frozenset({"dragonball", "hotpotqa", "musique"})

_DRAGONBALL_QUERIES = (
    "https://raw.githubusercontent.com/OpenBMB/RAGEval/main/dragonball_dataset/dragonball_queries.jsonl"
)
_DRAGONBALL_DOCS = (
    "https://raw.githubusercontent.com/OpenBMB/RAGEval/main/dragonball_dataset/dragonball_docs.jsonl"
)


@dataclass(frozen=True)
class Passage:
    """One source passage. Supporting passages are the labeled evidence."""

    title: str
    text: str


@dataclass(frozen=True)
class EvalQuestion:
    """One normalized evaluation question."""

    id: str
    question: str
    reference_answer: str
    category: str | None
    supporting: tuple[Passage, ...]
    documents: tuple[Passage, ...]


def load_dataset(name: str, path: str | Path) -> list[EvalQuestion]:
    """Read one official dataset file and return normalized questions."""
    key = name.strip().lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Choose one of {sorted(DATASETS)}.")
    payload = _read_payload(Path(path))
    if key == "dragonball":
        return _load_dragonball(payload)
    if key == "hotpotqa":
        return _load_hotpotqa(payload)
    return _load_musique(payload)


def parse_question_ids(question_id: str | None) -> list[str]:
    """Split a comma-separated id list. Blank entries and repeats are dropped."""
    if not question_id or not question_id.strip():
        return []
    seen: list[str] = []
    for part in question_id.split(","):
        item = part.strip()
        if item and item not in seen:
            seen.append(item)
    return seen


def select_questions(
    questions: list[EvalQuestion],
    *,
    category: str | None = None,
    limit: int | None = None,
    seed: int = 0,
    question_id: str | None = None,
) -> list[EvalQuestion]:
    """Filter by category, then take a deterministic sample.

    A comma-separated question id list selects those questions, in that order,
    and ignores category, limit, and seed. Ids that are not in the dataset are skipped.
    """
    wanted = parse_question_ids(question_id)
    if wanted:
        by_id: dict[str, EvalQuestion] = {}
        for item in questions:
            by_id.setdefault(item.id, item)
        return [by_id[item] for item in wanted if item in by_id]
    chosen = list(questions)
    if category:
        folded = category.casefold()
        chosen = [item for item in chosen if (item.category or "").casefold() == folded]
    chosen.sort(key=lambda item: item.id)
    if limit is None or limit >= len(chosen):
        return chosen
    if limit < 1:
        return []
    picker = random.Random(seed)
    return sorted(picker.sample(chosen, limit), key=lambda item: item.id)


def download_dragonball(destination: str | Path) -> Path:
    """Download the official DRAGONBall queries and documents into one JSON file."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        queries = _fetch_jsonl(_DRAGONBALL_QUERIES)
        documents = _fetch_jsonl(_DRAGONBALL_DOCS)
    except Exception as exc:
        raise RuntimeError("DRAGONBall could not be downloaded.") from exc
    target.write_text(
        json.dumps(_combine_dragonball(queries, documents), ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def _fetch_jsonl(url: str) -> list[Any]:
    with urlopen(url, timeout=120) as response:
        text = response.read().decode("utf-8")
    rows: list[Any] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _combine_dragonball(queries: list[Any], documents: list[Any]) -> list[dict[str, Any]]:
    """Join query rows to the documents named by their evidence ids."""
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for document in documents:
        if not isinstance(document, dict):
            continue
        by_key[
            (
                _text(document.get("language")),
                _text(document.get("domain")),
                _text(document.get("doc_id")),
            )
        ] = document
    combined: list[dict[str, Any]] = []
    for row in queries:
        if not isinstance(row, dict):
            continue
        query = row.get("query") if isinstance(row.get("query"), dict) else {}
        truth = row.get("ground_truth") if isinstance(row.get("ground_truth"), dict) else {}
        language = _text(row.get("language"))
        domain = _text(row.get("domain"))
        documents_for_query = []
        for doc_id in truth.get("doc_ids") or []:
            document = by_key.get((language, domain, _text(doc_id)))
            if document is None:
                continue
            documents_for_query.append(
                {
                    "title": _text(document.get("company_name")) or _text(doc_id),
                    "content": _text(document.get("content")),
                }
            )
        evidence = truth.get("references") or truth.get("keypoints") or []
        combined.append(
            {
                "id": f"{language}-{domain}-{_text(query.get('query_id')) or len(combined) + 1}",
                "question": _text(query.get("content") or row.get("question")),
                "answer": _text(truth.get("content") or row.get("answer")),
                "domain": domain or None,
                "documents": documents_for_query,
                "evidence": evidence if isinstance(evidence, list) else [evidence],
            }
        )
    return combined


def _read_payload(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_dragonball(payload: Any) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for index, row in enumerate(_rows(payload), start=1):
        if "qa_pairs" in row and isinstance(row["qa_pairs"], list):
            documents = _passages(row.get("documents") or row.get("document") or row.get("doc"))
            category = _text(row.get("domain") or row.get("category"))
            for pair_index, pair in enumerate(row["qa_pairs"], start=1):
                questions.append(
                    _question(
                        f"{_text(row.get('id')) or index}-{pair_index}",
                        pair,
                        category=category or _text(pair.get("type")),
                        documents=documents,
                    )
                )
            continue
        questions.append(_question(str(index), row, category=_category(row), documents=()))
    return [item for item in questions if item.question]


def _load_hotpotqa(payload: Any) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for index, row in enumerate(_rows(payload), start=1):
        context = _hotpot_passages(row.get("context"))
        titles = {str(fact[0]) for fact in row.get("supporting_facts") or [] if fact}
        supporting = tuple(item for item in context if item.title in titles) or context
        questions.append(
            EvalQuestion(
                id=_text(row.get("_id") or row.get("id")) or str(index),
                question=_text(row.get("question")),
                reference_answer=_answer(row),
                category=_text(row.get("type") or row.get("level")),
                supporting=supporting,
                documents=context,
            )
        )
    return [item for item in questions if item.question]


def _load_musique(payload: Any) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for index, row in enumerate(_rows(payload), start=1):
        if row.get("answerable") is False:
            continue
        documents = _musique_passages(row.get("paragraphs"))
        supporting = tuple(
            item
            for item, paragraph in zip(documents, row.get("paragraphs") or [])
            if isinstance(paragraph, dict) and paragraph.get("is_supporting")
        )
        questions.append(
            EvalQuestion(
                id=_text(row.get("id")) or str(index),
                question=_text(row.get("question")),
                reference_answer=_answer(row),
                category=_text(row.get("id") or "").split("_", 1)[0] or None,
                supporting=supporting or documents,
                documents=documents,
            )
        )
    return [item for item in questions if item.question]


def _question(
    fallback_id: str,
    row: dict[str, Any],
    *,
    category: str | None,
    documents: tuple[Passage, ...],
) -> EvalQuestion:
    own_documents = documents or _passages(
        row.get("documents") or row.get("document") or row.get("doc") or row.get("context")
    )
    supporting = _passages(
        row.get("evidence")
        or row.get("supporting_evidence")
        or row.get("positive")
        or row.get("key_points")
        or row.get("references")
    )
    return EvalQuestion(
        id=_text(row.get("id") or row.get("_id")) or fallback_id,
        question=_text(row.get("question") or row.get("query")),
        reference_answer=_answer(row),
        category=category or _category(row),
        supporting=supporting or own_documents,
        documents=own_documents or supporting,
    )


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "questions", "examples", "qa_pairs"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def _passages(value: Any) -> tuple[Passage, ...]:
    if value is None:
        return ()
    if isinstance(value, str) and value.strip():
        return (Passage(title="", text=value.strip()),)
    if isinstance(value, dict):
        text = _text(value.get("content") or value.get("text") or value.get("paragraph"))
        if not text:
            return ()
        return (Passage(title=_text(value.get("title")), text=text),)
    if isinstance(value, list):
        passages: list[Passage] = []
        for item in value:
            passages.extend(_passages(item))
        return tuple(passages)
    return ()


def _hotpot_passages(value: Any) -> tuple[Passage, ...]:
    passages: list[Passage] = []
    for item in value or []:
        if isinstance(item, list) and len(item) >= 2:
            sentences = item[1] if isinstance(item[1], list) else [item[1]]
            text = " ".join(str(sentence) for sentence in sentences)
            passages.append(Passage(title=str(item[0]), text=text))
        else:
            passages.extend(_passages(item))
    return tuple(passages)


def _musique_passages(value: Any) -> tuple[Passage, ...]:
    passages: list[Passage] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        text = _text(item.get("paragraph_text") or item.get("text"))
        if text:
            passages.append(Passage(title=_text(item.get("title")), text=text))
    return tuple(passages)


def _answer(row: dict[str, Any]) -> str:
    value = row.get("answer", row.get("ground_truth", row.get("answers", row.get("reference"))))
    if isinstance(value, list):
        return _text(value[0] if value else "")
    return _text(value)


def _category(row: dict[str, Any]) -> str | None:
    return _text(row.get("domain") or row.get("category") or row.get("type")) or None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
