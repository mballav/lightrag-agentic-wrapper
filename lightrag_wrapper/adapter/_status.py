"""Helpers for document ingest track status."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from lightrag_wrapper.models import DocumentIngestStatus, TrackIngestStatus

IN_PROGRESS_STATUSES = frozenset(
    {
        "pending",
        "processing",
        "parsing",
        "analyzing",
        "queued",
        "preprocessed",
    }
)
TERMINAL_FAILURE_STATUSES = frozenset({"failed"})
TERMINAL_SUCCESS_STATUSES = frozenset({"processed", "success"})
INTERNAL_METADATA_KEYS = frozenset({"smartheading_llm_cache_ids"})
MAX_TRACK_IDS = 50
TRACK_STATUS_CONCURRENCY = 8


def normalize_track_ids(track_ids: Sequence[str]) -> list[str]:
    """Strip, drop empties, and de-duplicate while preserving order."""
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in track_ids:
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    if not normalized:
        raise ValueError("Provide one or more non-empty track_ids.")
    if len(normalized) > MAX_TRACK_IDS:
        raise ValueError(f"At most {MAX_TRACK_IDS} track_ids per request.")
    return normalized


def normalize_doc_status(value: Any) -> str:
    if value is None:
        return "unknown"
    text = value.value if hasattr(value, "value") else str(value)
    text = text.strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower() or "unknown"


def document_is_in_progress(status: str) -> bool:
    return (not status) or status in IN_PROGRESS_STATUSES


def coerce_document_payloads(raw: Any) -> list[dict[str, Any]]:
    """Accept LightRAG's list or legacy dict-of-docs track_status payload."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, Mapping):
        documents: list[dict[str, Any]] = []
        for key, value in raw.items():
            if isinstance(value, Mapping):
                item = dict(value)
                item.setdefault("id", str(key))
                documents.append(item)
            else:
                documents.append({"id": str(key), "status": value})
        return documents
    return []


def document_status_from_payload(
    raw: Any,
    *,
    fallback_id: str | None = None,
) -> DocumentIngestStatus:
    data = _as_mapping(raw)
    doc_id = str(
        data.get("id")
        or data.get("doc_id")
        or fallback_id
        or ""
    ).strip()
    metadata = _public_metadata(data.get("metadata"))
    return DocumentIngestStatus(
        id=doc_id or "unknown",
        status=normalize_doc_status(data.get("status")),
        file_path=_optional_str(data.get("file_path")),
        content_summary=_optional_str(data.get("content_summary")),
        content_length=_optional_int(data.get("content_length")),
        chunks_count=_optional_int(data.get("chunks_count")),
        error_msg=_optional_str(data.get("error_msg") or data.get("error")),
        created_at=_as_timestamp(data.get("created_at")),
        updated_at=_as_timestamp(data.get("updated_at")),
        track_id=_optional_str(data.get("track_id")),
        metadata=metadata,
    )


def document_status_from_sdk(
    doc_id: str,
    raw: Any,
) -> DocumentIngestStatus:
    return document_status_from_payload(raw, fallback_id=doc_id)


def build_track_status(
    track_id: str,
    documents: Sequence[DocumentIngestStatus],
) -> TrackIngestStatus:
    docs = list(documents)
    summary: dict[str, int] = {}
    for doc in docs:
        summary[doc.status] = summary.get(doc.status, 0) + 1

    if not docs:
        return TrackIngestStatus(
            track_id=track_id,
            found=False,
            status="not_found",
            message=f"No documents found for track {track_id}.",
            total_count=0,
            status_summary=summary,
            documents=[],
        )

    statuses = [doc.status for doc in docs]
    in_progress = any(document_is_in_progress(status) for status in statuses)
    failed = [
        status
        for status in statuses
        if status in TERMINAL_FAILURE_STATUSES or status.endswith("failed")
    ]
    if in_progress:
        track_status = "pending"
        message = f"Indexing in progress for track {track_id} ({summary})."
    elif failed and len(failed) == len(docs):
        track_status = "failure"
        message = f"Indexing failed for track {track_id} ({summary})."
    elif failed:
        track_status = "failure"
        message = f"Partial indexing failure for track {track_id} ({summary})."
    elif all(status in TERMINAL_SUCCESS_STATUSES for status in statuses):
        track_status = "success"
        message = f"Indexed {len(docs)} document(s) (track {track_id})."
    else:
        track_status = "pending"
        message = f"Indexing status mixed for track {track_id} ({summary})."

    return TrackIngestStatus(
        track_id=track_id,
        found=True,
        status=track_status,
        message=message,
        total_count=len(docs),
        status_summary=summary,
        documents=docs,
    )


def _as_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if is_dataclass(raw) and not isinstance(raw, type):
        return asdict(raw)
    data: dict[str, Any] = {}
    for key in (
        "id",
        "doc_id",
        "status",
        "file_path",
        "content_summary",
        "content_length",
        "chunks_count",
        "error_msg",
        "error",
        "created_at",
        "updated_at",
        "track_id",
        "metadata",
    ):
        if hasattr(raw, key):
            data[key] = getattr(raw, key)
    return data


def _public_metadata(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    cleaned = {
        str(key): item
        for key, item in value.items()
        if key not in INTERNAL_METADATA_KEYS
    }
    return cleaned or None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    text = str(value).strip()
    return text or None
