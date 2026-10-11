"""Build the document catalog passed to the query decomposer."""

from __future__ import annotations

from typing import Any

from lightrag_wrapper.models import DocumentIngestStatus

CATALOG_PAGE_SIZE = 50
CATALOG_MAX_DOCUMENTS = 100


def build_document_catalog(
    documents: list[DocumentIngestStatus],
    *,
    processed_total: int,
    status_counts: dict[str, int],
) -> dict[str, Any]:
    """Shape processed-document rows into the decomposer metadata payload."""
    listed = documents[:CATALOG_MAX_DOCUMENTS]
    total = max(processed_total, len(listed))
    complete = total <= len(listed)
    if complete:
        policy = "All processed documents are listed."
    else:
        policy = (
            f"{len(listed)} most recently updated processed documents. "
            "Other processed documents exist and are searchable by semantic retrieval."
        )
    return {
        "catalog_complete": complete,
        "processed_document_count": total,
        "listed_document_count": len(listed),
        "listing_policy": policy,
        "status_counts": status_counts,
        "documents": [_public_document(doc) for doc in listed],
    }


def unavailable_catalog(reason: str) -> dict[str, Any]:
    return {
        "catalog_complete": False,
        "processed_document_count": None,
        "listed_document_count": 0,
        "listing_policy": "Document metadata could not be loaded.",
        "error": reason,
        "documents": [],
    }


def coerce_status_counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    counts: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        counts[str(key)] = value
    return counts


def _public_document(doc: DocumentIngestStatus) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": doc.id,
        "file_path": doc.file_path,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "content_length": doc.content_length,
        "chunks_count": doc.chunks_count,
        "preview": doc.content_summary,
    }
    if doc.metadata:
        item["metadata"] = doc.metadata
    return {key: value for key, value in item.items() if value is not None}
