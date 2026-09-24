from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from lightrag_wrapper.adapter._status import (
    MAX_TRACK_IDS,
    build_track_status,
    coerce_document_payloads,
    document_is_in_progress,
    document_status_from_payload,
    document_status_from_sdk,
    normalize_doc_status,
    normalize_track_ids,
)
from lightrag_wrapper.models import DocumentIngestStatus


class _Status:
    def __init__(self, value):
        self.value = value


@dataclass
class Doc:
    id: str
    status: str
    file_path: str | None = None
    content_summary: str | None = None
    content_length: int | None = None
    chunks_count: int | None = None
    error_msg: str | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    track_id: str | None = None
    metadata: dict | None = None


class ObjDoc:
    id = "obj"
    doc_id = None
    status = "processed"
    file_path = "  "
    content_summary = "sum"
    content_length = "12"
    chunks_count = "nope"
    error_msg = None
    error = "err"
    created_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    updated_at = ""
    track_id = "t"
    metadata = {"keep": 1, "smartheading_llm_cache_ids": ["x"]}


def test_normalize_track_ids():
    assert normalize_track_ids([" a ", "", "a", "b"]) == ["a", "b"]
    with pytest.raises(ValueError, match="non-empty"):
        normalize_track_ids([" ", ""])
    with pytest.raises(ValueError, match="At most"):
        normalize_track_ids([str(i) for i in range(MAX_TRACK_IDS + 1)])


def test_normalize_doc_status_and_progress():
    assert normalize_doc_status(None) == "unknown"
    assert normalize_doc_status(_Status("DocStatus.PROCESSING")) == "processing"
    assert normalize_doc_status("  ") == "unknown"
    assert normalize_doc_status("Processed") == "processed"
    assert document_is_in_progress("")
    assert document_is_in_progress("queued")
    assert not document_is_in_progress("processed")


def test_coerce_and_payload_shapes():
    assert coerce_document_payloads(None) == []
    assert coerce_document_payloads("nope") == []
    assert coerce_document_payloads([{"id": "1"}, "skip"]) == [{"id": "1"}]
    docs = coerce_document_payloads(
        {"d1": {"status": "processed", "metadata": {}}, "d2": "failed"}
    )
    assert docs[0]["id"] == "d1"
    assert docs[1] == {"id": "d2", "status": "failed"}

    parsed = document_status_from_payload(
        {
            "doc_id": "abc",
            "status": "failed",
            "content_length": "",
            "chunks_count": 3,
            "error": "boom",
            "created_at": "2026-01-01",
            "updated_at": None,
            "metadata": "nope",
        }
    )
    assert parsed.id == "abc"
    assert parsed.content_length is None
    assert parsed.error_msg == "boom"
    assert parsed.metadata is None

    from_sdk = document_status_from_sdk("sdk-id", Doc(id="", status="processed"))
    assert from_sdk.id == "sdk-id"

    obj = document_status_from_payload(ObjDoc())
    assert obj.content_length == 12
    assert obj.chunks_count is None
    assert obj.created_at.startswith("2026-01-02")
    assert obj.updated_at is None
    assert obj.metadata == {"keep": 1}
    assert obj.file_path is None

    empty_meta = document_status_from_payload(
        {"id": "m", "status": "processed", "metadata": {"smartheading_llm_cache_ids": []}}
    )
    assert empty_meta.metadata is None
    unknown = document_status_from_payload({}, fallback_id=None)
    assert unknown.id == "unknown"


def test_build_track_status_branches():
    missing = build_track_status("t", [])
    assert missing.status == "not_found"

    pending = build_track_status(
        "t", [DocumentIngestStatus(id="1", status="processing")]
    )
    assert pending.status == "pending"

    failed = build_track_status(
        "t",
        [
            DocumentIngestStatus(id="1", status="failed"),
            DocumentIngestStatus(id="2", status="embedding_failed"),
        ],
    )
    assert failed.status == "failure"
    assert "failed" in failed.message

    partial = build_track_status(
        "t",
        [
            DocumentIngestStatus(id="1", status="processed"),
            DocumentIngestStatus(id="2", status="failed"),
        ],
    )
    assert "Partial" in partial.message

    success = build_track_status(
        "t",
        [
            DocumentIngestStatus(id="1", status="processed"),
            DocumentIngestStatus(id="2", status="success"),
        ],
    )
    assert success.status == "success"
    assert success.found

    mixed = build_track_status(
        "t", [DocumentIngestStatus(id="1", status="archived")]
    )
    assert mixed.status == "pending"
    assert "mixed" in mixed.message
