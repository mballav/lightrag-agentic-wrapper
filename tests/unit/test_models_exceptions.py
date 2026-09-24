import pytest
from pydantic import ValidationError

from lightrag_wrapper.exceptions import LLMServiceError, LightRAGAdapterError
from lightrag_wrapper.models import (
    DocumentIngestStatus,
    IngestResult,
    QueryMode,
    QueryResult,
    Reference,
    TrackIngestStatus,
    TrackStatusBatchResult,
)


def test_models_round_trip():
    result = IngestResult(
        status="success",
        message="ok",
        track_id="t1",
        sources=["a.txt"],
    )
    assert result.track_ids == []
    query = QueryResult(
        response="answer",
        mode=QueryMode.LOCAL,
        references=[Reference(reference_id="1", file_path="a.txt", content=["x"])],
        raw={"hidden": True},
    )
    dumped = query.model_dump()
    assert "raw" not in dumped
    assert QueryMode("mix") is QueryMode.MIX
    batch = TrackStatusBatchResult(
        tracks=[
            TrackIngestStatus(
                track_id="t1",
                found=True,
                status="success",
                message="done",
                documents=[DocumentIngestStatus(id="d1", status="processed")],
            )
        ]
    )
    assert batch.tracks[0].documents[0].id == "d1"


def test_query_request_validation_lives_on_enum():
    with pytest.raises(ValueError):
        QueryMode("bypass-nope")


def test_exceptions():
    adapter = LightRAGAdapterError("boom", status_code=503)
    assert adapter.status_code == 503
    assert str(adapter) == "boom"
    llm = LLMServiceError(
        "llm", status_code=429, operation="completion", retryable=True
    )
    assert llm.retryable
    assert llm.operation == "completion"
    with pytest.raises(ValidationError):
        IngestResult(status="nope", message="x")
