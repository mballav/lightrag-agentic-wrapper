"""Shared data models used by both adapter backends."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryMode(str, Enum):
    """LightRAG retrieval modes (default is mix)."""

    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    NAIVE = "naive"
    MIX = "mix"
    BYPASS = "bypass"


class Reference(BaseModel):
    reference_id: str | None = None
    file_path: str | None = None
    content: list[str] | None = None


class IngestResult(BaseModel):
    status: Literal["success", "pending", "failure"]
    message: str
    track_id: str | None = None
    track_ids: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestStatus(BaseModel):
    """Per-document ingest state for one LightRAG track_id."""

    id: str
    status: str
    file_path: str | None = None
    content_summary: str | None = None
    content_length: int | None = None
    chunks_count: int | None = None
    error_msg: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    track_id: str | None = None
    metadata: dict[str, Any] | None = None


class TrackIngestStatus(BaseModel):
    """Aggregated ingest state for a single track_id."""

    track_id: str
    found: bool
    status: Literal["success", "pending", "failure", "not_found", "error"]
    message: str
    total_count: int = 0
    status_summary: dict[str, int] = Field(default_factory=dict)
    documents: list[DocumentIngestStatus] = Field(default_factory=list)
    error: str | None = None


class TrackStatusBatchResult(BaseModel):
    tracks: list[TrackIngestStatus]


class QueryResult(BaseModel):
    response: str
    mode: QueryMode
    references: list[Reference] | None = None
    response_time: float | None = None
    raw: Any = Field(default=None, exclude=True)
