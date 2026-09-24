"""Helpers shared by adapter backends."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import aiofiles.ospath

from lightrag_wrapper.models import IngestResult


def make_document_source(hint: str | None = None) -> str:
    """Build a unique, path-like source id that LightRAG will accept."""
    raw = Path(hint).name if hint else "document.txt"
    if not raw or raw in {".", ".."}:
        raw = "document.txt"
    if Path(raw).suffix == "":
        raw = f"{raw}.txt"
    return f"ingested/{uuid4().hex}_{raw}"


async def unique_destination(directory: Path, filename: str) -> Path:
    """Pick a non-colliding path under directory for filename."""
    safe = Path(filename).name or "upload.bin"
    if safe in {".", ".."}:
        safe = "upload.bin"
    candidate = directory / safe
    if not await aiofiles.ospath.exists(candidate):
        return candidate
    stem, suffix = Path(safe).stem, Path(safe).suffix
    index = 1
    while True:
        candidate = directory / f"{stem}_{index}{suffix}"
        if not await aiofiles.ospath.exists(candidate):
            return candidate
        index += 1


def normalize_sources(
    count: int,
    sources: list[str] | None,
    *,
    fallback_hint: str | None = None,
) -> list[str]:
    if sources is None:
        return [make_document_source(fallback_hint) for _ in range(count)]
    if len(sources) != count:
        raise ValueError(
            f"sources length ({len(sources)}) must match document count ({count})"
        )
    return [source or make_document_source(fallback_hint) for source in sources]


def merge_ingest_results(results: list[IngestResult]) -> IngestResult:
    """Combine per-file ingest results into one response."""
    if not results:
        raise ValueError("merge_ingest_results requires at least one result")
    if len(results) == 1:
        single = results[0]
        if not single.track_ids and single.track_id:
            single.track_ids = [single.track_id]
        return single

    sources: list[str] = []
    track_ids: list[str] = []
    uploads: list[dict] = []
    statuses: list[str] = []
    for item in results:
        sources.extend(item.sources)
        if item.track_id:
            track_ids.append(item.track_id)
        for track_id in item.track_ids:
            if track_id not in track_ids:
                track_ids.append(track_id)
        statuses.append(item.status)
        uploads.append(item.details or {"message": item.message})

    if all(status == "success" for status in statuses):
        status = "success"
    elif all(status == "failure" for status in statuses):
        status = "failure"
    elif "failure" in statuses:
        status = "failure"
    elif "pending" in statuses:
        status = "pending"
    else:
        status = "success"

    return IngestResult(
        status=status,
        message=f"Ingested {len(results)} file(s) (status={status}).",
        track_id=track_ids[0] if track_ids else None,
        track_ids=track_ids,
        sources=sources,
        details={"uploads": uploads, "status_summary": statuses},
    )
