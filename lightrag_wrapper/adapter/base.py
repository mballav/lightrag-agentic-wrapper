"""Abstract LightRAG adapter: ingest documents and query by mode."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

import aiofiles.ospath

from lightrag_wrapper.adapter._status import (
    TRACK_STATUS_CONCURRENCY,
    build_track_status,
    normalize_track_ids,
)
from lightrag_wrapper.config import LightRAGConfig
from lightrag_wrapper.exceptions import LightRAGAdapterError
from lightrag_wrapper.models import (
    DocumentIngestStatus,
    IngestResult,
    QueryMode,
    QueryResult,
    TrackIngestStatus,
    TrackStatusBatchResult,
)

logger = logging.getLogger(__name__)


class LightRAGAdapter(ABC):
    """Backend-agnostic LightRAG client (embedded SDK or HTTP server)."""

    def __init__(self, config: LightRAGConfig) -> None:
        self.config = config

    @abstractmethod
    async def initialize(self) -> None:
        """Open connections / initialize storage. Must be called before use."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release backend resources."""

    @abstractmethod
    async def ingest_texts(
        self,
        texts: Sequence[str],
        *,
        sources: Sequence[str] | None = None,
        wait: bool | None = None,
    ) -> IngestResult:
        """Index one or more text documents."""

    @abstractmethod
    async def ingest_file(
        self,
        path: str | Path,
        *,
        source: str | None = None,
        wait: bool | None = None,
    ) -> IngestResult:
        """Index a file from disk (text locally; upload to the server)."""

    @abstractmethod
    async def ingest_files(
        self,
        paths: Sequence[str | Path],
        *,
        sources: Sequence[str] | None = None,
        wait: bool | None = None,
    ) -> IngestResult:
        """Index multiple files from disk in one call."""

    @abstractmethod
    async def query(
        self,
        query: str,
        *,
        mode: QueryMode | str | None = None,
        only_need_context: bool = False,
        user_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        include_references: bool = True,
        high_level_keywords: list[str] | None = None,
        low_level_keywords: list[str] | None = None,
    ) -> QueryResult:
        """Run a retrieval query using one of LightRAG's query modes."""

    async def ingest_status(
        self,
        track_ids: Sequence[str],
    ) -> TrackStatusBatchResult:
        """Return per-document ingest status for each track_id.

        Unknown tracks are reported as `not_found`. A failure fetching one
        track does not fail the rest of the batch.
        """
        ids = normalize_track_ids(track_ids)
        semaphore = asyncio.Semaphore(TRACK_STATUS_CONCURRENCY)
        tracks = await asyncio.gather(
            *[self._ingest_status_one(track_id, semaphore) for track_id in ids]
        )
        return TrackStatusBatchResult(tracks=list(tracks))

    async def _ingest_status_one(
        self,
        track_id: str,
        semaphore: asyncio.Semaphore,
    ) -> TrackIngestStatus:
        async with semaphore:
            try:
                documents = await self._fetch_track_documents(track_id)
            except LightRAGAdapterError as exc:
                logger.warning("Track status failed for %s: %s", track_id, exc)
                return TrackIngestStatus(
                    track_id=track_id,
                    found=False,
                    status="error",
                    message=str(exc),
                    error=str(exc),
                )
            except Exception as exc:
                logger.exception("Unexpected track status error for %s", track_id)
                return TrackIngestStatus(
                    track_id=track_id,
                    found=False,
                    status="error",
                    message=f"Unexpected error fetching status for {track_id}.",
                    error=str(exc),
                )
            return build_track_status(track_id, documents)

    @abstractmethod
    async def _fetch_track_documents(
        self,
        track_id: str,
    ) -> list[DocumentIngestStatus]:
        """Load documents associated with one LightRAG track_id."""

    async def ingest_text(
        self,
        text: str,
        *,
        source: str | None = None,
        wait: bool | None = None,
    ) -> IngestResult:
        sources = [source] if source else None
        return await self.ingest_texts([text], sources=sources, wait=wait)

    async def _normalize_file_batch(
        self,
        paths: Sequence[str | Path],
        sources: Sequence[str] | None,
    ) -> tuple[list[Path], list[str] | None]:
        files = [Path(path) for path in paths]
        if not files:
            raise ValueError("ingest_files requires at least one file")
        missing: list[str] = []
        for path in files:
            if not await aiofiles.ospath.isfile(path):
                missing.append(str(path))
        if missing:
            raise LightRAGAdapterError(f"File not found: {', '.join(missing)}")
        if sources is None:
            return files, None
        source_list = list(sources)
        if len(source_list) != len(files):
            raise ValueError(
                f"sources length ({len(source_list)}) must match file count ({len(files)})"
            )
        return files, source_list

    def resolve_mode(self, mode: QueryMode | str | None) -> QueryMode:
        if mode is None:
            return self.config.lightrag_query_mode
        return QueryMode(mode)

    async def __aenter__(self) -> LightRAGAdapter:
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
