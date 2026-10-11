"""Abstract LightRAG adapter: ingest documents and query by mode."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import aiofiles.ospath

from lightrag_wrapper.adapter._catalog import (
    CATALOG_MAX_DOCUMENTS,
    CATALOG_PAGE_SIZE,
    build_document_catalog,
)
from lightrag_wrapper.adapter._status import (
    TRACK_STATUS_CONCURRENCY,
    build_track_status,
    normalize_track_ids,
)
from lightrag_wrapper.config import LightRAGConfig
from lightrag_wrapper.exceptions import LightRAGAdapterError
from lightrag_wrapper.log import get_logger
from lightrag_wrapper.models import (
    DocumentIngestStatus,
    IngestResult,
    QueryMode,
    QueryResult,
    TrackIngestStatus,
    TrackStatusBatchResult,
)

logger = get_logger(__name__)


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
                logger.warning(
                    "Track status failed",
                    track_id=track_id,
                    error=str(exc),
                )
                return TrackIngestStatus(
                    track_id=track_id,
                    found=False,
                    status="error",
                    message=str(exc),
                    error=str(exc),
                )
            except Exception as exc:
                logger.exception(
                    "Unexpected track status error",
                    track_id=track_id,
                )
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

    async def list_document_catalog(self) -> dict[str, Any]:
        """Return processed-document metadata for the query decomposer.

        Lists at most ``CATALOG_MAX_DOCUMENTS`` rows, newest update first.
        ``content_summary`` is exposed as ``preview`` because LightRAG stores
        the first characters of the file, not an abstract.
        """
        collected: list[DocumentIngestStatus] = []
        total = 0
        status_counts: dict[str, int] = {}
        page = 1
        while len(collected) < CATALOG_MAX_DOCUMENTS:
            documents, page_total, counts, has_next = (
                await self._fetch_processed_document_page(
                    page=page,
                    page_size=CATALOG_PAGE_SIZE,
                )
            )
            total = page_total
            if counts:
                status_counts = counts
            if not documents:
                break
            room = CATALOG_MAX_DOCUMENTS - len(collected)
            collected.extend(documents[:room])
            if len(collected) >= CATALOG_MAX_DOCUMENTS or not has_next:
                break
            page += 1
        return build_document_catalog(
            collected,
            processed_total=total,
            status_counts=status_counts,
        )

    @abstractmethod
    async def _fetch_processed_document_page(
        self,
        *,
        page: int,
        page_size: int,
    ) -> tuple[list[DocumentIngestStatus], int, dict[str, int], bool]:
        """Load one page of processed documents.

        Returns ``(documents, processed_total, status_counts, has_next)``.
        """

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
