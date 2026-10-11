"""Standalone LightRAG server adapter (REST API)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiofiles
import aiofiles.ospath
import httpx

from lightrag_wrapper.adapter._catalog import coerce_status_counts
from lightrag_wrapper.adapter._sources import merge_ingest_results, normalize_sources
from lightrag_wrapper.adapter._status import (
    build_track_status,
    coerce_document_payloads,
    document_status_from_payload,
)
from lightrag_wrapper.adapter.base import LightRAGAdapter
from lightrag_wrapper.config import LightRAGConfig
from lightrag_wrapper.exceptions import LightRAGAdapterError
from lightrag_wrapper.log import get_logger
from lightrag_wrapper.models import (
    DocumentIngestStatus,
    IngestResult,
    QueryMode,
    QueryResult,
    Reference,
)

logger = get_logger(__name__)


class ServerLightRAGAdapter(LightRAGAdapter):
    """Talks to a running LightRAG API server over HTTP."""

    def __init__(self, config: LightRAGConfig) -> None:
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        if self._client is not None:
            return

        headers = {"Accept": "application/json"}
        if self.config.lightrag_api_key:
            headers["X-API-Key"] = self.config.lightrag_api_key

        timeout = httpx.Timeout(self.config.lightrag_server_timeout)
        self._client = httpx.AsyncClient(
            base_url=self.config.lightrag_server_url,
            headers=headers,
            timeout=timeout,
        )
        logger.info(
            "LightRAG server adapter ready",
            url=self.config.lightrag_server_url,
        )

    async def aclose(self) -> None:
        if self._client is None:
            return
        await self._client.aclose()
        self._client = None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise LightRAGAdapterError(
                "Adapter is not initialized. Call await adapter.initialize() first."
            )
        return self._client

    async def ingest_texts(
        self,
        texts: Sequence[str],
        *,
        sources: Sequence[str] | None = None,
        wait: bool | None = None,
    ) -> IngestResult:
        documents = [text.strip() for text in texts]
        if not documents or any(not text for text in documents):
            raise LightRAGAdapterError("ingest_texts requires non-empty document text")

        file_sources = normalize_sources(
            len(documents), list(sources) if sources else None
        )
        should_wait = self.config.lightrag_ingest_wait if wait is None else wait

        if len(documents) == 1:
            payload = {"text": documents[0], "file_source": file_sources[0]}
            data = await self._request("POST", "/documents/text", json=payload)
        else:
            payload = {"texts": documents, "file_sources": file_sources}
            data = await self._request("POST", "/documents/texts", json=payload)

        result = IngestResult(
            status="pending" if should_wait else "success",
            message=str(data.get("message") or "Documents accepted by LightRAG server."),
            track_id=data.get("track_id"),
            track_ids=[data["track_id"]] if data.get("track_id") else [],
            sources=file_sources,
            details=data if isinstance(data, dict) else {"raw": data},
        )

        if should_wait:
            return await self._wait_for_track(result)
        result.status = "success" if data.get("status") != "failure" else "failure"
        return result

    async def ingest_file(
        self,
        path: str | Path,
        *,
        source: str | None = None,
        wait: bool | None = None,
    ) -> IngestResult:
        file_path = Path(path)
        if not await aiofiles.ospath.isfile(file_path):
            raise LightRAGAdapterError(f"File not found: {file_path}")

        should_wait = self.config.lightrag_ingest_wait if wait is None else wait
        filename = source or file_path.name
        data = await self._upload_file(file_path, filename)

        result = IngestResult(
            status="pending",
            message=str(data.get("message") or f"Uploaded {filename}."),
            track_id=data.get("track_id"),
            track_ids=[data["track_id"]] if data.get("track_id") else [],
            sources=[filename],
            details=data if isinstance(data, dict) else {"raw": data},
        )
        if should_wait:
            return await self._wait_for_track(result)
        result.status = "success" if data.get("status") != "failure" else "failure"
        return result

    async def ingest_files(
        self,
        paths: Sequence[str | Path],
        *,
        sources: Sequence[str] | None = None,
        wait: bool | None = None,
    ) -> IngestResult:
        files, source_list = await self._normalize_file_batch(paths, sources)
        should_wait = self.config.lightrag_ingest_wait if wait is None else wait
        results: list[IngestResult] = []
        for index, file_path in enumerate(files):
            filename = source_list[index] if source_list else file_path.name
            results.append(
                await self.ingest_file(file_path, source=filename, wait=False)
            )
        if should_wait:
            waited = [await self._wait_for_track(item) for item in results]
            return merge_ingest_results(waited)
        return merge_ingest_results(results)

    async def _upload_file(self, file_path: Path, filename: str) -> dict[str, Any]:
        async with aiofiles.open(file_path, "rb") as handle:
            content = await handle.read()
        return await self._request(
            "POST",
            "/documents/upload",
            files={"file": (filename, content)},
        )

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
        resolved_mode = self.resolve_mode(mode)
        payload: dict[str, Any] = {
            "query": query.strip(),
            "mode": resolved_mode.value,
            "only_need_context": only_need_context,
            "include_references": include_references,
            "top_k": self.config.top_k,
            "chunk_top_k": self.config.chunk_top_k,
            "max_entity_tokens": self.config.max_entity_tokens,
            "max_relation_tokens": self.config.max_relation_tokens,
            "max_total_tokens": self.config.max_total_tokens,
        }
        if high_level_keywords is not None:
            payload["hl_keywords"] = high_level_keywords
        if low_level_keywords is not None:
            payload["ll_keywords"] = low_level_keywords
        if user_prompt is not None:
            payload["user_prompt"] = user_prompt
        if conversation_history:
            payload["conversation_history"] = conversation_history

        data = await self._request("POST", "/query", json=payload)
        references = None
        if include_references and isinstance(data, dict) and data.get("references"):
            references = [Reference.model_validate(item) for item in data["references"]]

        return QueryResult(
            response=str(data.get("response", "")),
            mode=resolved_mode,
            references=references,
            response_time=data.get("response_time"),
            raw=data,
        )

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        client = self._require_client()
        try:
            response = await client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise LightRAGAdapterError(
                f"LightRAG server request failed: {method} {path}: {exc}"
            ) from exc

        if response.status_code >= 400:
            detail = _response_detail(response)
            raise LightRAGAdapterError(
                f"LightRAG server returned {response.status_code} for "
                f"{method} {path}: {detail}",
                status_code=response.status_code,
            )

        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            raise LightRAGAdapterError(
                f"LightRAG server returned non-JSON for {method} {path}"
            ) from exc
        if not isinstance(payload, dict):
            return {"data": payload}
        return payload

    async def _wait_for_track(self, result: IngestResult) -> IngestResult:
        if not result.track_id:
            result.status = "success"
            return result

        deadline = asyncio.get_running_loop().time() + self.config.lightrag_ingest_timeout
        last_summary: dict[str, Any] = {}

        while True:
            data = await self._request(
                "GET",
                f"/documents/track_status/{quote(result.track_id, safe='')}",
            )
            documents = [
                document_status_from_payload(item)
                for item in coerce_document_payloads(data.get("documents"))
            ]
            track = build_track_status(result.track_id, documents)
            last_summary = track.status_summary or data.get("status_summary") or {}
            result.details = data

            if track.found and track.status != "pending":
                result.status = "failure" if track.status == "failure" else "success"
                result.message = track.message
                return result

            if asyncio.get_running_loop().time() >= deadline:
                result.status = "pending"
                result.message = (
                    f"Timed out waiting for track {result.track_id} "
                    f"(last status={last_summary}). Documents may still be indexing."
                )
                return result

            await asyncio.sleep(1.0)

    async def _fetch_track_documents(
        self,
        track_id: str,
    ) -> list[DocumentIngestStatus]:
        try:
            data = await self._request(
                "GET",
                f"/documents/track_status/{quote(track_id, safe='')}",
            )
        except LightRAGAdapterError as exc:
            if exc.status_code == 404:
                return []
            raise
        return [
            document_status_from_payload(item)
            for item in coerce_document_payloads(data.get("documents"))
        ]

    async def _fetch_processed_document_page(
        self,
        *,
        page: int,
        page_size: int,
    ) -> tuple[list[DocumentIngestStatus], int, dict[str, int], bool]:
        data = await self._request(
            "POST",
            "/documents/paginated",
            json={
                "status_filter": "processed",
                "page": page,
                "page_size": page_size,
                "sort_field": "updated_at",
                "sort_direction": "desc",
            },
        )
        documents = [
            document_status_from_payload(item)
            for item in coerce_document_payloads(data.get("documents"))
        ]
        documents = [doc for doc in documents if doc.status == "processed"]
        pagination = data.get("pagination") if isinstance(data.get("pagination"), dict) else {}
        total = int(pagination.get("total_count") or 0)
        has_next = bool(pagination.get("has_next"))
        if "has_next" not in pagination:
            has_next = page * page_size < total
        return documents, total, coerce_status_counts(data.get("status_counts")), has_next


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)[:500]
