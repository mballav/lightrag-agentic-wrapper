"""Embedded LightRAG adapter (in-process SDK)."""

from __future__ import annotations

import inspect
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from lightrag_wrapper.adapter._catalog import coerce_status_counts
from lightrag_wrapper.adapter._sources import normalize_sources
from lightrag_wrapper.adapter._status import document_status_from_sdk
from lightrag_wrapper.adapter.base import LightRAGAdapter
from lightrag_wrapper.config import LightRAGConfig
from lightrag_wrapper.exceptions import LightRAGAdapterError
from lightrag_wrapper.log import get_logger
from lightrag_wrapper.llm import LLMService
from lightrag_wrapper.models import (
    DocumentIngestStatus,
    IngestResult,
    QueryMode,
    QueryResult,
    Reference,
)

logger = get_logger(__name__)


class EmbeddedLightRAGAdapter(LightRAGAdapter):
    """Talks to LightRAG through the Python SDK in the same process."""

    def __init__(self, config: LightRAGConfig) -> None:
        super().__init__(config)
        self._rag: Any = None
        self._llm: LLMService | None = None

    async def initialize(self) -> None:
        if self._rag is not None:
            return

        await aiofiles.os.makedirs(self.config.working_dir, exist_ok=True)
        try:
            self.config.require_embedded_bindings()
        except ValueError as exc:
            raise LightRAGAdapterError(str(exc)) from exc

        try:
            from lightrag import LightRAG
            from lightrag.utils import setup_logger
        except ImportError as exc:
            raise LightRAGAdapterError(
                "lightrag-hku is required for embedded mode. "
                "Install it from requirements.txt."
            ) from exc

        setup_logger("lightrag", level=self.config.log_level.upper())
        _export_storage_env(self.config)

        self._llm = LLMService(self.config)
        kwargs: dict[str, Any] = {
            "working_dir": self.config.working_dir,
            "llm_model_func": self._llm.as_lightrag_llm_func(),
            "llm_model_name": self.config.llm_model,
            "llm_model_max_async": self.config.max_async_llm,
            "embedding_func": self._llm.as_lightrag_embedding_func(
                embedding_dim=self.config.embedding_dim,
                max_token_size=self.config.embedding_token_limit,
                model_name=self.config.embedding_model,
            ),
            "embedding_batch_num": self.config.embedding_batch_num,
            "embedding_func_max_async": self.config.embedding_func_max_async,
            "kv_storage": self.config.lightrag_kv_storage,
            "vector_storage": self.config.lightrag_vector_storage,
            "graph_storage": self.config.lightrag_graph_storage,
            "doc_status_storage": self.config.lightrag_doc_status_storage,
            "max_parallel_insert": self.config.max_parallel_insert,
            "chunk_token_size": self.config.chunk_token_size,
            "chunk_overlap_token_size": self.config.chunk_overlap_token_size,
            "enable_llm_cache": self.config.enable_llm_cache,
            "addon_params": {"language": self.config.summary_language},
        }
        if self.config.workspace:
            kwargs["workspace"] = self.config.workspace

        rag = LightRAG(**_supported_kwargs(LightRAG, kwargs))
        await rag.initialize_storages()
        self._rag = rag
        logger.info(
            "Embedded LightRAG initialized",
            working_dir=self.config.working_dir,
            llm=self.config.llm_model,
            embedding=self.config.embedding_model,
        )

    async def aclose(self) -> None:
        if self._rag is None:
            self._llm = None
            return
        try:
            await self._rag.finalize_storages()
        finally:
            self._rag = None
            self._llm = None

    def _require_rag(self) -> Any:
        if self._rag is None:
            raise LightRAGAdapterError(
                "Adapter is not initialized. Call await adapter.initialize() first."
            )
        return self._rag

    async def ingest_texts(
        self,
        texts: Sequence[str],
        *,
        sources: Sequence[str] | None = None,
        wait: bool | None = None,
    ) -> IngestResult:
        del wait  # SDK insert is synchronous with respect to indexing.
        rag = self._require_rag()
        documents = [text.strip() for text in texts]
        if not documents or any(not text for text in documents):
            raise LightRAGAdapterError("ingest_texts requires non-empty document text")

        file_paths = normalize_sources(len(documents), list(sources) if sources else None)
        try:
            try:
                await rag.ainsert(list(documents), file_paths=file_paths)
            except TypeError:
                await rag.ainsert(list(documents))
        except Exception as exc:
            raise LightRAGAdapterError(f"Embedded ingest failed: {exc}") from exc

        return IngestResult(
            status="success",
            message=f"Indexed {len(documents)} document(s) with embedded LightRAG.",
            sources=file_paths,
        )

    async def ingest_file(
        self,
        path: str | Path,
        *,
        source: str | None = None,
        wait: bool | None = None,
    ) -> IngestResult:
        sources = [source] if source else None
        return await self.ingest_files([path], sources=sources, wait=wait)

    async def ingest_files(
        self,
        paths: Sequence[str | Path],
        *,
        sources: Sequence[str] | None = None,
        wait: bool | None = None,
    ) -> IngestResult:
        files, source_list = await self._normalize_file_batch(paths, sources)
        documents: list[str] = []
        names: list[str] = []
        for index, file_path in enumerate(files):
            try:
                async with aiofiles.open(file_path, encoding="utf-8") as handle:
                    documents.append(await handle.read())
            except UnicodeDecodeError as exc:
                raise LightRAGAdapterError(
                    f"Embedded mode can only ingest UTF-8 text files; "
                    f"use server mode to upload '{file_path.name}'. ({exc})"
                ) from exc
            if source_list:
                names.append(source_list[index] or file_path.name)
            else:
                names.append(file_path.name)
        return await self.ingest_texts(documents, sources=names, wait=wait)

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
        rag = self._require_rag()
        resolved_mode = self.resolve_mode(mode)

        try:
            from lightrag import QueryParam
        except ImportError as exc:
            raise LightRAGAdapterError("lightrag-hku is required for embedded mode.") from exc

        param_kwargs: dict[str, Any] = {
            "mode": resolved_mode.value,
            "only_need_context": only_need_context,
            "user_prompt": user_prompt,
            "conversation_history": conversation_history or [],
            "top_k": self.config.top_k,
            "chunk_top_k": self.config.chunk_top_k,
            "max_entity_tokens": self.config.max_entity_tokens,
            "max_relation_tokens": self.config.max_relation_tokens,
            "max_total_tokens": self.config.max_total_tokens,
        }
        if high_level_keywords is not None:
            param_kwargs["hl_keywords"] = high_level_keywords
        if low_level_keywords is not None:
            param_kwargs["ll_keywords"] = low_level_keywords
        param = QueryParam(**_supported_kwargs(QueryParam, param_kwargs))

        try:
            raw = await rag.aquery(query.strip(), param=param)
        except Exception as exc:
            raise LightRAGAdapterError(f"Embedded query failed: {exc}") from exc

        response_text, references = _normalize_query_output(raw, include_references)
        return QueryResult(
            response=response_text,
            mode=resolved_mode,
            references=references,
            raw=raw,
        )

    async def _fetch_track_documents(
        self,
        track_id: str,
    ) -> list[DocumentIngestStatus]:
        rag = self._require_rag()
        if not hasattr(rag, "aget_docs_by_track_id"):
            raise LightRAGAdapterError(
                "This LightRAG SDK build does not support aget_docs_by_track_id."
            )
        try:
            docs_by_id = await rag.aget_docs_by_track_id(track_id)
        except Exception as exc:
            raise LightRAGAdapterError(
                f"Embedded track status failed for {track_id}: {exc}"
            ) from exc

        if not docs_by_id:
            return []
        if not isinstance(docs_by_id, dict):
            raise LightRAGAdapterError(
                f"Unexpected track status payload for {track_id}."
            )
        return [
            document_status_from_sdk(str(doc_id), status)
            for doc_id, status in docs_by_id.items()
        ]

    async def _fetch_processed_document_page(
        self,
        *,
        page: int,
        page_size: int,
    ) -> tuple[list[DocumentIngestStatus], int, dict[str, int], bool]:
        rag = self._require_rag()
        doc_status = getattr(rag, "doc_status", None)
        if doc_status is None or not hasattr(doc_status, "get_docs_paginated"):
            raise LightRAGAdapterError(
                "This LightRAG SDK build does not support document listing."
            )
        try:
            from lightrag.base import DocStatus
        except ImportError as exc:
            raise LightRAGAdapterError(
                "lightrag-hku is required for embedded mode."
            ) from exc

        method = doc_status.get_docs_paginated
        parameters = inspect.signature(method).parameters
        call: dict[str, Any] = {
            "page": page,
            "page_size": page_size,
            "sort_field": "updated_at",
            "sort_direction": "desc",
        }
        if "status_filters" in parameters:
            call["status_filters"] = [DocStatus.PROCESSED]
        elif "status_filter" in parameters:
            call["status_filter"] = DocStatus.PROCESSED

        try:
            rows, total = await method(**call)
        except Exception as exc:
            raise LightRAGAdapterError(
                f"Embedded document listing failed: {exc}"
            ) from exc

        documents = [
            document_status_from_sdk(str(doc_id), status)
            for doc_id, status in rows
        ]
        documents = [doc for doc in documents if doc.status == "processed"]
        counts = await _status_counts(doc_status)
        processed_total = int(total)
        has_next = page * page_size < processed_total
        return documents, processed_total, counts, has_next


async def _status_counts(doc_status: Any) -> dict[str, int]:
    counter = getattr(doc_status, "get_all_status_counts", None) or getattr(
        doc_status, "get_status_counts", None
    )
    if counter is None:
        return {}
    try:
        raw = await counter()
    except Exception:
        logger.warning("Embedded document status counts failed", exc_info=True)
        return {}
    return coerce_status_counts(raw)


def _export_storage_env(config: LightRAGConfig) -> None:
    """Push storage connection settings into os.environ for the LightRAG SDK."""
    values: dict[str, str | None] = {
        "POSTGRES_HOST": config.postgres_host,
        "POSTGRES_PORT": str(config.postgres_port),
        "POSTGRES_USER": config.postgres_user,
        "POSTGRES_PASSWORD": config.postgres_password,
        "POSTGRES_DATABASE": config.postgres_database,
        "POSTGRES_MAX_CONNECTIONS": str(config.postgres_max_connections),
        "NEO4J_URI": config.neo4j_uri,
        "NEO4J_USERNAME": config.neo4j_username,
        "NEO4J_PASSWORD": config.neo4j_password,
        "NEO4J_DATABASE": config.neo4j_database,
        "MILVUS_URI": config.milvus_uri,
        "MILVUS_DB_NAME": config.milvus_db_name,
        "MILVUS_DEVICE": config.milvus_device,
        "MILVUS_USER": config.milvus_user,
        "MILVUS_PASSWORD": config.milvus_password,
    }
    for key, value in values.items():
        if value is not None:
            os.environ[key] = value


def _supported_kwargs(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs that the installed LightRAG version does not accept."""
    import inspect
    from dataclasses import fields, is_dataclass

    if is_dataclass(target):
        names = {item.name for item in fields(target)}
        return {key: value for key, value in kwargs.items() if key in names}

    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return kwargs

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    names = set(parameters) - {"self"}
    return {key: value for key, value in kwargs.items() if key in names}


def _normalize_query_output(
    raw: Any, include_references: bool
) -> tuple[str, list[Reference] | None]:
    references: list[Reference] | None = None

    if isinstance(raw, str):
        return raw, None

    if isinstance(raw, dict):
        response = raw.get("response") or raw.get("answer") or raw.get("content")
        if response is None:
            import json

            response = json.dumps(raw, ensure_ascii=False, default=str)
        if include_references and raw.get("references"):
            references = [Reference.model_validate(item) for item in raw["references"]]
        return str(response), references

    return str(raw), None
