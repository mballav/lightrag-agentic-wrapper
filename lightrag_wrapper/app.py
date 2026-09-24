"""Minimal HTTP API that exposes the LightRAG adapter."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from lightrag_wrapper.adapter import LightRAGAdapter, create_adapter
from lightrag_wrapper.adapter._sources import unique_destination
from lightrag_wrapper.adapter._status import MAX_TRACK_IDS
from lightrag_wrapper.config import LightRAGConfig, get_config
from lightrag_wrapper.exceptions import LightRAGAdapterError
from lightrag_wrapper.llm import LLMService
from lightrag_wrapper.models import (
    IngestResult,
    QueryResult,
    TrackStatusBatchResult,
)
from lightrag_wrapper.query_rewriter import QueryRewriter
from lightrag_wrapper.router import QueryModeRouter

logger = logging.getLogger(__name__)


class IngestTextRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    source: str | None = None
    sources: list[str] | None = None
    wait: bool | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    conversation_history: list[dict[str, str]] | None = None


class TrackStatusRequest(BaseModel):
    track_ids: list[str] = Field(
        min_length=1,
        max_length=MAX_TRACK_IDS,
        description="LightRAG track IDs returned by ingest endpoints.",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = await get_config()
    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    adapter = create_adapter(config)
    llm = LLMService(config)
    rewriter = QueryRewriter(llm)
    router = QueryModeRouter(llm)
    logger.info(
        "Starting LightRAG adapter in %s mode (APP_ENV=%s)",
        config.lightrag_mode,
        config.app_env,
    )
    await adapter.initialize()
    app.state.adapter = adapter
    app.state.config = config
    app.state.llm = llm
    app.state.query_rewriter = rewriter
    app.state.query_router = router
    try:
        yield
    finally:
        await adapter.aclose()


app = FastAPI(
    title="LightRAG Agentic Wrapper",
    description="Adapter API over embedded or standalone LightRAG.",
    lifespan=lifespan,
)


def get_adapter() -> LightRAGAdapter:
    return app.state.adapter


def get_settings() -> LightRAGConfig:
    return app.state.config


def get_query_rewriter() -> QueryRewriter:
    return app.state.query_rewriter


def get_query_router() -> QueryModeRouter:
    return app.state.query_router


@app.get("/health")
async def health(
    adapter: LightRAGAdapter = Depends(get_adapter),
    config: LightRAGConfig = Depends(get_settings),
) -> dict[str, Any]:
    return {
        "status": "ok",
        "mode": config.lightrag_mode,
        "app_env": config.app_env,
        "query_mode": config.lightrag_query_mode.value,
        "server_url": config.lightrag_server_url if config.is_server_mode else None,
        "working_dir": config.working_dir if config.is_embedded_mode else None,
        "storage": {
            "kv": config.lightrag_kv_storage,
            "vector": config.lightrag_vector_storage,
            "graph": config.lightrag_graph_storage,
            "doc_status": config.lightrag_doc_status_storage,
        },
        "initialized": True,
        "backend": adapter.__class__.__name__,
    }


@app.post("/ingest", response_model=IngestResult)
async def ingest(
    body: IngestTextRequest,
    adapter: LightRAGAdapter = Depends(get_adapter),
) -> IngestResult:
    texts = body.texts or ([body.text] if body.text else None)
    if not texts:
        raise HTTPException(status_code=422, detail="Provide `text` or `texts`.")
    sources = body.sources
    if sources is None and body.source is not None:
        sources = [body.source]
    try:
        return await adapter.ingest_texts(texts, sources=sources, wait=body.wait)
    except LightRAGAdapterError as exc:
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/ingest/file", response_model=IngestResult)
async def ingest_file(
    files: list[UploadFile] | None = File(default=None),
    file: list[UploadFile] | None = File(default=None),
    wait: bool | None = Query(default=None),
    adapter: LightRAGAdapter = Depends(get_adapter),
    config: LightRAGConfig = Depends(get_settings),
) -> IngestResult:
    uploads = list(files or []) + list(file or [])
    if not uploads:
        raise HTTPException(
            status_code=422,
            detail="Provide one or more files via the `files` form field.",
        )

    target_dir = Path(config.wrapper_upload_dir)
    await aiofiles.os.makedirs(target_dir, exist_ok=True)

    destinations: list[Path] = []
    names: list[str] = []
    try:
        for upload in uploads:
            filename = Path(upload.filename or "upload.bin").name or "upload.bin"
            destination = await unique_destination(target_dir, filename)
            async with aiofiles.open(destination, "wb") as handle:
                await handle.write(await upload.read())
            destinations.append(destination)
            names.append(destination.name)
        return await adapter.ingest_files(destinations, sources=names, wait=wait)
    except LightRAGAdapterError as exc:
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        for upload in uploads:
            await upload.close()


@app.post("/ingest/status", response_model=TrackStatusBatchResult)
async def ingest_status(
    body: TrackStatusRequest,
    adapter: LightRAGAdapter = Depends(get_adapter),
) -> TrackStatusBatchResult:
    try:
        return await adapter.ingest_status(body.track_ids)
    except LightRAGAdapterError as exc:
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResult)
async def query(
    body: QueryRequest,
    adapter: LightRAGAdapter = Depends(get_adapter),
    rewriter: QueryRewriter = Depends(get_query_rewriter),
    router: QueryModeRouter = Depends(get_query_router),
) -> QueryResult:
    try:
        rewritten = await rewriter.rewrite(body.query, body.conversation_history)
        mode = await router.route(rewritten.rewritten_query)
        return await adapter.query(
            rewritten.rewritten_query,
            mode=mode,
            only_need_context=True,
            user_prompt=None,
            conversation_history=body.conversation_history,
            include_references=True,
            high_level_keywords=rewritten.high_level_keywords,
            low_level_keywords=rewritten.low_level_keywords,
        )
    except LightRAGAdapterError as exc:
        raise HTTPException(status_code=exc.status_code or 502, detail=str(exc)) from exc
