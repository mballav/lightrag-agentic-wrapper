"""End-to-end ingestion of the fixture corpus through both adapters and the API."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from lightrag_wrapper.adapter.embedded import EmbeddedLightRAGAdapter
from lightrag_wrapper.adapter.server import ServerLightRAGAdapter
from lightrag_wrapper.exceptions import LightRAGAdapterError
from lightrag_wrapper.models import QueryMode
from tests.conftest import make_config
from tests.unit.test_embedded import FakeRAG, QueryParam


async def test_embedded_rejects_pdf_fixtures(tmp_path, monkeypatch, document_paths):
    import lightrag

    holder = {}

    class LightRAG(FakeRAG):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            holder["rag"] = self

    monkeypatch.setattr(lightrag, "LightRAG", LightRAG)
    monkeypatch.setattr(lightrag, "QueryParam", QueryParam)
    monkeypatch.setattr(lightrag.utils, "setup_logger", lambda *a, **k: None)

    adapter = EmbeddedLightRAGAdapter(make_config(working_dir=str(tmp_path / "rag")))
    async with adapter:
        with pytest.raises(LightRAGAdapterError, match="UTF-8") as exc:
            await adapter.ingest_files(
                document_paths,
                sources=[path.name for path in document_paths],
                wait=True,
            )
        assert document_paths[0].name in str(exc.value)
        assert holder["rag"].inserted == []

        holder["rag"].query_result = {
            "response": "e-ticket",
            "references": [
                {
                    "reference_id": "1",
                    "file_path": document_paths[0].name,
                    "content": ["itinerary"],
                }
            ],
        }
        answer = await adapter.query(
            "What is the passenger itinerary?",
            mode=QueryMode.MIX,
            include_references=True,
        )
        assert document_paths[0].name in answer.references[0].file_path


async def test_server_ingests_fixture_documents(tmp_path, monkeypatch, document_paths):
    uploads = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/documents/upload":
            uploads.append(request.content)
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "message": "uploaded",
                    "track_id": f"track-{len(uploads)}",
                },
            )
        if path.startswith("/documents/track_status/"):
            return httpx.Response(
                200,
                json={
                    "documents": [
                        {
                            "id": "doc",
                            "status": "processed",
                            "file_path": "fixture",
                            "chunks_count": 1,
                        }
                    ]
                },
            )
        return httpx.Response(404, json={"detail": "no"})

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        real(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)

    async def _sleep(*_a, **_k):
        return None

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    adapter = ServerLightRAGAdapter(
        make_config(lightrag_mode="server", lightrag_ingest_wait=True)
    )
    async with adapter:
        result = await adapter.ingest_files(document_paths)
        assert result.status == "success"
        assert len(result.track_ids) == len(document_paths)
        assert len(uploads) == len(document_paths)
        for path, body in zip(document_paths, uploads):
            assert path.read_bytes() in body
        status = await adapter.ingest_status(result.track_ids)
        assert all(item.status == "success" for item in status.tracks)


def test_api_ingests_fixture_documents(monkeypatch, tmp_path, document_paths):
    import lightrag_wrapper.app as app_module

    seen = {}

    class RecordingAdapter:
        def __init__(self, config):
            self.config = config

        async def initialize(self):
            return None

        async def aclose(self):
            return None

        async def ingest_files(self, paths, *, sources=None, wait=None):
            from lightrag_wrapper.models import IngestResult

            bodies = [path.read_bytes() for path in paths]
            seen["bodies"] = bodies
            seen["sources"] = list(sources or [])
            return IngestResult(
                status="success",
                message="indexed fixtures",
                track_id="e2e-track",
                track_ids=["e2e-track"],
                sources=list(sources or []),
            )

        async def ingest_status(self, track_ids):
            from lightrag_wrapper.models import TrackIngestStatus, TrackStatusBatchResult

            return TrackStatusBatchResult(
                tracks=[
                    TrackIngestStatus(
                        track_id=track_id,
                        found=True,
                        status="success",
                        message="Indexed fixture documents.",
                        total_count=len(document_paths),
                    )
                    for track_id in track_ids
                ]
            )

        async def ingest_texts(self, texts, **kwargs):
            raise AssertionError("file ingest should be used")

        async def query(self, query, **kwargs):
            raise AssertionError("not this test")

    config = make_config(wrapper_upload_dir=str(tmp_path / "uploads"))

    async def fake_get_config():
        return config

    monkeypatch.setattr(app_module, "get_config", fake_get_config)
    monkeypatch.setattr(app_module, "create_adapter", lambda cfg: RecordingAdapter(cfg))
    monkeypatch.setattr(app_module, "LLMService", lambda config: object())
    monkeypatch.setattr(app_module, "QueryRewriter", lambda llm: object())
    monkeypatch.setattr(app_module, "QueryModeRouter", lambda llm: object())

    with TestClient(app_module.app) as client:
        files = [
            ("files", (path.name, path.read_bytes(), "application/pdf"))
            for path in document_paths
        ]
        response = client.post("/ingest/file", files=files)
        assert response.status_code == 200
        assert response.json()["track_id"] == "e2e-track"
        assert seen["sources"] == [path.name for path in document_paths]
        for path, body in zip(document_paths, seen["bodies"]):
            assert body == path.read_bytes()
            assert body.startswith(b"%PDF")

        status = client.post("/ingest/status", json={"track_ids": ["e2e-track"]})
        assert status.status_code == 200
        assert status.json()["tracks"][0]["total_count"] == len(document_paths)
