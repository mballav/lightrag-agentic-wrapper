"""HTTP API tests. The lifespan uses a fake adapter so no backend is required."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from lightrag_wrapper.exceptions import LightRAGAdapterError
from lightrag_wrapper.models import (
    IngestResult,
    QueryMode,
    QueryPlanNode,
    QueryResult,
    TrackIngestStatus,
    TrackStatusBatchResult,
)
from lightrag_wrapper.query_decomposer.query_decomposer import Decomposition
from lightrag_wrapper.query_rewriter.query_rewriter import RewrittenQuery
from tests.conftest import make_config


class APIAdapter:
    def __init__(self, config):
        self.config = config
        self.closed = False
        self.fail = None
        self.last = None

    async def initialize(self):
        return None

    async def aclose(self):
        self.closed = True

    async def ingest_texts(self, texts, *, sources=None, wait=None):
        if self.fail == "adapter":
            raise LightRAGAdapterError("ingest down", status_code=503)
        if self.fail == "value":
            raise ValueError("bad texts")
        return IngestResult(
            status="success",
            message=f"Indexed {len(texts)}",
            track_id="track-text",
            track_ids=["track-text"],
            sources=list(sources or []),
        )

    async def ingest_files(self, paths, *, sources=None, wait=None):
        if self.fail == "adapter":
            raise LightRAGAdapterError("file down", status_code=None)
        if self.fail == "value":
            raise ValueError("bad files")
        return IngestResult(
            status="success",
            message="files",
            track_id="track-file",
            track_ids=["track-file"],
            sources=list(sources or []),
            details={"wait": wait},
        )

    async def ingest_status(self, track_ids):
        if self.fail == "adapter":
            raise LightRAGAdapterError("status down", status_code=502)
        if self.fail == "value":
            raise ValueError("bad ids")
        return TrackStatusBatchResult(
            tracks=[
                TrackIngestStatus(
                    track_id=track_id,
                    found=True,
                    status="success",
                    message="done",
                )
                for track_id in track_ids
            ]
        )

    async def query(self, query, **kwargs):
        self.last = (query, kwargs)
        if self.fail == "adapter":
            raise LightRAGAdapterError("query down", status_code=500)
        return QueryResult(response=f"ctx:{query}", mode=kwargs["mode"], references=[])


class APIRewriter:
    def __init__(self, llm):
        self.llm = llm

    async def rewrite(self, query, conversation_history=None):
        return RewrittenQuery(
            rewritten_query=query.strip(),
            high_level_keywords=["leave"],
            low_level_keywords=["Berlin"],
        )


class APIDecomposer:
    def __init__(self, llm, adapter):
        self.llm = llm
        self.adapter = adapter
        self.return_plan = True

    async def decompose(self, query, conversation_history=None):
        if not self.return_plan:
            return None
        return Decomposition(
            nodes=[
                QueryPlanNode(
                    Id=1,
                    nodeType="retrieve",
                    subquery=query.strip(),
                    depends_on=[],
                    isResultNode=True,
                )
            ],
            knowledge_base_metadata="{}",
        )


class APIRouter:
    def __init__(self, llm):
        self.llm = llm

    async def route(self, query):
        return QueryMode.LOCAL


@pytest.fixture
def api_app(monkeypatch, tmp_path):
    import lightrag_wrapper.app as app_module

    config = make_config(wrapper_upload_dir=str(tmp_path))
    holder = {}

    async def fake_get_config():
        return config

    def fake_create(cfg):
        holder["adapter"] = APIAdapter(cfg)
        return holder["adapter"]

    monkeypatch.setattr(app_module, "get_config", fake_get_config)
    monkeypatch.setattr(app_module, "create_adapter", fake_create)
    monkeypatch.setattr(app_module, "LLMService", lambda config: object())
    monkeypatch.setattr(app_module, "QueryRewriter", APIRewriter)
    monkeypatch.setattr(app_module, "QueryModeRouter", APIRouter)
    monkeypatch.setattr(app_module, "QueryDecomposer", APIDecomposer)

    with TestClient(app_module.app) as client:
        yield client, holder


def test_health_and_ingest_text(api_app, document_paths):
    client, holder = api_app
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["backend"] == "APIAdapter"
    assert body["initialized"] is True

    missing = client.post("/ingest", json={})
    assert missing.status_code == 422

    text = "E-ticket passenger itinerary"
    ingested = client.post("/ingest", json={"text": text, "source": document_paths[0].name})
    assert ingested.status_code == 200
    assert ingested.json()["track_id"] == "track-text"
    assert ingested.json()["sources"] == [document_paths[0].name]

    many = client.post(
        "/ingest",
        json={"texts": ["a", "b"], "sources": ["a.txt", "b.txt"], "wait": False},
    )
    assert many.status_code == 200

    holder["adapter"].fail = "adapter"
    assert client.post("/ingest", json={"text": "x"}).status_code == 503
    holder["adapter"].fail = "value"
    assert client.post("/ingest", json={"text": "x"}).status_code == 422
    holder["adapter"].fail = None


def test_ingest_files_and_status(api_app, document_paths):
    client, holder = api_app
    files = [
        ("files", (path.name, path.read_bytes(), "application/pdf")) for path in document_paths
    ]
    response = client.post("/ingest/file", files=files, params={"wait": "false"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["track_id"] == "track-file"
    assert len(payload["sources"]) == len(document_paths)

    again = client.post(
        "/ingest/file",
        files=[("file", (document_paths[0].name, b"duplicate", "text/plain"))],
    )
    assert again.status_code == 200
    assert again.json()["sources"][0] != document_paths[0].name or "_" in again.json()["sources"][0]

    empty = client.post("/ingest/file", files=[])
    assert empty.status_code == 422

    traversal = client.post(
        "/ingest/file",
        files=[("files", ("../secret.txt", b"hidden", "text/plain"))],
    )
    assert traversal.status_code == 200
    assert traversal.json()["sources"] == ["secret.txt"]

    holder["adapter"].fail = "adapter"
    failed = client.post(
        "/ingest/file", files=[("files", ("n.txt", b"n", "text/plain"))]
    )
    assert failed.status_code == 502
    holder["adapter"].fail = "value"
    assert (
        client.post("/ingest/file", files=[("files", ("n.txt", b"n", "text/plain"))]).status_code
        == 422
    )
    holder["adapter"].fail = None

    status = client.post("/ingest/status", json={"track_ids": ["track-file"]})
    assert status.status_code == 200
    assert status.json()["tracks"][0]["status"] == "success"
    holder["adapter"].fail = "adapter"
    assert client.post("/ingest/status", json={"track_ids": ["t"]}).status_code == 502
    holder["adapter"].fail = "value"
    assert client.post("/ingest/status", json={"track_ids": ["t"]}).status_code == 422


def test_query_pipeline(api_app):
    client, holder = api_app
    response = client.post(
        "/query",
        json={
            "query": "How many leave days does Berlin grant?",
            "conversation_history": [{"role": "user", "content": "leave policy"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    line = json.loads(response.text)
    assert line["id"] == 1
    assert "leave days" in line["response"]

    holder["adapter"].fail = "adapter"
    failed = client.post("/query", json={"query": "x"})
    assert failed.status_code == 200
    assert json.loads(failed.text) == {"error": "query down"}

    assert client.post("/query", json={"query": ""}).status_code == 422


async def test_a_failure_after_a_result_line_ends_the_stream_with_an_error():
    from lightrag_wrapper.app import _plan_lines
    from lightrag_wrapper.exceptions import LightRAGAdapterError

    class _Executor:
        async def stream(self, nodes, **kwargs):
            yield {"id": 1, "response": "first", "references": []}
            raise LightRAGAdapterError("query down", status_code=500)

    lines = [
        json.loads(line)
        async for line in _plan_lines(
            _Executor(),
            [],
            user_query="question",
            conversation_history=None,
            high_level_keywords=None,
            low_level_keywords=None,
        )
    ]
    assert lines == [
        {"id": 1, "response": "first", "references": []},
        {"error": "query down"},
    ]


def test_query_without_a_plan_retrieves_context_only(api_app):
    import lightrag_wrapper.app as app_module

    client, holder = api_app
    app_module.get_query_decomposer().return_plan = False
    response = client.post("/query", json={"query": "How many leave days?"})
    assert response.status_code == 200
    line = json.loads(response.text)
    assert line["id"] is None
    assert "How many leave days?" in line["response"]
    query, kwargs = holder["adapter"].last
    assert query == "How many leave days?"
    assert kwargs["only_need_context"] is True
    assert kwargs["high_level_keywords"] == ["leave"]
    assert kwargs["low_level_keywords"] == ["Berlin"]


def test_query_without_a_plan_still_returns_an_http_error(api_app):
    import lightrag_wrapper.app as app_module

    client, holder = api_app
    app_module.get_query_decomposer().return_plan = False
    holder["adapter"].fail = "adapter"
    response = client.post("/query", json={"query": "How many leave days?"})
    assert response.status_code == 500


def test_dependency_accessors_after_startup(api_app):
    import lightrag_wrapper.app as app_module

    assert app_module.get_adapter() is api_app[1]["adapter"]
    assert app_module.get_settings().lightrag_mode == "embedded"
    assert app_module.get_query_rewriter() is not None
    assert app_module.get_query_router() is not None
    assert app_module.get_query_decomposer() is not None
    assert app_module.get_query_executor() is not None
    assert api_app[1]["adapter"].closed is False
