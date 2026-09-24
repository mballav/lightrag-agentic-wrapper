import asyncio

import httpx
import pytest

from lightrag_wrapper.adapter.server import ServerLightRAGAdapter, _response_detail
from lightrag_wrapper.exceptions import LightRAGAdapterError
from tests.conftest import make_config


class Router:
    def __init__(self):
        self.calls = []
        self.track_docs = [
            {"id": "d1", "status": "processing", "file_path": "a.txt"},
            {"id": "d1", "status": "processed", "file_path": "a.txt"},
        ]
        self.fail_status = None
        self.stuck = False
        self.empty = False
        self.non_json = False
        self.list_body = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        path = request.url.path
        if self.non_json and path == "/query":
            return httpx.Response(200, content=b"not-json")
        if self.list_body and path == "/query":
            return httpx.Response(200, json=["x"])
        if self.empty and path == "/documents/text":
            return httpx.Response(200, content=b"")
        if path == "/documents/text":
            return httpx.Response(
                200, json={"status": "success", "message": "accepted", "track_id": "t-text"}
            )
        if path == "/documents/texts":
            return httpx.Response(
                200, json={"status": "failure", "message": "batch", "track_id": "t-batch"}
            )
        if path == "/documents/upload":
            return httpx.Response(
                200, json={"status": "success", "message": "uploaded", "track_id": "t-file"}
            )
        if path.startswith("/documents/track_status/"):
            if self.stuck:
                return httpx.Response(
                    200,
                    json={"documents": [{"id": "d", "status": "processing"}], "status_summary": {}},
                )
            if self.fail_status == 404:
                return httpx.Response(404, json={"detail": "missing"})
            if self.fail_status == 500:
                return httpx.Response(500, text="boom")
            if not self.track_docs:
                doc = {"id": "d", "status": "processed", "file_path": "a.txt"}
            else:
                doc = self.track_docs.pop(0)
            return httpx.Response(200, json={"documents": [doc], "status_summary": {}})
        if path == "/query":
            return httpx.Response(
                200,
                json={
                    "response": "from server",
                    "references": [{"reference_id": "1", "file_path": "a.txt"}],
                    "response_time": 0.2,
                },
            )
        if path == "/down":
            return httpx.Response(400, json={"detail": "bad"})
        return httpx.Response(404, json={"detail": "nope"})


@pytest.fixture
def router(monkeypatch):
    route = Router()
    transport = httpx.MockTransport(route)
    real = httpx.AsyncClient.__init__

    def patched(self, *args, **kwargs):
        kwargs["transport"] = transport
        real(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)
    return route


async def test_server_ingest_pipeline(tmp_path, router, document_paths, monkeypatch):
    async def _done(*_args, **_kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", _done)
    config = make_config(
        lightrag_mode="server",
        lightrag_api_key="secret",
        lightrag_ingest_wait=True,
        lightrag_ingest_timeout=30,
    )
    adapter = ServerLightRAGAdapter(config)
    await adapter.initialize()
    await adapter.initialize()
    assert adapter._client.headers["X-API-Key"] == "secret"

    text = "E-ticket passenger itinerary"
    one = await adapter.ingest_texts([text], sources=[document_paths[0].name], wait=True)
    assert one.status == "success"
    assert one.track_id == "t-text"

    no_wait = await adapter.ingest_texts(
        [text, text], sources=["a.txt", "b.txt"], wait=False
    )
    assert no_wait.status == "failure"

    uploaded = await adapter.ingest_files(document_paths, wait=True)
    assert uploaded.status == "success"
    assert len(uploaded.sources) == len(document_paths)

    pending = await adapter.ingest_file(document_paths[0], wait=False)
    assert pending.status == "success"

    queried = await adapter.query(
        " leave ",
        mode="hybrid",
        user_prompt="cite",
        conversation_history=[{"role": "user", "content": "hi"}],
        high_level_keywords=["leave"],
        low_level_keywords=["Berlin"],
    )
    assert queried.response == "from server"
    assert queried.references[0].reference_id == "1"
    assert queried.response_time == 0.2

    await adapter.aclose()
    await adapter.aclose()


async def test_server_errors(tmp_path, router, document_paths):
    adapter = ServerLightRAGAdapter(make_config(lightrag_mode="server", lightrag_ingest_wait=False))
    with pytest.raises(LightRAGAdapterError, match="not initialized"):
        await adapter.ingest_texts(["hello"])
    await adapter.initialize()

    with pytest.raises(LightRAGAdapterError, match="non-empty"):
        await adapter.ingest_texts([" "])
    with pytest.raises(LightRAGAdapterError, match="File not found"):
        await adapter.ingest_file(tmp_path / "missing.txt")

    with pytest.raises(LightRAGAdapterError, match="400"):
        await adapter._request("GET", "/down")

    router.non_json = True
    with pytest.raises(LightRAGAdapterError, match="non-JSON"):
        await adapter.query("q")
    router.non_json = False
    router.list_body = True
    listed = await adapter.query("q", include_references=False, user_prompt=None)
    assert listed.response == ""
    router.list_body = False

    router.empty = True
    empty = await adapter.ingest_texts(["body"], wait=False)
    assert empty.status == "success"
    router.empty = False

    def explode(request):
        raise httpx.ConnectError("refused", request=request)

    adapter._client._transport = httpx.MockTransport(explode)
    with pytest.raises(LightRAGAdapterError, match="request failed"):
        await adapter.ingest_texts(["body"], wait=False)


async def test_wait_timeout_and_missing_track(monkeypatch, router):
    async def _done(*_args, **_kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", _done)
    loop = asyncio.get_running_loop()
    ticks = {"n": 0}

    def fast_time():
        ticks["n"] += 1
        return 0 if ticks["n"] < 3 else 10_000

    monkeypatch.setattr(loop, "time", fast_time)
    adapter = ServerLightRAGAdapter(
        make_config(lightrag_mode="server", lightrag_ingest_timeout=1)
    )
    await adapter.initialize()
    router.stuck = True
    from lightrag_wrapper.models import IngestResult

    pending = await adapter._wait_for_track(
        IngestResult(status="pending", message="m", track_id="slow", sources=["a"])
    )
    assert pending.status == "pending"
    assert "Timed out" in pending.message

    skipped = await adapter._wait_for_track(
        IngestResult(status="pending", message="m", sources=["a"])
    )
    assert skipped.status == "success"

    router.stuck = False
    router.track_docs = [{"id": "d", "status": "failed"}]
    failed = await adapter._wait_for_track(
        IngestResult(status="pending", message="m", track_id="bad", sources=["a"])
    )
    assert failed.status == "failure"

    router.fail_status = 404
    assert await adapter._fetch_track_documents("gone") == []
    router.fail_status = 500
    with pytest.raises(LightRAGAdapterError):
        await adapter._fetch_track_documents("gone")


def test_response_detail_branches():
    json_resp = httpx.Response(400, json={"detail": "no"})
    assert _response_detail(json_resp) == "no"
    text_resp = httpx.Response(400, text="plain")
    assert "plain" in _response_detail(text_resp)
    other = httpx.Response(400, json={"error": "x"})
    assert "error" in _response_detail(other)


def test_create_adapter_selects_backend():
    from lightrag_wrapper.adapter import create_adapter

    assert create_adapter(make_config()).__class__.__name__ == "EmbeddedLightRAGAdapter"
    assert (
        create_adapter(make_config(lightrag_mode="server")).__class__.__name__
        == "ServerLightRAGAdapter"
    )
