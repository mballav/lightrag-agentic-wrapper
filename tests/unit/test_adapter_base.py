import pytest

from lightrag_wrapper.adapter.base import LightRAGAdapter
from lightrag_wrapper.exceptions import LightRAGAdapterError
from lightrag_wrapper.models import DocumentIngestStatus, IngestResult, QueryMode
from tests.conftest import make_config


class StubAdapter(LightRAGAdapter):
    def __init__(self, config, *, fetch=None, boom=None):
        super().__init__(config)
        self.fetch = fetch or {}
        self.boom = boom or {}
        self.initialized = False
        self.closed = False

    async def initialize(self):
        self.initialized = True

    async def aclose(self):
        self.closed = True

    async def ingest_texts(self, texts, *, sources=None, wait=None):
        return IngestResult(
            status="success",
            message="indexed",
            sources=list(sources or []),
            details={"n": len(texts), "wait": wait},
        )

    async def ingest_file(self, path, *, source=None, wait=None):
        return await self.ingest_files([path], sources=[source] if source else None, wait=wait)

    async def ingest_files(self, paths, *, sources=None, wait=None):
        files, names = await self._normalize_file_batch(paths, sources)
        return IngestResult(
            status="success",
            message="files",
            sources=names or [path.name for path in files],
        )

    async def query(self, query, **kwargs):
        raise NotImplementedError

    async def _fetch_track_documents(self, track_id):
        if track_id in self.boom:
            raise self.boom[track_id]
        return self.fetch.get(track_id, [])

    async def _fetch_processed_document_page(self, *, page, page_size):
        return [], 0, {}, False


async def test_context_manager_and_ingest_text(tmp_path):
    config = make_config()
    async with StubAdapter(config) as adapter:
        assert adapter.initialized
        result = await adapter.ingest_text("hello", source="notes.txt", wait=False)
        assert result.sources == ["notes.txt"]
        bare = await adapter.ingest_text("hello")
        assert bare.sources == []
    assert adapter.closed


async def test_normalize_file_batch(tmp_path):
    adapter = StubAdapter(make_config())
    good = tmp_path / "a.txt"
    good.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        await adapter._normalize_file_batch([], None)
    with pytest.raises(LightRAGAdapterError, match="File not found"):
        await adapter._normalize_file_batch([tmp_path / "missing.txt"], None)
    with pytest.raises(ValueError, match="sources length"):
        await adapter._normalize_file_batch([good], ["a", "b"])
    files, sources = await adapter._normalize_file_batch([good], None)
    assert sources is None
    assert files == [good]


def test_resolve_mode():
    adapter = StubAdapter(make_config(lightrag_query_mode=QueryMode.HYBRID))
    assert adapter.resolve_mode(None) is QueryMode.HYBRID
    assert adapter.resolve_mode("local") is QueryMode.LOCAL
    assert adapter.resolve_mode(QueryMode.NAIVE) is QueryMode.NAIVE


async def test_ingest_status_isolates_failures():
    adapter = StubAdapter(
        make_config(),
        fetch={
            "ok": [DocumentIngestStatus(id="1", status="processed")],
        },
        boom={
            "adapter": LightRAGAdapterError("down", status_code=500),
            "unexpected": RuntimeError("nope"),
        },
    )
    batch = await adapter.ingest_status(["ok", "missing", "adapter", "unexpected", "ok"])
    by_id = {item.track_id: item for item in batch.tracks}
    assert by_id["ok"].status == "success"
    assert by_id["missing"].status == "not_found"
    assert by_id["adapter"].status == "error"
    assert by_id["unexpected"].status == "error"
    assert "Unexpected" in by_id["unexpected"].message
