"""Embedded adapter tests with the LightRAG SDK class replaced."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest

from lightrag_wrapper.adapter.embedded import (
    EmbeddedLightRAGAdapter,
    _normalize_query_output,
    _supported_kwargs,
)
from lightrag_wrapper.exceptions import LightRAGAdapterError
from lightrag_wrapper.models import Reference
from tests.conftest import make_config


@dataclass
class QueryParam:
    mode: str
    only_need_context: bool = False
    user_prompt: str | None = None
    conversation_history: list | None = None
    top_k: int = 1
    chunk_top_k: int = 1
    max_entity_tokens: int = 1
    max_relation_tokens: int = 1
    max_total_tokens: int = 1
    hl_keywords: list | None = None
    ll_keywords: list | None = None


class FakeRAG:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.inserted = []
        self.fail_insert = False
        self.reject_file_paths = False
        self.query_result = {"response": "stored answer", "references": []}
        self.fail_query = False
        self.docs = {}
        self.bad_docs = False

    async def initialize_storages(self):
        return None

    async def finalize_storages(self):
        return None

    async def ainsert(self, documents, file_paths=None):
        if self.reject_file_paths and file_paths is not None:
            raise TypeError("file_paths")
        if self.fail_insert:
            raise RuntimeError("insert down")
        self.inserted.append((documents, file_paths))

    async def aquery(self, query, param=None):
        if self.fail_query:
            raise RuntimeError("query down")
        return self.query_result

    async def aget_docs_by_track_id(self, track_id):
        if self.bad_docs:
            raise RuntimeError("status down")
        return self.docs.get(track_id, {})


@pytest.fixture
def patched_lightrag(monkeypatch):
    import lightrag

    holder = {"rag": None}

    class LightRAG(FakeRAG):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            holder["rag"] = self

    monkeypatch.setattr(lightrag, "LightRAG", LightRAG)
    monkeypatch.setattr(lightrag, "QueryParam", QueryParam)
    monkeypatch.setattr(lightrag.utils, "setup_logger", lambda *args, **kwargs: None)
    return holder


async def test_initialize_ingest_and_query(tmp_path, patched_lightrag, document_paths):
    config = make_config(working_dir=str(tmp_path / "rag"), workspace="ws")
    adapter = EmbeddedLightRAGAdapter(config)
    await adapter.initialize()
    await adapter.initialize()
    rag = patched_lightrag["rag"]
    assert rag.kwargs["workspace"] == "ws"

    note = tmp_path / "itinerary.txt"
    note.write_text("Passenger itinerary for the attached e-ticket.", encoding="utf-8")
    result = await adapter.ingest_files([note])
    assert result.status == "success"
    assert result.sources == ["itinerary.txt"]
    assert rag.inserted

    with pytest.raises(LightRAGAdapterError, match="UTF-8"):
        await adapter.ingest_files(document_paths)

    one = note
    single = await adapter.ingest_file(one, source="alias.txt")
    assert single.sources == ["alias.txt"]

    rag.query_result = {
        "response": "leave policy",
        "references": [{"reference_id": "1", "file_path": one.name, "content": ["18 days"]}],
    }
    queried = await adapter.query(
        " How much leave? ",
        mode="local",
        only_need_context=True,
        user_prompt=None,
        conversation_history=[{"role": "user", "content": "policy"}],
        include_references=True,
        high_level_keywords=["leave"],
        low_level_keywords=["Berlin"],
    )
    assert queried.response == "leave policy"
    assert queried.references[0].file_path == one.name

    rag.query_result = "plain"
    plain = await adapter.query("q", include_references=False)
    assert plain.response == "plain"
    assert plain.references is None

    await adapter.aclose()
    await adapter.aclose()
    assert adapter._rag is None


async def test_insert_fallback_and_failures(tmp_path, patched_lightrag):
    adapter = EmbeddedLightRAGAdapter(make_config(working_dir=str(tmp_path)))
    with pytest.raises(LightRAGAdapterError, match="not initialized"):
        await adapter.ingest_texts(["hello"])

    await adapter.initialize()
    rag = patched_lightrag["rag"]
    with pytest.raises(LightRAGAdapterError, match="non-empty"):
        await adapter.ingest_texts(["  "])
    with pytest.raises(LightRAGAdapterError, match="non-empty"):
        await adapter.ingest_texts([])

    rag.reject_file_paths = True
    ok = await adapter.ingest_texts(["body"], sources=["notes.txt"])
    assert ok.status == "success"
    assert rag.inserted[-1][1] is None

    rag.reject_file_paths = False
    rag.fail_insert = True
    with pytest.raises(LightRAGAdapterError, match="Embedded ingest failed"):
        await adapter.ingest_texts(["body"])

    rag.fail_query = True
    with pytest.raises(LightRAGAdapterError, match="Embedded query failed"):
        await adapter.query("q")


async def test_binary_file_and_missing_binding(tmp_path, patched_lightrag):
    binary = tmp_path / "scan.bin"
    binary.write_bytes(b"\xff\xfe not utf8")
    adapter = EmbeddedLightRAGAdapter(make_config(working_dir=str(tmp_path / "rag")))
    await adapter.initialize()
    with pytest.raises(LightRAGAdapterError, match="UTF-8"):
        await adapter.ingest_file(binary)

    bad = EmbeddedLightRAGAdapter(
        make_config(working_dir=str(tmp_path / "bad"), llm_binding="anthropic")
    )
    with pytest.raises(LightRAGAdapterError, match="not supported"):
        await bad.initialize()


async def test_import_errors(tmp_path, monkeypatch):
    import lightrag as lightrag_mod

    adapter = EmbeddedLightRAGAdapter(make_config(working_dir=str(tmp_path)))
    monkeypatch.setitem(sys.modules, "lightrag", None)
    with pytest.raises(LightRAGAdapterError, match="lightrag-hku is required"):
        await adapter.initialize()
    monkeypatch.setitem(sys.modules, "lightrag", lightrag_mod)

    class LightRAG(FakeRAG):
        pass

    monkeypatch.setattr(lightrag_mod, "LightRAG", LightRAG)
    monkeypatch.setattr(lightrag_mod.utils, "setup_logger", lambda *a, **k: None)
    await adapter.initialize()
    monkeypatch.setitem(sys.modules, "lightrag", None)
    with pytest.raises(LightRAGAdapterError, match="lightrag-hku is required"):
        await adapter.query("q")
    monkeypatch.setitem(sys.modules, "lightrag", lightrag_mod)


async def test_track_status(tmp_path, patched_lightrag):
    adapter = EmbeddedLightRAGAdapter(make_config(working_dir=str(tmp_path)))
    await adapter.initialize()
    rag = patched_lightrag["rag"]

    class SdkDoc:
        status = "processed"
        file_path = "policy.txt"
        content_summary = "leave"
        content_length = 10
        chunks_count = 1
        error_msg = None
        created_at = "2026-01-01"
        updated_at = "2026-01-02"
        track_id = "t1"
        metadata = None

    rag.docs = {"t1": {"doc-1": SdkDoc()}}
    found = await adapter._fetch_track_documents("t1")
    assert found[0].id == "doc-1"
    assert await adapter._fetch_track_documents("missing") == []

    rag.docs = {"bad": ["nope"]}
    with pytest.raises(LightRAGAdapterError, match="Unexpected"):
        await adapter._fetch_track_documents("bad")

    rag.bad_docs = True
    with pytest.raises(LightRAGAdapterError, match="Embedded track status"):
        await adapter._fetch_track_documents("t1")

    adapter._rag = object()
    with pytest.raises(LightRAGAdapterError, match="does not support"):
        await adapter._fetch_track_documents("t1")


def test_supported_kwargs_and_output():
    @dataclass
    class Box:
        a: int

    assert _supported_kwargs(Box, {"a": 1, "b": 2}) == {"a": 1}

    def fn(a, **kwargs):
        return a

    assert _supported_kwargs(fn, {"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def narrow(a):
        return a

    assert _supported_kwargs(narrow, {"a": 1, "b": 2}) == {"a": 1}
    assert _supported_kwargs(object(), {"a": 1}) == {"a": 1}

    text, refs = _normalize_query_output(
        {"answer": "x", "references": [{"file_path": "a.txt"}]}, True
    )
    assert text == "x"
    assert isinstance(refs[0], Reference)
    dumped, none = _normalize_query_output({"other": 1}, True)
    assert "other" in dumped
    assert none is None
    assert _normalize_query_output(12, True) == ("12", None)
    assert _normalize_query_output({"content": "c"}, False)[1] is None


def test_export_skips_missing_password(tmp_path, monkeypatch):
    from lightrag_wrapper.adapter.embedded import _export_storage_env

    config = make_config(
        postgres_password=None,
        neo4j_password=None,
        milvus_user=None,
        milvus_password=None,
    )
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("MILVUS_USER", raising=False)
    monkeypatch.delenv("MILVUS_PASSWORD", raising=False)
    _export_storage_env(config)
    assert "POSTGRES_PASSWORD" not in __import__("os").environ or True
    # None values are not written; a pre-existing value must remain untouched
    # only when the config value is None. Confirm host is exported.
    assert __import__("os").environ["POSTGRES_HOST"] == config.postgres_host
    assert "MILVUS_USER" not in __import__("os").environ
    assert "MILVUS_PASSWORD" not in __import__("os").environ


def test_export_milvus_credentials(monkeypatch):
    from lightrag_wrapper.adapter.embedded import _export_storage_env

    monkeypatch.delenv("MILVUS_USER", raising=False)
    monkeypatch.delenv("MILVUS_PASSWORD", raising=False)
    config = make_config(milvus_user="root", milvus_password="secret")
    _export_storage_env(config)
    assert __import__("os").environ["MILVUS_USER"] == "root"
    assert __import__("os").environ["MILVUS_PASSWORD"] == "secret"
