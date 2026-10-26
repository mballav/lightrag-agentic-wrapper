"""Query path through the HTTP API with the real rewriter and mode router."""

import json

from fastapi.testclient import TestClient

from lightrag_wrapper.models import IngestResult, QueryResult
from tests.conftest import make_config


class ScriptedLLM:
    last_decomposer_prompt = None

    def __init__(self, config):
        self.config = config

    async def complete_prompt(self, prompt, **kwargs):
        system = kwargs.get("system_prompt") or ""
        if "retrieval-mode router" in system:
            return '{"mode": "hybrid", "reason": "entities relate"}'
        if "Query Decomposer" in system:
            ScriptedLLM.last_decomposer_prompt = prompt
            return (
                '[{"Id": 1, "nodeType": "retrieve",'
                ' "subquery": "Retrieve the Berlin annual leave accrual",'
                ' "depends_on": [], "isResultNode": true}]'
            )
        if "Evidence Evaluator" in system:
            return (
                '{"decision": "SUFFICIENT", "missing_information": [],'
                ' "next_action": {"type": "SUFFICIENT", "purpose": null,'
                ' "query": null, "mode": null}}'
            )
        if "user-facing answer" in system:
            return "Employees accrue 18 days of annual leave."
        return (
            '{"rewritten_query": "Annual leave accrual for the Berlin office",'
            ' "high_level_keywords": ["employee leave policy"],'
            ' "low_level_keywords": ["Berlin", "18 days"],'
            ' "is_context_dependent": true, "ambiguities": [], "confidence": 0.9}'
        )


class QueryAdapter:
    def __init__(self, config):
        self.config = config
        self.last = None

    async def initialize(self):
        return None

    async def aclose(self):
        return None

    async def ingest_texts(self, texts, *, sources=None, wait=None):
        return IngestResult(
            status="success",
            message="ok",
            track_id="q-track",
            track_ids=["q-track"],
            sources=list(sources or ["notes.txt"]),
        )

    async def list_document_catalog(self):
        return {
            "catalog_complete": True,
            "processed_document_count": 1,
            "listed_document_count": 1,
            "listing_policy": "All processed documents are listed.",
            "status_counts": {"processed": 1},
            "documents": [
                {
                    "id": "doc-leave",
                    "file_path": "leave-policy.pdf",
                    "preview": "Full-time employees accrue 18 days",
                }
            ],
        }

    async def query(self, query, **kwargs):
        self.last = (query, kwargs)
        return QueryResult(
            response="Employees accrue 18 days.",
            mode=kwargs["mode"],
            references=[],
        )


def test_query_uses_rewriter_and_router(monkeypatch, tmp_path):
    import lightrag_wrapper.app as app_module

    holder = {}
    config = make_config(wrapper_upload_dir=str(tmp_path))

    async def fake_get_config():
        return config

    def fake_create(cfg):
        holder["adapter"] = QueryAdapter(cfg)
        return holder["adapter"]

    monkeypatch.setattr(app_module, "get_config", fake_get_config)
    monkeypatch.setattr(app_module, "create_adapter", fake_create)
    monkeypatch.setattr(app_module, "LLMService", ScriptedLLM)

    with TestClient(app_module.app) as client:
        ingested = client.post(
            "/ingest",
            json={
                "text": "Full-time employees accrue 18 days of annual leave.",
                "source": "ETicket.pdf",
            },
        )
        assert ingested.status_code == 200
        response = client.post(
            "/query",
            json={
                "query": "How many days do they get?",
                "conversation_history": [
                    {"role": "user", "content": "Tell me about leave in Berlin."}
                ],
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        line = json.loads(response.text)
        assert line["id"] == 1
        assert "18 days" in line["response"]
        rewritten, kwargs = holder["adapter"].last
        assert "Berlin" in rewritten
        assert kwargs["high_level_keywords"] == ["employee leave policy"]
        assert kwargs["only_need_context"] is True
        assert kwargs["include_references"] is True
        assert "leave-policy.pdf" in ScriptedLLM.last_decomposer_prompt
