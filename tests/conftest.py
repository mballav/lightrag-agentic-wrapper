"""Shared fixtures. Tests never call a live LLM or storage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag_wrapper.config import LightRAGConfig

FIXTURES = Path(__file__).parent / "fixtures" / "documents"


@pytest.fixture(autouse=True)
def _reset_singletons():
    import lightrag_wrapper.config as config_mod
    import lightrag_wrapper.llm.service as llm_mod

    config_mod._config = None
    llm_mod._llm_service = None
    yield
    config_mod._config = None
    llm_mod._llm_service = None


@pytest.fixture
def document_paths() -> list[Path]:
    paths = sorted(FIXTURES.glob("*.pdf"))
    assert len(paths) >= 4, "e-ticket PDF fixtures are missing"
    return paths


def make_config(**overrides) -> LightRAGConfig:
    values = {
        "app_env": "dev",
        "lightrag_mode": "embedded",
        "lightrag_server_url": "http://127.0.0.1:9621",
        "lightrag_api_key": None,
        "lightrag_server_timeout": 30.0,
        "lightrag_ingest_wait": True,
        "lightrag_ingest_timeout": 5.0,
        "llm_binding": "openai",
        "embedding_binding": "openai",
        "llm_binding_api_key": "test-key",
        "embedding_binding_api_key": None,
        "llm_binding_host": "https://llm.example/v1",
        "embedding_binding_host": "https://embed.example/v1",
        "llm_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-large",
        "llm_num_retries": 1,
        "llm_retry_min_wait": 0.0,
        "llm_retry_max_wait": 0.0,
        "llm_retry_strategy": "constant_retry",
        "wrapper_upload_dir": "./data/wrapper/uploads",
        "save_dag_visual": False,
        "dag_visual_dir": "./data/query_dags",
        "working_dir": "./data/embedded/rag_storage",
        "postgres_password": "rag",
        "neo4j_password": "neo4j",
    }
    values.update(overrides)
    return LightRAGConfig(**values)
