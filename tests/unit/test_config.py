from pathlib import Path

import pytest

from lightrag_wrapper.config import (
    LightRAGConfig,
    _apply_dotenv_files,
    get_config,
    resolve_app_env,
    resolve_env_files,
)
from tests.conftest import make_config


def test_resolve_app_env(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    assert resolve_app_env(None) == "dev"
    assert resolve_app_env(" PROD ") == "prod"
    assert resolve_app_env("staging") == "dev"
    monkeypatch.setenv("APP_ENV", "prod")
    assert resolve_app_env(None) == "prod"


async def test_resolve_env_files_skips_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("STACK_ENV_FILE", raising=False)
    monkeypatch.delenv("WRAPPER_ENV_FILE", raising=False)
    monkeypatch.delenv("LIGHTRAG_ENV_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert await resolve_env_files("dev") == ()

    stack = tmp_path / "env" / "stack" / "dev"
    wrapper = tmp_path / "env" / "wrapper"
    lightrag = tmp_path / "env" / "lightrag-config"
    stack.mkdir(parents=True)
    wrapper.mkdir(parents=True)
    lightrag.mkdir(parents=True)
    (stack / "postgres.env").write_text("POSTGRES_USER=lightrag\n", encoding="utf-8")
    (stack / "minio.env").write_text("MINIO_ROOT_USER=minioadmin\n", encoding="utf-8")
    (lightrag / ".env.dev").write_text("LLM_MODEL=from-file\n", encoding="utf-8")
    (wrapper / ".env.dev").write_text("LOG_LEVEL=DEBUG\n", encoding="utf-8")
    files = await resolve_env_files("dev")
    assert [Path(item).name for item in files] == [
        "postgres.env",
        ".env.dev",
        ".env.dev",
    ]
    assert "minio.env" not in {Path(item).name for item in files}
    assert "stack" in files[0]
    assert "lightrag-config" in files[1]
    assert "wrapper" in files[2]


async def test_resolve_env_files_explicit_override(tmp_path):
    rag = tmp_path / "rag.env"
    wrap = tmp_path / "wrap.env"
    rag.write_text("LLM_MODEL=a\n", encoding="utf-8")
    files = await resolve_env_files(
        "prod",
        stack_file=tmp_path / "absent.env",
        wrapper_file=wrap,
        lightrag_file=rag,
    )
    assert files == (str(rag),)


async def test_apply_dotenv_does_not_override_process_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_MODEL=from-file\nEMPTY=\n# comment\nBARE\nKEEP_ME=file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KEEP_ME", "process")
    await _apply_dotenv_files((str(env_file), str(tmp_path / "missing.env")))
    assert __import__("os").environ["LLM_MODEL"] == "from-file"
    assert __import__("os").environ["KEEP_ME"] == "process"
    monkeypatch.delenv("LLM_MODEL", raising=False)


def test_validators_and_properties():
    config = make_config(
        lightrag_mode=" SERVER ",
        llm_binding="OpenAI_Compatible",
        embedding_binding="azure_openai",
        lightrag_server_url="http://rag.example/",
        embedding_binding_api_key=None,
        llm_binding_api_key="llm-key",
        workspace="team",
    )
    assert config.lightrag_mode == "server"
    assert config.llm_binding == "openai"
    assert config.embedding_binding == "azure_openai"
    assert config.lightrag_server_url == "http://rag.example"
    assert config.is_server_mode
    assert not config.is_embedded_mode
    assert config.resolved_embedding_api_key == "llm-key"
    assert config.workspace == "team"

    with pytest.raises(Exception):
        LightRAGConfig.model_validate(
            {"llm_binding": 3, "app_env": 1, "lightrag_mode": "embedded"}
        )


def test_save_dag_visual_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("SAVE_DAG_VISUAL", "true")
    monkeypatch.setenv("DAG_VISUAL_DIR", "/app/data/query_dags")
    config = LightRAGConfig(app_env="dev", lightrag_mode="embedded")
    assert config.save_dag_visual is True
    assert config.dag_visual_dir == "/app/data/query_dags"

    monkeypatch.setenv("SAVE_DAG_VISUAL", "false")
    assert LightRAGConfig(app_env="dev", lightrag_mode="embedded").save_dag_visual is False


def test_require_embedded_bindings():
    make_config().require_embedded_bindings()
    with pytest.raises(ValueError, match="LLM_BINDING"):
        make_config(llm_binding="anthropic").require_embedded_bindings()
    with pytest.raises(ValueError, match="EMBEDDING_BINDING"):
        make_config(embedding_binding="bedrock").require_embedded_bindings()


async def test_from_env_and_get_config(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LOG_LEVEL=WARNING\nAPP_PORT=8123\n", encoding="utf-8")
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("APP_PORT", raising=False)
    loaded = await LightRAGConfig.from_env(env_file)
    assert loaded.log_level == "WARNING"
    assert loaded.app_port == 8123

    other = tmp_path / "other.env"
    other.write_text("SUMMARY_LANGUAGE=German\n", encoding="utf-8")
    monkeypatch.delenv("SUMMARY_LANGUAGE", raising=False)
    pair = await LightRAGConfig.from_env((env_file, other))
    assert pair.summary_language == "German"

    async def _empty_files(app_env=None):
        return ()

    monkeypatch.setattr(
        "lightrag_wrapper.config.resolve_env_files", _empty_files
    )
    first = await get_config()
    second = await get_config()
    assert first is second
