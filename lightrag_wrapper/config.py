"""LightRAG configuration loaded from environment variables / `.env` files."""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Literal

import aiofiles
import aiofiles.ospath
from dotenv import dotenv_values
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from lightrag_wrapper.models import QueryMode

LightRAGMode = Literal["embedded", "server"]
AppEnv = Literal["dev", "prod"]
SUPPORTED_EMBEDDED_BINDINGS = frozenset({"openai", "azure_openai", "ollama"})
SUPPORTED_APP_ENVS = frozenset({"dev", "prod"})


def resolve_app_env(app_env: str | None = None) -> AppEnv:
    value = (app_env or os.getenv("APP_ENV") or "dev").strip().lower()
    if value not in SUPPORTED_APP_ENVS:
        return "dev"
    return value  # type: ignore[return-value]


async def resolve_env_files(
    app_env: str | None = None,
    *,
    stack_file: str | Path | None = None,
    wrapper_file: str | Path | None = None,
    lightrag_file: str | Path | None = None,
) -> tuple[str, ...]:
    """Locate env files for the active APP_ENV.

    Order is client credentials (Postgres, Neo4j, Milvus), then LightRAG, then
    wrapper. Later files override earlier ones. Missing files are skipped so
    process env / Compose injection still works. MinIO is not loaded.
    """
    env_name = resolve_app_env(app_env)
    stack = Path(
        stack_file
        or os.getenv("STACK_ENV_FILE")
        or Path("env") / "stack" / env_name
    )
    lightrag = Path(
        lightrag_file
        or os.getenv("LIGHTRAG_ENV_FILE")
        or Path("env") / "lightrag-config" / f".env.{env_name}"
    )
    wrapper = Path(
        wrapper_file
        or os.getenv("WRAPPER_ENV_FILE")
        or Path("env") / "wrapper" / f".env.{env_name}"
    )
    files: list[str] = []
    for path in (*_stack_paths(stack), lightrag, wrapper):
        if await aiofiles.ospath.isfile(path):
            files.append(str(path))
    return tuple(files)


# Clients talk to Postgres, Neo4j, and Milvus. MinIO is only the Milvus object store.
_CLIENT_STACK_FILES = ("postgres.env", "neo4j.env", "milvus.env")


def _stack_paths(stack: Path) -> tuple[Path, ...]:
    """Credential files the LightRAG client needs. ``minio.env`` is excluded."""
    if stack.is_dir():
        return tuple(stack / name for name in _CLIENT_STACK_FILES)
    return (stack,)


async def _load_dotenv_file(path: str | Path) -> dict[str, str]:
    async with aiofiles.open(path, encoding="utf-8") as handle:
        text = await handle.read()
    parsed = dotenv_values(stream=StringIO(text))
    return {
        key: value
        for key, value in parsed.items()
        if key and value is not None
    }


async def _apply_dotenv_files(files: tuple[str, ...]) -> None:
    values: dict[str, str] = {}
    for file in files:
        if not await aiofiles.ospath.isfile(file):
            continue
        values.update(await _load_dotenv_file(file))
    for key, value in values.items():
        os.environ.setdefault(key, value)


class LightRAGConfig(BaseSettings):
    """Wrapper + LightRAG settings from env vars and split `.env` files.

    `env/stack/{dev|prod}/*.env` holds credentials shared with the data services.
    `env/lightrag-config/.env.{dev|prod}` holds LLM, embedding, storage
    endpoints, and server process settings. `env/wrapper/.env.{dev|prod}` holds
    wrapper process settings. `APP_ENV` selects the set. Later files win.
    """

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_env: AppEnv = "dev"

    # --- Adapter routing ---
    lightrag_mode: LightRAGMode = Field(
        default="embedded",
        description="Which backend to use: embedded SDK or HTTP server.",
    )
    lightrag_server_url: str = Field(
        default="http://127.0.0.1:9621",
        description="Base URL of a standalone LightRAG server (server mode).",
    )
    lightrag_api_key: str | None = Field(
        default=None,
        description="Sent as X-API-Key when calling the LightRAG server.",
    )
    lightrag_server_timeout: float = Field(
        default=120.0,
        ge=1,
        description="HTTP timeout in seconds for LightRAG server calls.",
    )
    lightrag_ingest_wait: bool = Field(
        default=True,
        description="Wait for server-side indexing to finish after ingest.",
    )
    lightrag_ingest_timeout: float = Field(
        default=600.0,
        ge=1,
        description="How long to wait for server ingest to reach a terminal state.",
    )
    lightrag_query_mode: QueryMode = Field(
        default=QueryMode.MIX,
        description="Default query mode when the caller does not specify one.",
    )

    # --- Wrapper HTTP app ---
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    wrapper_upload_dir: str = Field(
        default="./data/wrapper/uploads",
        description="Where the wrapper API stages multipart uploads before ingest.",
    )
    save_dag_visual: bool = Field(
        default=False,
        description="Write a PNG of each query DAG when SAVE_DAG_VISUAL is true.",
    )
    dag_visual_dir: str = Field(
        default="./data/query_dags",
        description="Directory for saved query DAG images.",
    )

    # --- LightRAG working directories / isolation ---
    working_dir: str = "./data/embedded/rag_storage"
    workspace: str | None = None
    input_dir: str = Field(
        default="./data/embedded/inputs",
        description="LightRAG INPUT_DIR (document input), not wrapper upload staging.",
    )

    # --- LLM (LightRAG-native names) ---
    llm_binding: str = "openai"
    llm_binding_host: str | None = "https://api.openai.com/v1"
    llm_binding_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout: int = Field(default=240, ge=0)
    max_async_llm: int = Field(default=4, ge=1)
    azure_openai_api_version: str | None = None

    # --- LiteLLM service (retries / reliability) ---
    # LiteLLM exposes num_retries on completion/responses; other APIs are uneven.
    # LLMService applies a uniform tenacity policy (see llm/service.py).
    llm_num_retries: int = Field(
        default=3,
        ge=0,
        description="Attempts for each LiteLLM call (including the first).",
    )
    llm_retry_strategy: Literal[
        "exponential_backoff_retry", "constant_retry"
    ] = Field(
        default="exponential_backoff_retry",
        description="Backoff between retries for LLMService calls.",
    )
    llm_retry_min_wait: float = Field(
        default=1.0,
        ge=0,
        description="Minimum wait seconds for exponential backoff.",
    )
    llm_retry_max_wait: float = Field(
        default=10.0,
        ge=0,
        description="Maximum wait seconds for exponential backoff.",
    )
    llm_drop_params: bool = Field(
        default=True,
        description="Drop unsupported OpenAI params instead of raising.",
    )

    # --- Embedding (LightRAG-native names) ---
    embedding_binding: str = "openai"
    embedding_binding_host: str | None = "https://api.openai.com/v1"
    embedding_binding_api_key: str | None = None
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = Field(default=3072, ge=1)
    embedding_token_limit: int = Field(default=8192, ge=1)
    embedding_batch_num: int = Field(default=32, ge=1)
    embedding_func_max_async: int = Field(default=8, ge=1)

    # --- Storage backends ---
    lightrag_kv_storage: str = "PGKVStorage"
    lightrag_vector_storage: str = "MilvusVectorDBStorage"
    lightrag_graph_storage: str = "Neo4JStorage"
    lightrag_doc_status_storage: str = "PGDocStatusStorage"

    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_user: str = "rag"
    postgres_password: str | None = "rag"
    postgres_database: str = "rag"
    postgres_max_connections: int = Field(default=25, ge=1)

    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str | None = "neo4j@lightrag"
    neo4j_database: str = "neo4j"

    milvus_uri: str = "http://localhost:19530"
    milvus_db_name: str = "lightrag"
    milvus_device: str = "cpu"
    milvus_user: str | None = Field(
        default=None,
        description="Milvus root user. LightRAG reads this as MILVUS_USER.",
    )
    milvus_password: str | None = Field(
        default=None,
        description="Milvus root password. LightRAG reads this as MILVUS_PASSWORD.",
    )

    # --- Indexing ---
    max_parallel_insert: int = Field(default=3, ge=1)
    chunk_token_size: int = Field(default=1200, ge=1)
    chunk_overlap_token_size: int = Field(default=100, ge=0)
    summary_language: str = "English"
    enable_llm_cache: bool = True

    # --- Query defaults (also read by LightRAG QueryParam from env) ---
    top_k: int = Field(default=40, ge=1)
    chunk_top_k: int = Field(default=20, ge=1)
    max_entity_tokens: int = Field(default=6000, ge=1)
    max_relation_tokens: int = Field(default=8000, ge=1)
    max_total_tokens: int = Field(default=30000, ge=1)

    @field_validator("app_env", "lightrag_mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("llm_binding", "embedding_binding", mode="before")
    @classmethod
    def _normalize_binding(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"openai_compatible", "openai-compatible"}:
                return "openai"
            return normalized
        return value

    def require_embedded_bindings(self) -> None:
        """Embedded mode only wires openai, azure_openai, and ollama."""
        for kind, binding in (
            ("LLM_BINDING", self.llm_binding),
            ("EMBEDDING_BINDING", self.embedding_binding),
        ):
            if binding not in SUPPORTED_EMBEDDED_BINDINGS:
                raise ValueError(
                    f"{kind}={binding!r} is not supported in embedded mode. "
                    f"Use one of {sorted(SUPPORTED_EMBEDDED_BINDINGS)}, "
                    "or set LIGHTRAG_MODE=server and let LightRAG Server handle the provider."
                )

    @field_validator("lightrag_server_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def is_server_mode(self) -> bool:
        return self.lightrag_mode == "server"

    @property
    def is_embedded_mode(self) -> bool:
        return self.lightrag_mode == "embedded"

    @property
    def resolved_embedding_api_key(self) -> str | None:
        return self.embedding_binding_api_key or self.llm_binding_api_key

    @classmethod
    async def from_env(
        cls,
        env_file: str | Path | tuple[str | Path, ...] | None = None,
        *,
        app_env: str | None = None,
    ) -> LightRAGConfig:
        """Build config from process env and the stack, LightRAG, and wrapper files."""
        if env_file is None:
            files = await resolve_env_files(app_env)
        elif isinstance(env_file, (str, Path)):
            files = (str(env_file),)
        else:
            files = tuple(str(path) for path in env_file)
        await _apply_dotenv_files(files)
        return cls()


_config: LightRAGConfig | None = None


async def get_config() -> LightRAGConfig:
    global _config
    if _config is None:
        _config = await LightRAGConfig.from_env()
    return _config
