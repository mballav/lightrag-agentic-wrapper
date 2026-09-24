"""Adapter layer around LightRAG (embedded SDK or standalone server)."""

from lightrag_wrapper.adapter import LightRAGAdapter, create_adapter
from lightrag_wrapper.config import LightRAGConfig, resolve_env_files
from lightrag_wrapper.exceptions import LightRAGAdapterError, LLMServiceError
from lightrag_wrapper.llm import LLMService, get_llm_service
from lightrag_wrapper.models import (
    DocumentIngestStatus,
    IngestResult,
    QueryMode,
    QueryResult,
    Reference,
    TrackIngestStatus,
    TrackStatusBatchResult,
)
from lightrag_wrapper.router import QueryModeRouter

__all__ = [
    "DocumentIngestStatus",
    "IngestResult",
    "LLMService",
    "LLMServiceError",
    "LightRAGAdapter",
    "LightRAGAdapterError",
    "LightRAGConfig",
    "QueryMode",
    "QueryModeRouter",
    "QueryResult",
    "Reference",
    "TrackIngestStatus",
    "TrackStatusBatchResult",
    "create_adapter",
    "get_llm_service",
    "resolve_env_files",
]
