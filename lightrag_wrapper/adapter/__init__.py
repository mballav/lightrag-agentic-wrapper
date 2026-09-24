"""Factory for LightRAG adapters selected by `LIGHTRAG_MODE`."""

from __future__ import annotations

from lightrag_wrapper.adapter.base import LightRAGAdapter
from lightrag_wrapper.adapter.embedded import EmbeddedLightRAGAdapter
from lightrag_wrapper.adapter.server import ServerLightRAGAdapter
from lightrag_wrapper.config import LightRAGConfig


def create_adapter(config: LightRAGConfig) -> LightRAGAdapter:
    """Return the embedded or server adapter based on `LIGHTRAG_MODE`."""
    if config.is_server_mode:
        return ServerLightRAGAdapter(config)
    return EmbeddedLightRAGAdapter(config)


__all__ = [
    "EmbeddedLightRAGAdapter",
    "LightRAGAdapter",
    "ServerLightRAGAdapter",
    "create_adapter",
]
