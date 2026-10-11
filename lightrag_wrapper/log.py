"""Structured logging for the wrapper and the libraries it hosts."""

from __future__ import annotations

import logging
import logging.config
import sys
from typing import Any

import structlog


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to ``name``."""
    return structlog.get_logger(name)


def configure_logging(level: str = "INFO") -> None:
    """Install structlog and route stdlib logs through the same renderer."""
    resolved = _level_name(level)
    structlog.configure(
        processors=[
            *_shared_processors(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    logging.config.dictConfig(uvicorn_log_config(resolved))
    logging.captureWarnings(True)


def uvicorn_log_config(level: str = "INFO") -> dict[str, Any]:
    """Uvicorn ``log_config`` that renders access and error logs with structlog."""
    resolved = _level_name(level)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": _structured_formatter,
            },
        },
        "handlers": {
            "structured": {
                "class": "logging.StreamHandler",
                "formatter": "structured",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["structured"],
                "level": resolved,
                "propagate": False,
            },
            "uvicorn.error": {"level": resolved},
            "uvicorn.access": {
                "handlers": ["structured"],
                "level": resolved,
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["structured"],
            "level": resolved,
        },
    }


def _structured_formatter() -> structlog.stdlib.ProcessorFormatter:
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_shared_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _renderer(),
        ],
    )


def _shared_processors() -> list[structlog.types.Processor]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def _renderer() -> structlog.types.Processor:
    if sys.stderr.isatty():
        return structlog.dev.ConsoleRenderer()
    return structlog.processors.JSONRenderer()


def _level_name(level: str) -> str:
    name = str(level or "INFO").upper()
    value = getattr(logging, name, None)
    if isinstance(value, int):
        return name
    return "INFO"
