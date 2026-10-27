"""Structured logging for the wrapper and the libraries it hosts."""

from __future__ import annotations

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any

import structlog

_LOG_FILE = Path("/app/data/wrapper/logs/wrapper.log")
_LOG_MAX_BYTES = 10 * 1024 * 1024
# The active file plus these rolled files is five files on disk.
_LOG_BACKUP_COUNT = 4


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
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _close_configured_handlers()
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
            "file": {
                "()": _file_formatter,
            },
        },
        "handlers": {
            "structured": {
                "class": "logging.StreamHandler",
                "formatter": "structured",
                "stream": "ext://sys.stderr",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "file",
                "filename": str(_LOG_FILE),
                "maxBytes": _LOG_MAX_BYTES,
                "backupCount": _LOG_BACKUP_COUNT,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["structured", "file"],
                "level": resolved,
                "propagate": False,
            },
            "uvicorn.error": {"level": resolved},
            "uvicorn.access": {
                "handlers": ["structured", "file"],
                "level": resolved,
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["structured", "file"],
            "level": resolved,
        },
    }


def _close_configured_handlers() -> None:
    """Close handlers installed by an earlier ``configure_logging`` call."""
    seen: set[int] = set()
    for name in ("", "uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            handler.close()
        logger.handlers.clear()


def _structured_formatter() -> structlog.stdlib.ProcessorFormatter:
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_shared_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _renderer(),
        ],
    )


def _file_formatter() -> structlog.stdlib.ProcessorFormatter:
    """JSON for the log file, including when stderr is a color terminal."""
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_shared_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
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
