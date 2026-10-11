"""Run the wrapper API: `python -m lightrag_wrapper`."""

import asyncio

import uvicorn

from lightrag_wrapper.config import get_config
from lightrag_wrapper.log import configure_logging, uvicorn_log_config


def main() -> None:
    config = asyncio.run(get_config())
    configure_logging(config.log_level)
    uvicorn.run(
        "lightrag_wrapper.app:app",
        host=config.app_host,
        port=config.app_port,
        reload=False,
        log_level=config.log_level.lower(),
        log_config=uvicorn_log_config(config.log_level),
    )


if __name__ == "__main__":
    main()
