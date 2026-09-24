"""Run the wrapper API: `python -m lightrag_wrapper`."""

import asyncio

import uvicorn

from lightrag_wrapper.config import get_config


def main() -> None:
    config = asyncio.run(get_config())
    uvicorn.run(
        "lightrag_wrapper.app:app",
        host=config.app_host,
        port=config.app_port,
        reload=False,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
