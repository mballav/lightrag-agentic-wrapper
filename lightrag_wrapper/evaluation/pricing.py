"""Configurable model prices, in currency units per one million tokens."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PRICES_PATH = Path(__file__).with_name("openai_prices.json")


def load_prices(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Read ``{"model": {"input": number, "output": number}}``.

    ``None`` loads the bundled OpenAI price file. A path that does not exist
    means no prices.
    """
    file = DEFAULT_PRICES_PATH if path is None else Path(path)
    if not file.is_file():
        return {}
    payload = json.loads(file.read_text(encoding="utf-8"))
    prices: dict[str, dict[str, float]] = {}
    for model, value in payload.items():
        if not isinstance(value, dict):
            continue
        if "input" not in value or "output" not in value:
            continue
        prices[str(model)] = {"input": float(value["input"]), "output": float(value["output"])}
    return prices
