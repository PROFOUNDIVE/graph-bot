from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict


def load_pricing_table(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pricing table not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_cost(
    pricing_table: Dict[str, Any],
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    models = pricing_table.get("models", {})
    if model_name not in models:
        return 0.0

    config = models[model_name]
    input_usd_per_1k = config.get("input_usd_per_1k", 0.0)
    output_usd_per_1k = config.get("output_usd_per_1k", 0.0)

    cost = (prompt_tokens / 1000.0) * input_usd_per_1k + (
        completion_tokens / 1000.0
    ) * output_usd_per_1k
    return cost
