from __future__ import annotations
import pytest
import yaml
from graph_bot.utils.pricing import load_pricing_table, calculate_cost


def test_load_pricing_table(tmp_path):
    pricing_file = tmp_path / "pricing.yaml"
    data = {
        "models": {
            "gpt-4o-mini": {"input_usd_per_1k": 0.00015, "output_usd_per_1k": 0.0006}
        }
    }
    with pricing_file.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)

    loaded = load_pricing_table(pricing_file)
    assert loaded == data


def test_load_pricing_table_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pricing_table(tmp_path / "non_existent.yaml")


def test_calculate_cost():
    pricing_table = {
        "models": {"gpt-4": {"input_usd_per_1k": 0.03, "output_usd_per_1k": 0.06}}
    }

    cost = calculate_cost(pricing_table, "gpt-4", 1000, 2000)
    assert pytest.approx(cost) == 0.15


def test_calculate_cost_unknown_model():
    pricing_table = {"models": {}}
    assert calculate_cost(pricing_table, "unknown", 1000, 1000) == 0.0


def test_calculate_cost_missing_pricing():
    pricing_table = {"models": {"gpt-4": {}}}
    assert calculate_cost(pricing_table, "gpt-4", 1000, 1000) == 0.0
