from __future__ import annotations
from datetime import datetime, timezone
from unittest.mock import patch
import pytest
from graph_bot.datatypes import (
    ReasoningEdge,
    ReasoningNode,
    ReasoningTree,
    SeedData,
)
from graph_bot.settings import settings


@pytest.fixture
def mock_settings(tmp_path):
    metagraph_path = tmp_path / "metagraph.json"
    pricing_path = tmp_path / "pricing.yaml"
    pricing_path.write_text(
        """
pricing_version: "v0"
models:
  llama3-8b-instruct:
    input_usd_per_1k: 0.0
    output_usd_per_1k: 0.0
  mock:
    input_usd_per_1k: 0.0
    output_usd_per_1k: 0.0
""".lstrip(),
        encoding="utf-8",
    )
    with (
        patch.object(settings, "metagraph_path", metagraph_path),
        patch.object(settings, "pricing_path", pricing_path),
    ):
        metagraph_path.parent.mkdir(parents=True, exist_ok=True)
        yield settings


@pytest.fixture
def sample_seed_data() -> list[SeedData]:
    return [SeedData(id="seed1", content="2 4 6 -> 12")]


@pytest.fixture
def sample_tree() -> ReasoningTree:
    root = ReasoningNode(node_id="root", text="root", type="thought")
    child = ReasoningNode(node_id="child", text="child", type="answer")
    edge = ReasoningEdge(src="root", dst="child", relation="leads_to")
    return ReasoningTree(
        tree_id="tree1", root_id="root", nodes=[root, child], edges=[edge]
    )


@pytest.fixture
def mock_datetime():
    fixed_now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    with patch("graph_bot.adapters.graphrag.datetime") as mock_g:
        mock_g.now.return_value = fixed_now
        mock_g.fromisoformat.side_effect = datetime.fromisoformat
        mock_g.timezone = timezone
        yield fixed_now
