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
    with patch.object(
        settings, "metagraph_path", tmp_path / "metagraph.json"
    ), patch.object(settings, "pricing_path", tmp_path / "pricing.yaml"):
        (tmp_path / "metagraph.json").parent.mkdir(parents=True, exist_ok=True)
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
