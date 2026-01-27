from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graph_bot.datatypes import RetrievalPath, RetrievalResult, SeedData, UserQuery
from graph_bot.pipelines.build_trees import build_reasoning_trees_from_seeds
from graph_bot.pipelines.retrieve import retrieve_k_optimal_paths


def test_build_reasoning_trees_from_seeds(mock_settings):
    seeds = [
        SeedData(id="seed1", content="2 4 6 -> 12", metadata={"task": "game24"}),
    ]

    mock_settings.max_tree_depth = 1
    mock_settings.beam_width = 1

    with patch("graph_bot.pipelines.build_trees.HiARICLAdapter") as MockAdapter:
        mock_instance = MockAdapter.return_value
        mock_instance.generate.return_value = ["mock_tree"]

        trees = build_reasoning_trees_from_seeds(seeds)

        MockAdapter.assert_called_once()
        mock_instance.generate.assert_called_once_with(seeds)
        assert trees == ["mock_tree"]


def test_build_reasoning_trees_integration(mock_settings):
    seeds = [
        SeedData(id="seed1", content="2 4 6 -> 12", metadata={"task": "game24"}),
    ]

    mock_settings.max_tree_depth = 1
    mock_settings.beam_width = 1

    trees = build_reasoning_trees_from_seeds(seeds)

    assert len(trees) == 1
    tree = trees[0]
    assert isinstance(tree.provenance, dict)
    assert tree.provenance.get("seed_id") == "seed1"
    assert tree.provenance.get("adapter") == "HiARICLAdapter"
    assert len(tree.nodes) > 0


def test_retrieve_k_optimal_paths_calls_adapter(mock_settings):
    query = UserQuery(id="q1", question="test query")

    mock_adapter = MagicMock()
    mock_result = RetrievalResult(
        query_id="q1",
        paths=[RetrievalPath(path_id="p1", node_ids=["n1"], score=1.0)],
        concatenated_context="context",
    )
    mock_adapter.retrieve_paths.return_value = mock_result

    result = retrieve_k_optimal_paths(query, adapter=mock_adapter, k=5)

    mock_adapter.retrieve_paths.assert_called_once_with(query=query, k=5)
    assert result == mock_result
    assert isinstance(result, RetrievalResult)


def test_retrieve_k_optimal_paths_default_adapter(mock_settings):
    query = UserQuery(id="q1", question="test query")

    with patch("graph_bot.pipelines.retrieve.GraphRAGAdapter") as MockAdapter:
        mock_instance = MockAdapter.return_value
        mock_instance.retrieve_paths.return_value = RetrievalResult(
            query_id="q1", paths=[], concatenated_context=""
        )

        retrieve_k_optimal_paths(query, k=3)

        MockAdapter.assert_called_once()
        mock_instance.retrieve_paths.assert_called_once()
