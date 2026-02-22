from __future__ import annotations

from unittest.mock import patch

from graph_bot.adapters.graphrag import GraphRAGAdapter
from graph_bot.datatypes import ReasoningNode, ReasoningTree


def test_insert_trees_bootstraps_edge_for_cold_start(tmp_path):
    metagraph_path = tmp_path / "metagraph.json"
    with patch("graph_bot.adapters.graphrag.settings.metagraph_path", metagraph_path):
        adapter = GraphRAGAdapter()

    tree = ReasoningTree(
        tree_id="t_bootstrap",
        root_id="n1",
        nodes=[
            ReasoningNode(node_id="n1", text="root", type="thought"),
            ReasoningNode(node_id="n2", text="child", type="thought"),
        ],
        edges=[],
        provenance={"task": "test-task"},
    )

    adapter.insert_trees([tree])

    graph = adapter.export_graph()
    assert len(graph.edges) > 0
    assert graph.edges[0].relation == "bootstrap"
    assert graph.edges[0].attributes["stats"]["n_traverse"] == 1
