from __future__ import annotations

from unittest.mock import patch

from graph_bot.adapters.graphrag import GraphRAGAdapter
from graph_bot.datatypes import ReasoningEdge, ReasoningNode, ReasoningTree


def test_insert_trees_resolves_edges_to_existing_graph_nodes(tmp_path) -> None:
    metagraph_path = tmp_path / "metagraph.json"
    with patch("graph_bot.adapters.graphrag.settings.metagraph_path", metagraph_path):
        adapter = GraphRAGAdapter()

    seed_tree = ReasoningTree(
        tree_id="seed-tree",
        root_id="seed",
        nodes=[ReasoningNode(node_id="seed", text="Seed Template", type="thought")],
        edges=[],
        provenance={"task": "test-task"},
    )
    adapter.insert_trees([seed_tree])
    seed_meta_id = adapter.export_graph().nodes[0].node_id

    followup_tree = ReasoningTree(
        tree_id="followup-tree",
        root_id="new-template",
        nodes=[
            ReasoningNode(
                node_id="new-template",
                text="Follow-up Template",
                type="thought",
            )
        ],
        edges=[
            ReasoningEdge(
                src=seed_meta_id,
                dst="new-template",
                relation="used_for",
            )
        ],
        provenance={"task": "test-task"},
    )

    adapter.insert_trees([followup_tree])
    graph = adapter.export_graph()
    followup_meta_id = next(
        node.node_id for node in graph.nodes if node.text == "Follow-up Template"
    )

    edge = next(
        (
            item
            for item in graph.edges
            if item.src == seed_meta_id and item.dst == followup_meta_id
        ),
        None,
    )
    assert edge is not None
    assert edge.relation == "used_for"
