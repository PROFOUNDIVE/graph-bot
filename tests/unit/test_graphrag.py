from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from graph_bot.adapters.graphrag import GraphRAGAdapter
from graph_bot.datatypes import (
    PathEvaluation,
    ReasoningEdge,
    ReasoningNode,
    ReasoningTree,
    UserQuery,
)


@pytest.fixture
def adapter(tmp_path):
    metagraph_path = tmp_path / "metagraph.json"
    with patch("graph_bot.adapters.graphrag.settings.metagraph_path", metagraph_path):
        yield GraphRAGAdapter()


def test_insert_trees_canonical_key_and_dedup(adapter):
    node1 = ReasoningNode(node_id="n1", text="Hello World", type="thought")
    node2 = ReasoningNode(node_id="n2", text="hello world", type="thought")
    node3 = ReasoningNode(node_id="n3", text="Different Text", type="thought")

    tree1 = ReasoningTree(
        tree_id="t1",
        root_id="n1",
        nodes=[node1, node3],
        edges=[ReasoningEdge(src="n1", dst="n3")],
        provenance={"task": "test-task"},
    )

    tree2 = ReasoningTree(
        tree_id="t2",
        root_id="n2",
        nodes=[node2],
        edges=[],
        provenance={"task": "test-task"},
    )

    adapter.insert_trees([tree1, tree2])

    graph = adapter.export_graph()

    assert len(graph.nodes) == 2

    node_texts = {node.text.lower() for node in graph.nodes}
    assert "hello world" in node_texts
    assert "different text" in node_texts

    hw_node = next(n for n in graph.nodes if n.text.lower() == "hello world")
    assert hw_node.attributes["stats"]["n_seen"] == 2


def test_update_with_feedback_ema(adapter):
    node = ReasoningNode(node_id="n1", text="Test Node", type="thought")
    tree = ReasoningTree(
        tree_id="t1",
        root_id="n1",
        nodes=[node],
        edges=[],
        provenance={"task": "test-task"},
    )
    adapter.insert_trees([tree])

    graph = adapter.export_graph()
    meta_node_id = graph.nodes[0].node_id

    assert graph.nodes[0].attributes["stats"]["ema_success"] == 0.0

    eval1 = PathEvaluation(
        path_id="p1",
        node_ids=[meta_node_id],
        success=True,
        tokens=100,
        latency_ms=50.0,
        cost_usd=0.01,
    )
    adapter.update_with_feedback([eval1])

    graph = adapter.export_graph()
    stats = graph.nodes[0].attributes["stats"]
    assert pytest.approx(stats["ema_success"]) == 0.1
    assert stats["n_success"] == 1
    assert stats["avg_tokens"] == 100.0

    eval2 = PathEvaluation(
        path_id="p2",
        node_ids=[meta_node_id],
        success=False,
        tokens=200,
        latency_ms=150.0,
        cost_usd=0.02,
    )
    adapter.update_with_feedback([eval2])

    graph = adapter.export_graph()
    stats = graph.nodes[0].attributes["stats"]
    assert pytest.approx(stats["ema_success"]) == 0.09
    assert stats["n_fail"] == 1
    assert stats["avg_tokens"] == 150.0


def test_time_decay(adapter):
    node = ReasoningNode(node_id="n1", text="Decay Node", type="thought")
    tree = ReasoningTree(
        tree_id="t1",
        root_id="n1",
        nodes=[node],
        edges=[],
        provenance={"task": "test-task"},
    )
    adapter.insert_trees([tree])

    graph = adapter.export_graph()
    graph.nodes[0].attributes["stats"]["ema_success"] = 1.0
    now = datetime.now(timezone.utc)
    graph.nodes[0].attributes["last_used_at"] = now.isoformat()
    adapter.import_graph(graph)

    future_time = now + timedelta(days=7)

    with patch("graph_bot.adapters.graphrag.datetime") as mock_datetime:
        mock_datetime.now.return_value = future_time
        mock_datetime.fromisoformat.side_effect = lambda x: datetime.fromisoformat(x)
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        adapter.prune_graph()

    graph = adapter.export_graph()
    ema = graph.nodes[0].attributes["stats"]["ema_success"]

    assert pytest.approx(ema) == math.exp(-1)


def test_prune_graph(adapter):
    node1 = ReasoningNode(node_id="n1", text="Bad Node", type="thought")
    node2 = ReasoningNode(node_id="n2", text="New Node", type="thought")
    node3 = ReasoningNode(node_id="n3", text="Good Node", type="thought")
    node4 = ReasoningNode(node_id="n4", text="Old Inactive Node", type="thought")

    tree = ReasoningTree(
        tree_id="t1",
        root_id="n1",
        nodes=[node1, node2, node3, node4],
        edges=[ReasoningEdge(src="n1", dst="n2"), ReasoningEdge(src="n2", dst="n3")],
        provenance={"task": "test-task"},
    )
    adapter.insert_trees([tree])

    graph = adapter.export_graph()
    now = datetime.now(timezone.utc)
    for node in graph.nodes:
        if "Bad Node" in node.text:
            node.attributes["stats"]["n_seen"] = 10
            node.attributes["stats"]["ema_success"] = 0.1
        elif "New Node" in node.text:
            node.attributes["stats"]["n_seen"] = 1
            node.attributes["stats"]["ema_success"] = 0.1
        elif "Good Node" in node.text:
            node.attributes["stats"]["n_seen"] = 10
            node.attributes["stats"]["ema_success"] = 0.9
        elif "Old Inactive Node" in node.text:
            node.attributes["stats"]["n_seen"] = 1
            node.attributes["stats"]["ema_success"] = 0.1
            old_time = now - timedelta(days=10)
            node.attributes["last_used_at"] = old_time.isoformat()

    adapter.import_graph(graph)

    pruned_count = adapter.prune_graph()
    assert pruned_count == 2

    graph = adapter.export_graph()
    assert len(graph.nodes) == 2
    assert not any("Bad Node" in n.text for n in graph.nodes)
    assert not any("Old Inactive Node" in n.text for n in graph.nodes)
    assert len(graph.edges) == 1
    assert graph.edges[0].dst == next(
        n.node_id for n in graph.nodes if "Good Node" in n.text
    )


def test_persistence(tmp_path):
    metagraph_path = tmp_path / "persistent_meta.json"

    with patch("graph_bot.adapters.graphrag.settings.metagraph_path", metagraph_path):
        adapter1 = GraphRAGAdapter()
        node = ReasoningNode(node_id="n1", text="Persistent", type="thought")
        tree = ReasoningTree(
            tree_id="t1",
            root_id="n1",
            nodes=[node],
            edges=[],
            provenance={"task": "test-task"},
        )
        adapter1.insert_trees([tree])

        assert metagraph_path.exists()

        adapter2 = GraphRAGAdapter()
    graph = adapter2.export_graph()
    assert len(graph.nodes) == 1
    assert graph.nodes[0].text == "Persistent"


def test_retrieve_paths(adapter):
    node1 = ReasoningNode(node_id="n1", text="Target Query", type="thought")
    tree = ReasoningTree(
        tree_id="t1",
        root_id="n1",
        nodes=[node1],
        edges=[],
        provenance={"task": "test-task"},
    )
    adapter.insert_trees([tree])

    query = UserQuery(id="q1", question="target query")
    result = adapter.retrieve_paths(query, k=1)

    assert result.query_id == "q1"
    assert len(result.paths) == 1
    assert "target query" in result.concatenated_context.lower()
