from graph_bot.adapters.graphrag import GraphRAGAdapter
from graph_bot.datatypes import MetaGraph, ReasoningNode, ReasoningEdge, UserQuery
from graph_bot.settings import settings


class TestEdgeScoring:
    def test_edge_scoring_affects_rank(self, tmp_path):
        """
        Verify that when use_edges=True, paths with higher edge EMA success
        are ranked higher than paths with lower edge EMA success, assuming
        node scores are identical.
        """
        # 1. Setup temporary file for graph
        settings.metagraph_path = tmp_path / "metagraph.json"

        # 2. Create Nodes
        # 'root' -> 'target'
        root = ReasoningNode(
            node_id="root",
            text="target root",
            type="thought",
            score=1.0,
            attributes={"stats": {"ema_success": 0.5}},  # Neutral node stats
        )
        # Two target nodes with identical text (same semantic score)
        bad_target = ReasoningNode(
            node_id="bad",
            text="target solution",
            type="thought",
            score=1.0,
            attributes={"stats": {"ema_success": 0.5}},
        )
        good_target = ReasoningNode(
            node_id="good",
            text="target solution",
            type="thought",
            score=1.0,
            attributes={"stats": {"ema_success": 0.5}},
        )

        # 3. Create Edges with DIFFERENT stats
        # Root -> Bad (Low success)
        edge_bad = ReasoningEdge(
            src="root",
            dst="bad",
            relation="next",
            attributes={"stats": {"ema_success": 0.1, "n_traverse": 10}},
        )

        # Root -> Good (High success)
        edge_good = ReasoningEdge(
            src="root",
            dst="good",
            relation="next",
            attributes={"stats": {"ema_success": 0.9, "n_traverse": 10}},
        )

        graph = MetaGraph(
            graph_id="test_graph",
            nodes=[root, bad_target, good_target],
            edges=[edge_bad, edge_good],
        )

        # 4. Initialize Adapter
        adapter = GraphRAGAdapter(use_edges=True)
        adapter._graph = graph

        # 5. Query matching "target solution"
        query = UserQuery(id="q1", question="target solution")

        # 6. Retrieve
        result = adapter.retrieve_paths(query, k=5)

        # Filter for paths of length 2 starting with root
        paths_len_2 = [
            p for p in result.paths if len(p.node_ids) == 2 and p.node_ids[0] == "root"
        ]

        assert (
            len(paths_len_2) >= 2
        ), f"Expected at least 2 paths of length 2, got {len(paths_len_2)}"

        # Identify which path is which
        path_good = next((p for p in paths_len_2 if "good" in p.node_ids), None)
        path_bad = next((p for p in paths_len_2 if "bad" in p.node_ids), None)

        assert path_good is not None
        assert path_bad is not None
        assert path_good.score is not None
        assert path_bad.score is not None

        # With edge scoring, Good > Bad
        assert path_good.score > path_bad.score

    def test_without_edge_scoring_logic_check(self, tmp_path):
        """
        Verify that if we DON'T consider edges (simulated by having equal edge stats),
        the scores are equal.
        """
        settings.metagraph_path = tmp_path / "metagraph.json"

        root = ReasoningNode(
            node_id="root",
            text="target root",
            type="thought",
            score=1.0,
            attributes={"stats": {"ema_success": 0.5}},
        )
        bad = ReasoningNode(
            node_id="bad",
            text="target solution",
            type="thought",
            score=1.0,
            attributes={"stats": {"ema_success": 0.5}},
        )
        good = ReasoningNode(
            node_id="good",
            text="target solution",
            type="thought",
            score=1.0,
            attributes={"stats": {"ema_success": 0.5}},
        )

        # Equal edge stats
        edge_bad = ReasoningEdge(
            src="root",
            dst="bad",
            relation="next",
            attributes={"stats": {"ema_success": 0.5}},
        )
        edge_good = ReasoningEdge(
            src="root",
            dst="good",
            relation="next",
            attributes={"stats": {"ema_success": 0.5}},
        )

        graph = MetaGraph(
            graph_id="test_graph", nodes=[root, bad, good], edges=[edge_bad, edge_good]
        )

        adapter = GraphRAGAdapter(use_edges=True)
        adapter._graph = graph
        query = UserQuery(id="q1", question="target solution")

        result = adapter.retrieve_paths(query, k=5)
        paths_len_2 = [
            p for p in result.paths if len(p.node_ids) == 2 and p.node_ids[0] == "root"
        ]

        path_good = next((p for p in paths_len_2 if "good" in p.node_ids), None)
        path_bad = next((p for p in paths_len_2 if "bad" in p.node_ids), None)

        assert path_good is not None
        assert path_bad is not None
        assert path_good.score is not None
        assert path_bad.score is not None

        # Should be equal (floating point tolerance)
        assert abs(path_good.score - path_bad.score) < 1e-6

    def test_flat_template_rag_initialization_fix(self, tmp_path):
        """
        Regression test: Ensure edge_index is initialized in flat_template_rag mode,
        preventing UnboundLocalError.
        """
        settings.metagraph_path = tmp_path / "metagraph.json"
        settings.rerank_top_n = 5

        # Create a simple graph with one node
        node_a = ReasoningNode(
            node_id="A",
            text="test node",
            type="thought",
            score=1.0,
            attributes={"stats": {"ema_success": 0.5}},
        )

        graph = MetaGraph(graph_id="test_flat", nodes=[node_a], edges=[])

        adapter = GraphRAGAdapter(mode="flat_template_rag", use_edges=False)
        adapter._graph = graph

        query = UserQuery(id="q1", question="test node")

        # This call triggered UnboundLocalError before the fix
        result = adapter.retrieve_paths(query, k=1)

        assert len(result.paths) == 1
        assert result.paths[0].node_ids == ["A"]
