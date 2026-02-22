from __future__ import annotations

import math
from unittest.mock import patch

from graph_bot.adapters.embeddings import DeterministicHashEmbeddingProvider
from graph_bot.adapters.graphrag import GraphRAGAdapter
from graph_bot.datatypes import ReasoningNode, ReasoningTree, UserQuery


def _cosine(a: list[float], b: list[float]) -> float:
    a_norm = math.sqrt(sum(value * value for value in a))
    b_norm = math.sqrt(sum(value * value for value in b))
    if a_norm <= 0.0 or b_norm <= 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (a_norm * b_norm)


def _find_dense_winner_text(
    provider: DeterministicHashEmbeddingProvider,
    *,
    query_text: str,
    sparse_text: str,
) -> str:
    query_vec = list(provider.encode([query_text])[0][0].tolist())
    sparse_vec = list(provider.encode([sparse_text])[0][0].tolist())
    sparse_score = _cosine(query_vec, sparse_vec)

    for idx in range(1, 2000):
        candidate = f"dense candidate {idx}"
        candidate_vec = list(provider.encode([candidate])[0][0].tolist())
        if _cosine(query_vec, candidate_vec) > sparse_score:
            return candidate

    raise AssertionError("Could not find deterministic dense winner candidate")


def _insert_fixture(
    adapter: GraphRAGAdapter, dense_text: str, sparse_text: str
) -> None:
    tree = ReasoningTree(
        tree_id="dense-fixture",
        root_id="n-sparse",
        nodes=[
            ReasoningNode(node_id="n-sparse", text=sparse_text, type="thought"),
            ReasoningNode(node_id="n-dense", text=dense_text, type="thought"),
            ReasoningNode(node_id="n-other", text="unrelated template", type="thought"),
        ],
        edges=[],
        provenance={"task": "test-task"},
    )
    adapter.insert_trees([tree])


def test_dense_template_deterministic_top1_differs_from_sparse(tmp_path, monkeypatch):
    provider = DeterministicHashEmbeddingProvider(dim=64)
    monkeypatch.setattr(
        GraphRAGAdapter,
        "_build_embedding_provider",
        lambda _self: provider,
    )

    query_text = "alpha beta"
    sparse_text = "alpha beta overlap anchor"
    dense_text = _find_dense_winner_text(
        provider,
        query_text=query_text,
        sparse_text=sparse_text,
    )

    sparse_path = tmp_path / "metagraph_sparse.json"
    with patch("graph_bot.adapters.graphrag.settings.metagraph_path", sparse_path):
        sparse_adapter = GraphRAGAdapter(retrieval_backend="sparse_jaccard")
    _insert_fixture(sparse_adapter, dense_text=dense_text, sparse_text=sparse_text)
    sparse_lookup = {
        node.text: node.node_id for node in sparse_adapter.export_graph().nodes
    }
    sparse_result = sparse_adapter.retrieve_paths(
        UserQuery(id="q-sparse", question=query_text, metadata={"task": "test-task"}),
        k=1,
    )
    sparse_top = sparse_result.paths[0].node_ids[0]
    assert sparse_top == sparse_lookup[sparse_text]

    dense_path = tmp_path / "metagraph_dense.json"
    with patch("graph_bot.adapters.graphrag.settings.metagraph_path", dense_path):
        dense_adapter = GraphRAGAdapter(retrieval_backend="dense_template")
    _insert_fixture(dense_adapter, dense_text=dense_text, sparse_text=sparse_text)
    dense_lookup = {
        node.text: node.node_id for node in dense_adapter.export_graph().nodes
    }
    dense_result = dense_adapter.retrieve_paths(
        UserQuery(id="q-dense", question=query_text, metadata={"task": "test-task"}),
        k=1,
    )
    dense_top = dense_result.paths[0].node_ids[0]

    assert dense_top == dense_lookup[dense_text]
    assert dense_top != dense_lookup[sparse_text]
    assert dense_top != sparse_top


def test_insert_trees_lazy_cache_fill_after_dense_retrieval(tmp_path, monkeypatch):
    provider = DeterministicHashEmbeddingProvider(dim=64)
    monkeypatch.setattr(
        GraphRAGAdapter,
        "_build_embedding_provider",
        lambda _self: provider,
    )

    dense_path = tmp_path / "metagraph_dense_cache.json"
    with patch("graph_bot.adapters.graphrag.settings.metagraph_path", dense_path):
        adapter = GraphRAGAdapter(retrieval_backend="dense_template")

    _insert_fixture(adapter, dense_text="dense cache node", sparse_text="sparse cache")
    graph = adapter.export_graph()
    node_ids = {node.node_id for node in graph.nodes}
    node_lookup = {node.text: node.node_id for node in graph.nodes}

    assert node_ids
    assert adapter._node_embedding_cache == {}

    result = adapter.retrieve_paths(
        UserQuery(
            id="q-cache",
            question="dense cache node",
            metadata={"task": "test-task"},
        ),
        k=1,
    )

    assert result.paths
    assert result.paths[0].node_ids[0] == node_lookup["dense cache node"]
    assert node_ids.issubset(set(adapter._node_embedding_cache.keys()))
