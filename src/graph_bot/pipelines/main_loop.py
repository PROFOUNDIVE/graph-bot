from __future__ import annotations
from __future__ import annotations

from typing import Iterable

from ..types import SeedData, UserQuery, LLMAnswer, RetrievalResult
from ..adapters.graphrag import GraphRAGAdapter
from .build_trees import build_reasoning_trees_from_seeds
from .retrieve import retrieve_k_optimal_paths


def process_seeds_and_write(seeds: Iterable[SeedData]) -> int:
    _adapter = GraphRAGAdapter()
    trees = build_reasoning_trees_from_seeds(seeds)
    written = _adapter.insert_trees(trees)
    return written


def postprocess_after_T_inputs(trees_count: int | None = None) -> int:
    del trees_count
    adapter = GraphRAGAdapter()
    return adapter.prune_graph()


def answer_with_retrieval(
    query: UserQuery,
    *,
    retrieval: RetrievalResult | None = None,
    adapter: GraphRAGAdapter | None = None,
) -> LLMAnswer:
    active_adapter = adapter or GraphRAGAdapter()
    active_retrieval = retrieval or retrieve_k_optimal_paths(
        query, adapter=active_adapter
    )
    active_adapter.register_usage(active_retrieval.paths)
    answer_text = f"Q: {query.question}\n\nContext:\n{active_retrieval.concatenated_context}\n\nA: [stubbed answer]"
    return LLMAnswer(query_id=query.id, answer=answer_text)
