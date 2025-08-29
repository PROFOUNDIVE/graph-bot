from __future__ import annotations

from typing import Iterable

from ..core.models import SeedData, UserQuery, LLMAnswer
from ..adapters.graphrag import GraphRAGAdapter
from .build_trees import build_reasoning_trees_from_seeds
from .retrieve import retrieve_k_optimal_paths


def process_seeds_and_write(seeds: Iterable[SeedData]) -> int:
    _adapter = GraphRAGAdapter()
    trees = build_reasoning_trees_from_seeds(seeds)
    written = _adapter.insert_trees(trees)
    return written


def postprocess_after_T_inputs(trees_count: int | None = None) -> int:
    # Pull all currently known trees from adapter (stub: internal state only)
    _adapter = GraphRAGAdapter()
    # This stub does not expose list, so this function is a placeholder.
    # In a real implementation, fetch recent T trees. Here we do nothing.
    return 0


def answer_with_retrieval(query: UserQuery) -> LLMAnswer:
    retrieval = retrieve_k_optimal_paths(query)
    # Concatenate query with paths and call LLM (stubbed as echo)
    answer_text = f"Q: {query.question}\n\nContext:\n{retrieval.concatenated_context}\n\nA: [stubbed answer]"
    return LLMAnswer(query_id=query.id, answer=answer_text)
