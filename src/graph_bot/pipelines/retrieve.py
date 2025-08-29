from __future__ import annotations

from ..adapters.graphrag import GraphRAGAdapter
from ..types import UserQuery, RetrievalResult
from ..settings import settings


def retrieve_k_optimal_paths(query: UserQuery) -> RetrievalResult:
    adapter = GraphRAGAdapter()
    return adapter.retrieve_paths(query=query, k=settings.top_k_paths)
