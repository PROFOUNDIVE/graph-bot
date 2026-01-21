from __future__ import annotations
from __future__ import annotations

from ..adapters.graphrag import GraphRAGAdapter
from ..datatypes import UserQuery, RetrievalResult
from ..settings import settings


def retrieve_k_optimal_paths(
    query: UserQuery,
    *,
    adapter: GraphRAGAdapter | None = None,
    k: int | None = None,
) -> RetrievalResult:
    active_adapter = adapter or GraphRAGAdapter()
    return active_adapter.retrieve_paths(query=query, k=k or settings.top_k_paths)
