from __future__ import annotations

from typing import Iterable, List

from ..types import (
    ReasoningTree,
    ReasoningNode,
    ReasoningEdge,
    RetrievalPath,
    RetrievalResult,
    UserQuery,
)
from ..settings import settings


class GraphRAGAdapter:
    """Stub adapter to persist and retrieve reasoning graphs.

    Replace with actual DB client (e.g., Neo4j, Memgraph, SQLite-backed graph, etc.).
    """

    def __init__(self) -> None:
        self.uri = settings.graphrag_uri
        # In-memory placeholder store
        self._trees: List[ReasoningTree] = []

    # --- Write APIs ---
    def insert_trees(self, trees: Iterable[ReasoningTree]) -> int:
        count = 0
        for tree in trees:
            self._trees.append(tree)
            count += 1
        return count

    def upsert_tidied(
        self, tree_id: str, nodes: List[ReasoningNode], edges: List[ReasoningEdge]
    ) -> None:
        # Replace nodes/edges of matching tree, simplistic logic
        for idx, t in enumerate(self._trees):
            if t.tree_id == tree_id:
                self._trees[idx] = ReasoningTree(
                    tree_id=t.tree_id,
                    root_id=t.root_id,
                    nodes=nodes or t.nodes,
                    edges=edges or t.edges,
                    provenance=t.provenance,
                )
                return

    # --- Read APIs ---
    def retrieve_paths(self, query: UserQuery, k: int) -> RetrievalResult:
        # Very naive scoring: choose first k chains from the most recent tree
        if not self._trees:
            return RetrievalResult(query_id=query.id, paths=[], concatenated_context="")

        tree = self._trees[-1]
        # Construct linear paths by following edges outward from root breadth-first
        children_map = {}
        for e in tree.edges:
            children_map.setdefault(e.src, []).append(e.dst)

        paths: List[RetrievalPath] = []

        def dfs(path: List[str]) -> None:
            if len(paths) >= k:
                return
            last = path[-1]
            if last not in children_map:
                paths.append(
                    RetrievalPath(
                        path_id="/".join(path), node_ids=path, score=float(len(path))
                    )
                )
                return
            for child in children_map[last]:
                dfs(path + [child])

        dfs([tree.root_id])
        paths = paths[:k]

        node_map = {n.node_id: n for n in tree.nodes}
        context_chunks: List[str] = []
        for p in paths:
            context_chunks.extend(
                [node_map[nid].text for nid in p.node_ids if nid in node_map]
            )
        concatenated_context = "\n".join(context_chunks)

        return RetrievalResult(
            query_id=query.id, paths=paths, concatenated_context=concatenated_context
        )
