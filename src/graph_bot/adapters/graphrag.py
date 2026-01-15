from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..settings import settings
from ..types import (
    MetaGraph,
    PathEvaluation,
    ReasoningEdge,
    ReasoningNode,
    ReasoningTree,
    RetrievalPath,
    RetrievalResult,
    UserQuery,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def _hash_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _default_node_stats() -> Dict[str, Any]:
    return {
        "n_seen": 0,
        "n_used": 0,
        "n_success": 0,
        "n_fail": 0,
        "ema_success": 0.0,
        "avg_tokens": 0.0,
        "avg_latency_ms": 0.0,
        "avg_cost_usd": 0.0,
    }


def _default_edge_stats() -> Dict[str, Any]:
    return {
        "n_traverse": 0,
        "n_success": 0,
        "n_fail": 0,
        "ema_success": 0.0,
    }


def _ensure_node_attributes(
    attributes: Dict[str, Any] | None, *, task: str, canonical_key: str
) -> Dict[str, Any]:
    attrs = dict(attributes or {})
    attrs.setdefault("task", task)
    attrs.setdefault("subtype", "template")
    attrs.setdefault("canonical_key", canonical_key)
    attrs.setdefault("slots", [])
    attrs.setdefault("created_at", _now_iso())
    attrs.setdefault("last_used_at", attrs.get("created_at", _now_iso()))
    stats = attrs.get("stats")
    if not isinstance(stats, dict):
        stats = _default_node_stats()
    else:
        for key, value in _default_node_stats().items():
            stats.setdefault(key, value)
    attrs["stats"] = stats
    quality = attrs.get("quality")
    if not isinstance(quality, dict):
        quality = {"validator_passed": False, "parse_ok": True}
    else:
        quality.setdefault("validator_passed", False)
        quality.setdefault("parse_ok", True)
    attrs["quality"] = quality
    return attrs


def _ensure_edge_attributes(
    attributes: Dict[str, Any] | None, *, task: str
) -> Dict[str, Any]:
    attrs = dict(attributes or {})
    attrs.setdefault("task", task)
    stats = attrs.get("stats")
    if not isinstance(stats, dict):
        stats = _default_edge_stats()
    else:
        for key, value in _default_edge_stats().items():
            stats.setdefault(key, value)
    attrs["stats"] = stats
    return attrs


def _update_running_avg(
    stats: Dict[str, Any], *, key: str, value: float | None, count: int
) -> None:
    if value is None or count <= 0:
        return
    current = float(stats.get(key, 0.0))
    stats[key] = (current * (count - 1) + value) / count


def _update_ema(stats: Dict[str, Any], *, success: bool) -> None:
    alpha = settings.ema_alpha
    previous = float(stats.get("ema_success", 0.0))
    stats["ema_success"] = (1 - alpha) * previous + alpha * (1.0 if success else 0.0)


def _apply_time_decay(stats: Dict[str, Any], *, last_used_at: str | None) -> None:
    tau_days = max(settings.ema_tau_days, 1)
    delta_days = (
        datetime.now(timezone.utc) - _parse_time(last_used_at)
    ).total_seconds() / 86400
    decay = math.exp(-delta_days / tau_days)
    stats["ema_success"] = float(stats.get("ema_success", 0.0)) * decay


def _semantic_similarity(query_tokens: List[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    query_set = set(query_tokens)
    token_set = set(tokens)
    overlap = len(query_set & token_set)
    union = len(query_set | token_set)
    return overlap / union if union else 0.0


class GraphRAGAdapter:
    """Stub adapter to persist and retrieve reasoning graphs.

    Replace with actual DB client (e.g., Neo4j, Memgraph, SQLite-backed graph, etc.).
    """

    def __init__(self) -> None:
        self.uri = settings.graphrag_uri
        self.store_path = Path(settings.metagraph_path)
        self._graph = self._load_or_init_graph()

    def _load_or_init_graph(self) -> MetaGraph:
        if self.store_path.exists():
            with self.store_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return MetaGraph.model_validate(data)
        return MetaGraph(
            graph_id="metagraph-v0.1",
            metadata={"created_at": _now_iso(), "version": "v0.1"},
        )

    def _save_graph(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("w", encoding="utf-8") as f:
            json.dump(self._graph.model_dump(), f, ensure_ascii=False, indent=2)

    def _touch_metadata(self) -> None:
        metadata = dict(self._graph.metadata or {})
        metadata.setdefault("created_at", _now_iso())
        metadata.setdefault("version", "v0.1")
        metadata["last_updated_at"] = _now_iso()
        self._graph.metadata = metadata

    def _index_nodes(self) -> Tuple[Dict[str, ReasoningNode], Dict[str, str]]:
        node_index = {node.node_id: node for node in self._graph.nodes}
        canonical_index: Dict[str, str] = {}
        for node in node_index.values():
            canonical_key = _normalize_text(node.text)
            task = "unknown"
            if node.attributes and "task" in node.attributes:
                task = str(node.attributes["task"])
            if node.attributes and "canonical_key" in node.attributes:
                canonical_key = str(node.attributes["canonical_key"])
            node.attributes = _ensure_node_attributes(
                node.attributes, task=task, canonical_key=canonical_key
            )
            canonical_index[canonical_key] = node.node_id
        return node_index, canonical_index

    def export_graph(self) -> MetaGraph:
        return self._graph

    def import_graph(self, graph: MetaGraph) -> None:
        self._graph = graph
        self._touch_metadata()
        self._save_graph()

    # --- Write APIs ---
    def insert_trees(self, trees: Iterable[ReasoningTree]) -> int:
        node_index, canonical_index = self._index_nodes()
        edge_index = {(edge.src, edge.dst): edge for edge in self._graph.edges}
        count = 0

        for tree in trees:
            task = "unknown"
            if tree.provenance and "task" in tree.provenance:
                task = str(tree.provenance["task"])
            now = _now_iso()
            node_id_map: Dict[str, str] = {}

            for node in tree.nodes:
                canonical_key = _normalize_text(node.text)
                existing_id = canonical_index.get(canonical_key)
                if existing_id and existing_id in node_index:
                    meta_node = node_index[existing_id]
                    meta_task = task
                    if meta_node.attributes and "task" in meta_node.attributes:
                        meta_task = str(meta_node.attributes["task"])
                    meta_node.attributes = _ensure_node_attributes(
                        meta_node.attributes,
                        task=meta_task,
                        canonical_key=canonical_key,
                    )
                    stats = meta_node.attributes["stats"]
                    stats["n_seen"] = int(stats.get("n_seen", 0)) + 1
                    meta_node.attributes["last_used_at"] = now
                    node_id_map[node.node_id] = existing_id
                else:
                    node_id = f"node-{_hash_key(canonical_key)}"
                    attributes = _ensure_node_attributes(
                        node.attributes, task=task, canonical_key=canonical_key
                    )
                    attributes["created_at"] = now
                    attributes["last_used_at"] = now
                    stats = attributes["stats"]
                    stats["n_seen"] = int(stats.get("n_seen", 0)) + 1
                    meta_node = ReasoningNode(
                        node_id=node_id,
                        text=node.text,
                        type=node.type,
                        score=node.score,
                        attributes=attributes,
                    )
                    node_index[node_id] = meta_node
                    canonical_index[canonical_key] = node_id
                    node_id_map[node.node_id] = node_id

            for edge in tree.edges:
                src = node_id_map.get(edge.src)
                dst = node_id_map.get(edge.dst)
                if not src or not dst:
                    continue
                key = (src, dst)
                if key in edge_index:
                    existing_edge = edge_index[key]
                    existing_edge.attributes = _ensure_edge_attributes(
                        existing_edge.attributes, task=task
                    )
                    stats = existing_edge.attributes["stats"]
                    stats["n_traverse"] = int(stats.get("n_traverse", 0)) + 1
                else:
                    attributes = _ensure_edge_attributes(edge.attributes, task=task)
                    stats = attributes["stats"]
                    stats["n_traverse"] = int(stats.get("n_traverse", 0)) + 1
                    edge_index[key] = ReasoningEdge(
                        src=src,
                        dst=dst,
                        relation=edge.relation,
                        attributes=attributes,
                    )

            count += 1

        self._graph.nodes = list(node_index.values())
        self._graph.edges = list(edge_index.values())
        self._touch_metadata()
        self._save_graph()
        return count

    def upsert_tidied(
        self, tree_id: str, nodes: List[ReasoningNode], edges: List[ReasoningEdge]
    ) -> None:
        tree = ReasoningTree(
            tree_id=tree_id,
            root_id=nodes[0].node_id if nodes else tree_id,
            nodes=nodes,
            edges=edges,
            provenance={"source": "tidied"},
        )
        self.insert_trees([tree])

    def register_usage(self, paths: Iterable[RetrievalPath]) -> None:
        node_index = {node.node_id: node for node in self._graph.nodes}
        edge_index = {(edge.src, edge.dst): edge for edge in self._graph.edges}
        now = _now_iso()

        for path in paths:
            for node_id in path.node_ids:
                node = node_index.get(node_id)
                if not node:
                    continue
                task = "unknown"
                if node.attributes and "task" in node.attributes:
                    task = str(node.attributes["task"])
                canonical_key = _normalize_text(node.text)
                node.attributes = _ensure_node_attributes(
                    node.attributes, task=task, canonical_key=canonical_key
                )
                stats = node.attributes["stats"]
                stats["n_used"] = int(stats.get("n_used", 0)) + 1
                node.attributes["last_used_at"] = now

            for src, dst in zip(path.node_ids, path.node_ids[1:]):
                edge = edge_index.get((src, dst))
                if not edge:
                    continue
                task = "unknown"
                if edge.attributes and "task" in edge.attributes:
                    task = str(edge.attributes["task"])
                edge.attributes = _ensure_edge_attributes(edge.attributes, task=task)
                stats = edge.attributes["stats"]
                stats["n_traverse"] = int(stats.get("n_traverse", 0)) + 1

        self._touch_metadata()
        self._save_graph()

    def update_with_feedback(self, evaluations: Iterable[PathEvaluation]) -> None:
        node_index = {node.node_id: node for node in self._graph.nodes}
        edge_index = {(edge.src, edge.dst): edge for edge in self._graph.edges}

        for evaluation in evaluations:
            for node_id in evaluation.node_ids:
                node = node_index.get(node_id)
                if not node:
                    continue
                task = "unknown"
                if node.attributes and "task" in node.attributes:
                    task = str(node.attributes["task"])
                canonical_key = _normalize_text(node.text)
                node.attributes = _ensure_node_attributes(
                    node.attributes, task=task, canonical_key=canonical_key
                )
                stats = node.attributes["stats"]
                key = "n_success" if evaluation.success else "n_fail"
                stats[key] = int(stats.get(key, 0)) + 1
                _update_ema(stats, success=evaluation.success)
                total = int(stats.get("n_success", 0)) + int(stats.get("n_fail", 0))
                _update_running_avg(
                    stats, key="avg_tokens", value=evaluation.tokens, count=total
                )
                _update_running_avg(
                    stats,
                    key="avg_latency_ms",
                    value=evaluation.latency_ms,
                    count=total,
                )
                _update_running_avg(
                    stats, key="avg_cost_usd", value=evaluation.cost_usd, count=total
                )

            for src, dst in zip(evaluation.node_ids, evaluation.node_ids[1:]):
                edge = edge_index.get((src, dst))
                if not edge:
                    continue
                task = "unknown"
                if edge.attributes and "task" in edge.attributes:
                    task = str(edge.attributes["task"])
                edge.attributes = _ensure_edge_attributes(edge.attributes, task=task)
                stats = edge.attributes["stats"]
                key = "n_success" if evaluation.success else "n_fail"
                stats[key] = int(stats.get(key, 0)) + 1
                _update_ema(stats, success=evaluation.success)

        self._touch_metadata()
        self._save_graph()

    # --- Read APIs ---
    def retrieve_paths(self, query: UserQuery, k: int) -> RetrievalResult:
        if not self._graph.nodes:
            return RetrievalResult(query_id=query.id, paths=[], concatenated_context="")

        query_tokens = _tokenize(query.question)
        node_index = {node.node_id: node for node in self._graph.nodes}
        node_scores: Dict[str, float] = {
            node_id: _semantic_similarity(query_tokens, node.text)
            for node_id, node in node_index.items()
        }
        sorted_nodes = sorted(
            node_scores.items(), key=lambda item: item[1], reverse=True
        )
        seed_nodes = [node_id for node_id, _ in sorted_nodes[: settings.rerank_top_n]]

        adjacency: Dict[str, List[str]] = {}
        for edge in self._graph.edges:
            adjacency.setdefault(edge.src, []).append(edge.dst)

        candidate_paths = self._build_candidate_paths(
            seed_nodes, adjacency, node_scores
        )
        scored_paths = []
        for path in candidate_paths:
            semantic_score, combined_score = self._score_path(
                path, node_scores, node_index
            )
            scored_paths.append((path, semantic_score, combined_score))

        semantic_sorted = sorted(scored_paths, key=lambda item: item[1], reverse=True)
        semantic_sorted = semantic_sorted[: settings.rerank_top_n]
        reranked = sorted(semantic_sorted, key=lambda item: item[2], reverse=True)
        selected = reranked[:k]

        paths: List[RetrievalPath] = []
        context_chunks: List[str] = []
        for path, _, score in selected:
            path_id = "/".join(path)
            paths.append(RetrievalPath(path_id=path_id, node_ids=path, score=score))
            for node_id in path:
                node = node_index.get(node_id)
                if node:
                    context_chunks.append(node.text)

        return RetrievalResult(
            query_id=query.id,
            paths=paths,
            concatenated_context="\n".join(context_chunks),
        )

    def _build_candidate_paths(
        self,
        seed_nodes: List[str],
        adjacency: Dict[str, List[str]],
        node_scores: Dict[str, float],
    ) -> List[List[str]]:
        max_depth = max(settings.max_tree_depth, 1)
        max_paths = max(settings.rerank_top_n, settings.top_k_paths)
        paths: List[List[str]] = []

        for seed in seed_nodes:
            queue: List[List[str]] = [[seed]]
            while queue and len(paths) < max_paths:
                path = queue.pop(0)
                last = path[-1]
                children = adjacency.get(last, [])
                if not children or len(path) >= max_depth:
                    paths.append(path)
                    continue
                sorted_children = sorted(
                    children,
                    key=lambda child: node_scores.get(child, 0.0),
                    reverse=True,
                )
                for child in sorted_children[: settings.beam_width]:
                    if child in path:
                        continue
                    queue.append(path + [child])
            if len(paths) >= max_paths:
                break

        if not paths:
            paths = [[node_id] for node_id in seed_nodes[:max_paths]]

        return paths

    def _score_path(
        self,
        node_ids: List[str],
        node_scores: Dict[str, float],
        node_index: Dict[str, ReasoningNode],
    ) -> Tuple[float, float]:
        if not node_ids:
            return 0.0, 0.0

        semantic_values = [node_scores.get(node_id, 0.0) for node_id in node_ids]
        stats_values = []
        cost_values = []
        for node_id in node_ids:
            node = node_index.get(node_id)
            if not node:
                continue
            task = "unknown"
            if node.attributes and "task" in node.attributes:
                task = str(node.attributes["task"])
            canonical_key = _normalize_text(node.text)
            node.attributes = _ensure_node_attributes(
                node.attributes, task=task, canonical_key=canonical_key
            )
            stats = node.attributes["stats"]
            _apply_time_decay(stats, last_used_at=node.attributes.get("last_used_at"))
            stats_values.append(float(stats.get("ema_success", 0.0)))
            cost_values.append(float(stats.get("avg_cost_usd", 0.0)))

        avg_semantic = sum(semantic_values) / len(semantic_values)
        avg_stats = sum(stats_values) / len(stats_values) if stats_values else 0.0
        avg_cost = sum(cost_values) / len(cost_values) if cost_values else 0.0
        length_penalty = 0.02 * max(len(node_ids) - 1, 0)
        combined = avg_semantic + avg_stats - length_penalty - (avg_cost * 0.01)
        return avg_semantic, combined

    def prune_graph(self) -> int:
        if not self._graph.nodes:
            return 0

        kept_nodes: List[ReasoningNode] = []
        kept_ids = set()
        pruned = 0
        updated = False

        for node in self._graph.nodes:
            task = "unknown"
            if node.attributes and "task" in node.attributes:
                task = str(node.attributes["task"])
            canonical_key = _normalize_text(node.text)
            node.attributes = _ensure_node_attributes(
                node.attributes, task=task, canonical_key=canonical_key
            )
            stats = node.attributes["stats"]
            before = float(stats.get("ema_success", 0.0))
            _apply_time_decay(stats, last_used_at=node.attributes.get("last_used_at"))
            if float(stats.get("ema_success", 0.0)) != before:
                updated = True
            n_seen = int(stats.get("n_seen", 0))
            ema_success = float(stats.get("ema_success", 0.0))
            last_used_at = node.attributes.get("last_used_at")
            age_days = (
                datetime.now(timezone.utc) - _parse_time(last_used_at)
            ).total_seconds() / 86400
            should_prune = (
                n_seen >= settings.ema_min_seen
                and ema_success < settings.ema_min_success
            )
            if (
                age_days > settings.ema_tau_days
                and ema_success < settings.ema_min_success
            ):
                should_prune = True

            if should_prune:
                pruned += 1
                updated = True
                continue

            kept_nodes.append(node)
            kept_ids.add(node.node_id)

        kept_edges = [
            edge
            for edge in self._graph.edges
            if edge.src in kept_ids and edge.dst in kept_ids
        ]

        self._graph.nodes = kept_nodes
        self._graph.edges = kept_edges

        if pruned or updated:
            self._touch_metadata()
            self._save_graph()
        return pruned
