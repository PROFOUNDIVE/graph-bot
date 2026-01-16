from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StreamCallMetrics(BaseModel):
    call_id: str
    parent_id: str | None = None
    t: int
    problem_id: str
    operation: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    api_cost_usd: float | None = None
    error_type: str | None = None


class StreamProblemMetrics(BaseModel):
    t: int
    problem_id: str
    solved: bool
    attempts: int = 1
    attempt_success_rate: float | None = None
    llm_calls: int = 0
    tokens_total: int = 0
    latency_total_ms: float = 0.0
    api_cost_usd: float = 0.0
    retrieval_hit: bool = False
    reuse_count: int = 0
    memory_n_nodes: int = 0
    memory_n_edges: int = 0
    contamination_rate: float | None = None


class StreamCumulativeMetrics(BaseModel):
    t: int
    cumulative_solved: int
    cumulative_api_cost_usd: float
    cost_per_solved: float
    contamination_rate: float | None = None


class SeedData(BaseModel):
    """Minimal unit to start HiAR-ICL tree generation."""

    id: str
    content: str
    metadata: Dict[str, Any] | None = None


class UserQuery(BaseModel):
    """Incoming query used for retrieval and answer generation."""

    id: str
    question: str
    metadata: Dict[str, Any] | None = None


class ReasoningNode(BaseModel):
    node_id: str
    text: str
    type: str = Field(default="thought")  # thought | evidence | action | answer
    score: Optional[float] = None
    attributes: Dict[str, Any] | None = None


class ReasoningEdge(BaseModel):
    src: str
    dst: str
    relation: str = Field(default="leads_to")
    attributes: Dict[str, Any] | None = None


class ReasoningTree(BaseModel):
    """A tree-shaped reasoning artifact. Represented as nodes + edges with a root."""

    tree_id: str
    root_id: str
    nodes: List[ReasoningNode]
    edges: List[ReasoningEdge]
    provenance: Dict[str, Any] | None = None


class MetaGraph(BaseModel):
    """Persistent graph buffer built from multiple reasoning trees."""

    graph_id: str
    nodes: List[ReasoningNode] = Field(default_factory=list)
    edges: List[ReasoningEdge] = Field(default_factory=list)
    metadata: Dict[str, Any] | None = None


class RetrievalPath(BaseModel):
    path_id: str
    node_ids: List[str]
    score: Optional[float] = None


class RetrievalResult(BaseModel):
    query_id: str
    paths: List[RetrievalPath]
    concatenated_context: str


class PathEvaluation(BaseModel):
    path_id: str
    node_ids: List[str]
    success: bool
    tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None


class LLMAnswer(BaseModel):
    query_id: str
    answer: str
    metadata: Dict[str, Any] | None = None


class TidiedElements(BaseModel):
    """Represents reranked/verbalized/pruned/augmented elements ready to persist."""

    tree_id: str
    updated_nodes: List[ReasoningNode] = Field(default_factory=list)
    updated_edges: List[ReasoningEdge] = Field(default_factory=list)
    notes: Dict[str, Any] | None = None
