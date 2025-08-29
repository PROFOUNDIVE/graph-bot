from __future__ import annotations

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


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


class RetrievalPath(BaseModel):
    path_id: str
    node_ids: List[str]
    score: Optional[float] = None


class RetrievalResult(BaseModel):
    query_id: str
    paths: List[RetrievalPath]
    concatenated_context: str


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
