from __future__ import annotations

import uuid
from typing import Iterable, List

from .models import SeedData, ReasoningTree, ReasoningNode, ReasoningEdge
from .settings import settings


class HiARICLGenerator:
    """Stub for HiAR-ICL reasoning tree generation.

    Given seed data, produce a tree up to configured depth and beam width.
    Replace internal placeholders with calls to actual LLMs/heuristics later.
    """

    def __init__(self) -> None:
        self.max_depth = settings.max_tree_depth
        self.beam_width = settings.beam_width

    def generate(self, seeds: Iterable[SeedData]) -> List[ReasoningTree]:
        trees: List[ReasoningTree] = []
        for seed in seeds:
            tree_id = str(uuid.uuid4())
            root_id = f"root-{seed.id}"
            nodes: List[ReasoningNode] = [
                ReasoningNode(node_id=root_id, text=seed.content, type="thought"),
            ]
            edges: List[ReasoningEdge] = []

            # Simple deterministic branching stub
            current_level = [root_id]
            for depth in range(1, self.max_depth + 1):
                next_level: List[str] = []
                for parent_id in current_level:
                    for b in range(self.beam_width):
                        node_id = f"{parent_id}-{depth}-{b}"
                        nodes.append(
                            ReasoningNode(
                                node_id=node_id,
                                text=f"Thought d={depth} b={b} for seed={seed.id}",
                                type="thought",
                            )
                        )
                        edges.append(
                            ReasoningEdge(
                                src=parent_id, dst=node_id, relation="leads_to"
                            )
                        )
                        next_level.append(node_id)
                current_level = next_level

            trees.append(
                ReasoningTree(
                    tree_id=tree_id,
                    root_id=root_id,
                    nodes=nodes,
                    edges=edges,
                    provenance={"seed_id": seed.id, "generator": "HiARICLGenerator"},
                )
            )

        return trees
