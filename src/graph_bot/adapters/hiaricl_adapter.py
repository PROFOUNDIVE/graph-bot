from __future__ import annotations

import uuid
from typing import Iterable, List

from ..types import SeedData, ReasoningTree, ReasoningNode, ReasoningEdge
from ..settings import settings


class HiARICLAdapter:
    """Adapter for HiAR-ICL generation.

    This currently provides a local stub generation to keep the pipeline working.
    Replace internals with calls to the upstream HiAR-ICL submodule when wired.
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
                    provenance={"seed_id": seed.id, "adapter": "HiARICLAdapter"},
                )
            )

        return trees
