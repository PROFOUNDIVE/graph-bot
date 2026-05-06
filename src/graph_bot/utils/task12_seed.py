from __future__ import annotations

from pathlib import Path
from typing import List

from ..datatypes import MetaGraph, ReasoningEdge, ReasoningNode


def build_task12_seeded_metagraph(numbers: List[int] | None = None) -> MetaGraph:
    seed_numbers = list(numbers or [1, 2, 3, 4])
    numbers_text = " ".join(str(value) for value in sorted(seed_numbers))

    seed_node = ReasoningNode(
        node_id="seed",
        text=f"{numbers_text} 24 retrieval root",
        type="thought",
        attributes={"task": "game24", "subtype": "template"},
    )
    child_node = ReasoningNode(
        node_id="child",
        text="followup child",
        type="thought",
        attributes={"task": "game24", "subtype": "template"},
    )
    edge = ReasoningEdge(
        src="seed",
        dst="child",
        relation="used_for",
        attributes={"task": "game24"},
    )

    return MetaGraph(
        graph_id="task12-seeded-gate0",
        nodes=[seed_node, child_node],
        edges=[edge],
        metadata={
            "purpose": "task12_seeded_gate0_validation",
            "task": "game24",
            "seed_numbers": sorted(seed_numbers),
        },
    )


def write_task12_seeded_metagraph(
    path: Path, numbers: List[int] | None = None
) -> MetaGraph:
    graph = build_task12_seeded_metagraph(numbers=numbers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    return graph
