from __future__ import annotations

from typing import Iterable, List

from ..core.models import SeedData, ReasoningTree
from ..core.hiar_icl import HiARICLGenerator


def build_reasoning_trees_from_seeds(seeds: Iterable[SeedData]) -> List[ReasoningTree]:
    generator = HiARICLGenerator()
    return generator.generate(seeds)
