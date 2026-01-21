from __future__ import annotations

from typing import Iterable, List

from ..datatypes import SeedData, ReasoningTree
from ..adapters.hiaricl_adapter import HiARICLAdapter


def build_reasoning_trees_from_seeds(seeds: Iterable[SeedData]) -> List[ReasoningTree]:
    _adapter = HiARICLAdapter()
    return _adapter.generate(seeds)
