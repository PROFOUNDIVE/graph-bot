from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from .base import TaskSpec
from .game24 import Game24Task


@dataclass
class TaskRegistry:
    _tasks: dict[str, TaskSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._tasks:
            game24 = Game24Task()
            mgsm_module = importlib.import_module("graph_bot.tasks.mgsm")
            mgsm_task_class = getattr(mgsm_module, "MGSMTask")
            mgsm = mgsm_task_class()
            self._tasks[game24.name] = game24
            self._tasks[mgsm.name] = mgsm
            wordsorting_module = importlib.import_module("graph_bot.tasks.wordsorting")
            wordsorting_task_class = getattr(wordsorting_module, "WordSortingTask")
            wordsorting = wordsorting_task_class()
            self._tasks[wordsorting.name] = wordsorting

    def get_task(self, name: str) -> TaskSpec:
        task = self._tasks.get(name.lower())
        if task is None:
            available = ", ".join(sorted(self._tasks.keys()))
            raise ValueError(f"Unknown task '{name}'. Available tasks: {available}")
        return task


registry = TaskRegistry()
