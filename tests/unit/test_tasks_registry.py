from __future__ import annotations

import json

import pytest

from graph_bot.tasks import registry


def test_registry_returns_game24_task() -> None:
    task = registry.get_task("game24")
    assert task.name == "game24"


def test_registry_unknown_task_error_is_clear() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unknown task 'unknown'\. Available tasks: .*wordsorting",
    ):
        registry.get_task("unknown")


def test_registry_returns_wordsorting_task() -> None:
    task = registry.get_task("wordsorting")
    assert task.name == "wordsorting"


def test_game24_task_load_problems_remains_compatible(tmp_path) -> None:
    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(
        json.dumps({"id": "q1", "numbers": [2, 5, 8, 11], "target": 24}) + "\n",
        encoding="utf-8",
    )

    task = registry.get_task("game24")
    loaded = list(task.load_problems(problems_file))

    assert len(loaded) == 1
    query = task.to_user_query(loaded[0])
    assert query.metadata is not None
    assert query.metadata["task"] == "game24"
