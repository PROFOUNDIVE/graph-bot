from __future__ import annotations

import json

from graph_bot.datatypes import UserQuery
from graph_bot.tasks import registry


def test_wordsorting_loader_parses_jsonl_and_builds_query_metadata(tmp_path) -> None:
    problems_file = tmp_path / "wordsorting.jsonl"
    problems_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "ws-1",
                        "input": "Sort these words alphabetically: pear apple banana",
                        "target": "apple banana pear",
                    }
                ),
                json.dumps(
                    {
                        "input": "Sort these words alphabetically: orange kiwi",
                        "target": "kiwi orange",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    task = registry.get_task("wordsorting")
    loaded = list(task.load_problems(problems_file))

    assert len(loaded) == 2

    first_query = task.to_user_query(loaded[0])
    assert first_query.id == "ws-1"
    assert first_query.question == "Sort these words alphabetically: pear apple banana"
    assert first_query.metadata is not None
    assert first_query.metadata["task"] == "wordsorting"
    assert first_query.metadata["target"] == "apple banana pear"

    second_query = task.to_user_query(loaded[1])
    assert second_query.id == "wordsorting-2"
    assert second_query.metadata is not None
    assert second_query.metadata["task"] == "wordsorting"
    assert second_query.metadata["target"] == "kiwi orange"


def test_wordsorting_extract_candidate_uses_first_non_empty_line() -> None:
    task = registry.get_task("wordsorting")
    query = UserQuery(id="q1", question="Sort: c b a", metadata={"task": "wordsorting"})
    raw_output = "\n\n   c    b\t\ta    \nfinal answer: a b c\n"

    candidate = task.extract_candidate(raw_output, query)

    assert candidate == "c b a"


def test_wordsorting_oracle_validate_matches_with_whitespace_normalization() -> None:
    task = registry.get_task("wordsorting")
    query = UserQuery(
        id="q1",
        question="Sort: pear apple banana",
        metadata={"task": "wordsorting", "target": "apple banana pear"},
    )

    ok, reason = task.oracle_validate(" apple\tbanana   pear ", query, problem={})
    assert ok is True
    assert reason is None


def test_wordsorting_oracle_validate_rejects_non_matching_order() -> None:
    task = registry.get_task("wordsorting")
    query = UserQuery(
        id="q1",
        question="Sort: pear apple banana",
        metadata={"task": "wordsorting", "target": "apple banana pear"},
    )

    ok, reason = task.oracle_validate("banana apple pear", query, problem={})
    assert ok is False
    assert reason == "mismatch"
