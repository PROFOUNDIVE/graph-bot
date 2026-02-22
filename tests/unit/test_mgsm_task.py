from __future__ import annotations

import json

import pytest

from graph_bot.datatypes import RetrievalResult, UserQuery
from graph_bot.tasks import registry


def test_registry_exposes_mgsm_task() -> None:
    task = registry.get_task("mgsm")
    assert task.name == "mgsm"


def test_mgsm_loader_maps_question_and_metadata(tmp_path) -> None:
    problems_file = tmp_path / "mgsm.jsonl"
    problems_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "mgsm-ko-1",
                        "language": "ko",
                        "question": "A has 40 apples and gives away 3. How many remain?",
                        "answer": "37",
                    }
                ),
                json.dumps(
                    {
                        "question": "B has 1,234 marbles and loses 34. How many remain?",
                        "answer": "1200",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    task = registry.get_task("mgsm")
    problems = list(task.load_problems(problems_file))

    assert len(problems) == 2

    query0 = task.to_user_query(problems[0])
    assert query0.id == "mgsm-ko-1"
    assert query0.question == "A has 40 apples and gives away 3. How many remain?"
    assert query0.metadata is not None
    assert query0.metadata["task"] == "mgsm"
    assert query0.metadata["gold_answer"] == "37"
    assert query0.metadata["language"] == "ko"

    query1 = task.to_user_query(problems[1])
    assert query1.id == "mgsm-2"
    assert query1.question == "B has 1,234 marbles and loses 34. How many remain?"
    assert query1.metadata is not None
    assert query1.metadata["task"] == "mgsm"
    assert query1.metadata["gold_answer"] == "1200"
    assert query1.metadata.get("language") is None


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("The answer is 42", "42"),
        ("42.", "42"),
        ("1,234", "1234"),
    ],
)
def test_mgsm_extract_candidate_normalizes_numeric_output(
    raw_output: str, expected: str
) -> None:
    task = registry.get_task("mgsm")
    candidate = task.extract_candidate(
        raw_output,
        query=UserQuery(id="q", question="dummy", metadata={"task": "mgsm"}),
    )
    assert candidate == expected


def test_mgsm_extract_candidate_uses_final_numeric_answer() -> None:
    task = registry.get_task("mgsm")
    query = UserQuery(id="q", question="dummy", metadata={"task": "mgsm"})
    raw_output = "Step 1 gives 30. Step 2 gives 41. Therefore, the answer is 42."

    assert task.extract_candidate(raw_output, query=query) == "42"


def test_mgsm_oracle_validate_strict_numeric_equality_after_normalization() -> None:
    task = registry.get_task("mgsm")
    problem = type(
        "P", (), {"id": "p", "question": "q", "answer": "1,234.00", "language": "en"}
    )
    query = task.to_user_query(problem)

    assert task.oracle_validate("1,234", query, problem) == (True, None)
    assert task.oracle_validate("1234.00", query, problem) == (True, None)
    assert task.oracle_validate("1234.01", query, problem) == (
        False,
        "numeric_mismatch",
    )


def test_mgsm_extract_candidate_prefers_answer_block_over_code_output() -> None:
    task = registry.get_task("mgsm")
    query = UserQuery(id="q", question="dummy", metadata={"task": "mgsm"})
    raw_output = (
        "```python\n"
        "# wrong intermediate output\n"
        "print(41)\n"
        "```\n"
        "<answer>42</answer>\n"
    )

    assert task.extract_candidate(raw_output, query=query) == "42"


def test_mgsm_graph_bot_exec_prompt_contract() -> None:
    task = registry.get_task("mgsm")
    query = UserQuery(id="q", question="2 + 2 = ?", metadata={"task": "mgsm"})
    retrieval = RetrievalResult(query_id="q", paths=[], concatenated_context="ctx")

    system, user = task.build_solver_prompt("graph_bot_exec", query, retrieval)

    assert "```python" in system
    assert "<answer>" in system and "</answer>" in system
    assert "no imports" in system.lower()
    assert "file" in system.lower() and "network" in system.lower()
    assert "print only" in system.lower()
    assert "Retrieved templates/context:" in user


def test_mgsm_weak_judge_rubric_draft_exists() -> None:
    task = registry.get_task("mgsm")
    rubric = getattr(task, "weak_judge_rubric_draft", "")
    assert isinstance(rubric, str)
    assert rubric.strip()
    assert "DRAFT" in rubric
