from __future__ import annotations

from graph_bot.datatypes import RetrievalResult, UserQuery
from graph_bot.tasks import registry


def test_game24_graph_bot_exec_prompt_contract() -> None:
    task = registry.get_task("game24")
    query = UserQuery(id="q", question="1 3 4 6 -> 24", metadata={"task": "game24"})
    retrieval = RetrievalResult(query_id="q", paths=[], concatenated_context="ctx")

    system, user = task.build_solver_prompt("graph_bot_exec", query, retrieval)

    assert "```python" in system
    assert "<answer>" in system and "</answer>" in system
    assert "no imports" in system.lower()
    assert "file" in system.lower() and "network" in system.lower()
    assert "print only" in system.lower()
    assert "Retrieved templates/context:" in user


def test_game24_extract_candidate_prefers_answer_block_over_code_output() -> None:
    task = registry.get_task("game24")
    query = UserQuery(id="q", question="4 9 10 13 -> 24", metadata={"task": "game24"})
    raw_output = (
        "```python\n"
        "# wrong intermediate output\n"
        "print('(1 + 3) * (4 + 2)')\n"
        "```\n"
        "<answer>(10 - 4) * (13 - 9)</answer>\n"
    )

    candidate = task.extract_candidate(raw_output, query)

    assert candidate == "(10 - 4) * (13 - 9)"
