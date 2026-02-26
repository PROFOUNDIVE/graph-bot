from __future__ import annotations


from graph_bot.datatypes import RetrievalResult, UserQuery
from graph_bot.tasks import registry


def test_game24_bot_prompt_matches_flat_template_rag_for_fair_comparison() -> None:
    task = registry.get_task("game24")
    query = UserQuery(
        id="q1",
        question="Input: 1 3 4 6",
        metadata={
            "task": "game24",
            "distilled_question": "1 3 4 6",
            "original_question": "Input: 1 3 4 6",
        },
    )
    retrieval = RetrievalResult(query_id="q1", paths=[], concatenated_context="ctx")

    assert task.build_solver_prompt(
        mode="bot",
        query=query,
        retrieval=retrieval,
    ) == task.build_solver_prompt(
        mode="flat_template_rag",
        query=query,
        retrieval=retrieval,
    )
