from __future__ import annotations

from graph_bot.adapters.mock_client import MockLLMClient


def test_wordsorting_mock_sorting():
    client = MockLLMClient(model="mock")
    system = "system prompt"
    user = "Input: Sort these words alphabetically: pear apple banana\nOutput:"
    text, _ = client.chat(system=system, user=user, temperature=0.0)
    assert text.strip() == "apple banana pear"


def test_mgsm_mock_numeric_from_problem_without_gold():
    client = MockLLMClient(model="mock")
    system = "system prompt"
    # MGSM deterministic path: include a Problem: line with numbers but no gold_answer
    user = "Problem: 3 + 5 = ?"
    text, _ = client.chat(system=system, user=user, temperature=0.0)
    assert text.strip() == "Answer: 8"
