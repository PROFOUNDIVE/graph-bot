from __future__ import annotations

from graph_bot.adapters.mock_client import MockLLMClient
from graph_bot.baselines.got.game24_search import run_search_variant
from graph_bot.eval.validators import Game24Validator
from graph_bot.datatypes import ReasoningNode


def test_got_search_unit_deterministic_and_validator() -> None:
    client = MockLLMClient(model="mock-model")
    result = run_search_variant(
        variant="got", numbers=[2, 4, 6, 8], client=client, temperature=0.0
    )

    assert result.best_terminal_state is not None
    # Reimplement test_game24 logic locally to avoid PyTest collection conflicts
    items = result.best_terminal_state.get("items", [])
    assert isinstance(items, list) and len(items) == 1
    assert float(items[0].get("value", 0.0)) == 24.0

    validator = Game24Validator()
    node = ReasoningNode(
        node_id="testnode",
        text="8 * (6 / (4 - 2))",
        type="answer",
        attributes={"problem": "2 4 6 8 -> 24", "llm_client": client},
    )
    assert validator.validate(node) == 1.0
