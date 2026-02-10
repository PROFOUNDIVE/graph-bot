from __future__ import annotations

from graph_bot.adapters import distiller as distiller_mod
from graph_bot.adapters.distiller import (
    LLMDistiller,
    NullDistiller,
    RuleBasedDistiller,
    get_distiller,
)
from graph_bot.datatypes import ReasoningNode, ReasoningTree


class _StubChatClient:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def chat(self, system: str, user: str, temperature: float):
        del system, user, temperature
        return self._response_text, None


def test_get_distiller_modes_and_unknown_mode():
    assert isinstance(get_distiller("rulebased"), RuleBasedDistiller)
    assert isinstance(get_distiller("llm"), LLMDistiller)
    assert isinstance(get_distiller("none"), NullDistiller)

    try:
        get_distiller("does-not-exist")
    except ValueError as exc:
        assert "Unknown distiller mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown distiller mode")


def test_null_distiller_is_noop():
    distiller = NullDistiller()

    assert distiller.distill_query("  keep  as-is  ") == "  keep  as-is  "
    tree = ReasoningTree(tree_id="t", root_id="r", nodes=[], edges=[])
    assert distiller.distill_trace(tree) == []


def test_rulebased_distiller_query_distillation_sorts_first_four_numbers():
    distiller = RuleBasedDistiller()

    # Pulls the first four numbers and sorts them.
    assert distiller.distill_query("8 2 6 4 -> 24") == "Solve 24 with 2 4 6 8"

    # Too few numbers => unchanged.
    assert distiller.distill_query("2 4 6") == "2 4 6"

    # No numbers => unchanged.
    assert distiller.distill_query("no numbers") == "no numbers"


def test_llm_distiller_cold_start_guard_skips_llm_and_uses_rulebased():
    distiller = LLMDistiller(model="mock-model")

    # If the cold-start guard triggers, the LLM path shouldn't run.
    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError("LLM path should be skipped for cold-start queries")

    setattr(distiller, "_chat", _should_not_be_called)

    assert distiller.distill_query("2 4 6 8 -> 24") == "Solve 24 with 2 4 6 8"


def test_llm_distiller_chat_failure_falls_back_to_rulebased(monkeypatch):
    distiller = LLMDistiller(model="mock-model")

    def _raise_on_build():
        raise RuntimeError("boom")

    monkeypatch.setattr(distiller, "_build_client", _raise_on_build)

    # This input should NOT be treated as a cold-start query (contains alpha + longer text),
    # so the LLM path is attempted, then falls back when _chat() returns None.
    query = "Please solve 2 4 6 8 for the 24 game using a reusable strategy"
    assert distiller.distill_query(query) == "Solve 24 with 2 4 6 8"


def test_sanitize_llm_output_strips_code_fences_and_known_prefixes():
    raw = "```\nNormalized Query:   hello   world  \n```"
    assert distiller_mod._sanitize_llm_output(raw) == "hello   world"


def test_llm_distiller_query_normalization(monkeypatch):
    distiller = LLMDistiller(model="mock-model")
    monkeypatch.setattr(
        distiller,
        "_build_client",
        lambda: _StubChatClient(
            "Normalized Query:   solve with reusable arithmetic steps   "
        ),
    )

    distilled = distiller.distill_query(
        "Need help solving this puzzle with reusable strategy"
    )

    assert distilled == "solve with reusable arithmetic steps"


def test_llm_distiller_trace_extraction(monkeypatch):
    distiller = LLMDistiller(model="mock-model")
    monkeypatch.setattr(
        distiller,
        "_build_client",
        lambda: _StubChatClient("Template: isolate denominator, then multiply"),
    )

    tree = ReasoningTree(
        tree_id="t1",
        root_id="root",
        nodes=[
            ReasoningNode(
                node_id="root",
                text="(8 * 6) / (4 - 2)",
                type="answer",
                attributes=None,
            )
        ],
        edges=[],
        provenance={
            "query": "2 4 6 8 -> 24",
            "solved": True,
            "task": "game24",
        },
    )

    distilled_nodes = distiller.distill_trace(tree)

    assert len(distilled_nodes) == 1
    node = distilled_nodes[0]
    assert node.node_id == "root"
    assert node.text == "isolate denominator, then multiply"
    assert node.attributes is not None
    assert node.attributes["subtype"] == "template"
    assert node.attributes["original_answer"] == "(8 * 6) / (4 - 2)"
    assert node.attributes["quality"]["validator_passed"] is True
    assert node.attributes["quality"]["parse_ok"] is True


def test_llm_distiller_empty_tree(monkeypatch):
    distiller = LLMDistiller(model="mock-model")
    monkeypatch.setattr(
        distiller,
        "_build_client",
        lambda: _StubChatClient("Template: should never be used"),
    )

    empty_tree = ReasoningTree(tree_id="empty", root_id="", nodes=[], edges=[])

    assert distiller.distill_trace(empty_tree) == []
