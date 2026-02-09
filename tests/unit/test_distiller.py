from __future__ import annotations

from graph_bot.adapters.distiller import LLMDistiller
from graph_bot.datatypes import ReasoningNode, ReasoningTree


class _StubChatClient:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def chat(self, system: str, user: str, temperature: float):
        del system, user, temperature
        return self._response_text, None


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
