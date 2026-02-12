from __future__ import annotations

from typing import Any, cast

from graph_bot.adapters import distiller as distiller_mod
from graph_bot.adapters.distiller import (
    LLMDistiller,
    NullDistiller,
    RuleBasedDistiller,
    get_distiller,
)
from graph_bot.datatypes import ReasoningNode, ReasoningTree, RetrievalResult, UserQuery
from graph_bot.pipelines import stream_loop


class _StubChatClient:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def chat(self, system: str, user: str, temperature: float):
        del system, user, temperature
        return self._response_text, None


class _RecordingAdapter:
    def __init__(self) -> None:
        self.inserted_trees = []

    def insert_trees(self, trees):
        self.inserted_trees.extend(trees)


class _RecordingMetrics:
    def __init__(self) -> None:
        self.run_id = "test_run"
        self._counter = 0

    def new_call_id(self):
        self._counter += 1
        return f"call-{self._counter}"

    def log_token_event(self, event):
        del event


class _RecordingDistiller:
    def __init__(self) -> None:
        self.last_tree = None

    def distill_query(self, query: str) -> str:
        return query

    def distill_trace(self, tree: ReasoningTree):
        self.last_tree = tree
        return [
            ReasoningNode(
                node_id="template-node",
                text="Task: wordsorting\nThought Template:\n- deterministic",
                type="thought",
                attributes={"subtype": "template", "task": "wordsorting"},
            )
        ]


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

    def _should_be_called(*_args, **_kwargs):
        return (
            "Key information:\n"
            "- numbers: 2 4 6 8\n\n"
            "Restrictions:\n"
            "- Objective: solve the 24 game\n"
            "- Constraints: use each number exactly once\n\n"
            "Distilled task:\n"
            "- Real-world scenario: arithmetic puzzle normalization\n"
            "- Variables: numbers (List[int])\n"
            "- Task statement: derive a correct expression that reaches target\n"
            "- Example: numbers=2 4 6 8"
        )

    setattr(distiller, "_chat", _should_be_called)

    distilled = distiller.distill_query("2 4 6 8 -> 24")
    assert distilled.startswith("Key information:")
    assert "Restrictions:" in distilled
    assert "Distilled task:" in distilled


def test_llm_distiller_chat_failure_falls_back_to_rulebased(monkeypatch):
    distiller = LLMDistiller(model="mock-model")

    def _raise_on_build():
        raise RuntimeError("boom")

    monkeypatch.setattr(distiller, "_build_client", _raise_on_build)

    query = "Please solve 2 4 6 8 for the 24 game using a reusable strategy"
    assert distiller.distill_query(query) == query


def test_sanitize_llm_output_strips_code_fences_and_known_prefixes():
    raw = "```\nNormalized Query:   hello   world  \n```"
    assert distiller_mod._sanitize_llm_output(raw) == "hello   world"


def test_llm_distiller_query_normalization(monkeypatch):
    distiller = LLMDistiller(model="mock-model")
    monkeypatch.setattr(
        distiller,
        "_build_client",
        lambda: _StubChatClient(
            "Key information:\n"
            "- intent: solve with reusable arithmetic steps\n\n"
            "Restrictions:\n"
            "- Objective: normalize the problem\n\n"
            "Distilled task:\n"
            "- Real-world scenario: reusable reasoning across similar inputs\n"
            "- Variables: intent (str)\n"
            "- Task statement: distill the problem\n"
            "- Example: intent='solve with reusable arithmetic steps'"
        ),
    )

    distilled = distiller.distill_query(
        "Need help solving this puzzle with reusable strategy"
    )

    assert distilled.startswith("Key information:")
    assert "intent: solve with reusable arithmetic steps" in distilled


def test_llm_distiller_trace_extraction(monkeypatch):
    distiller = LLMDistiller(model="mock-model")
    monkeypatch.setattr(
        distiller,
        "_build_client",
        lambda: _StubChatClient(
            "Core task summarization: arithmetic puzzle template\n"
            "Solution Steps Description: outline reusable steps\n"
            "General Answer Template: provide a reusable template"
        ),
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
    assert node.text.startswith("Task: game24")
    assert "Core task summarization" in node.text
    assert node.attributes is not None
    assert node.attributes["subtype"] == "template"
    assert node.attributes["original_answer"] == "(8 * 6) / (4 - 2)"
    assert node.attributes["steps_summary"] == ""
    assert node.attributes["final_candidate"] == "(8 * 6) / (4 - 2)"
    assert node.attributes["quality"]["validator_passed"] is True
    assert node.attributes["quality"]["parse_ok"] is True


def test_llm_distiller_trace_uses_bot_style_input_and_forces_task_prefix(monkeypatch):
    distiller = LLMDistiller(model="mock-model")
    captured = {"system": "", "user": ""}

    def _capture_chat(*, system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return "Thought Template:\n1) sort tokens\n2) output one line"

    monkeypatch.setattr(distiller, "_chat", _capture_chat)

    tree = ReasoningTree(
        tree_id="t2",
        root_id="root",
        nodes=[
            ReasoningNode(
                node_id="root",
                text="pear apple",
                type="answer",
                attributes=None,
            )
        ],
        edges=[],
        provenance={
            "query": "Sort these words: pear apple",
            "solved": True,
            "task": "wordsorting",
            "steps_summary": "Identify words, sort alphabetically, emit one line.",
            "final_candidate": "apple pear",
            "distill_input": (
                "Task: wordsorting\n"
                "Problem: Sort these words: pear apple\n"
                "Solution Steps Summary: Identify words, sort alphabetically, emit one line.\n"
                "Final Candidate: apple pear"
            ),
        },
    )

    distilled_nodes = distiller.distill_trace(tree)

    assert "Prompt for Template Distillation" in captured["system"]
    assert "Core task summarization" in captured["system"]
    assert "[Problem Description]" in captured["user"]
    assert "[Solution Steps or Code]" in captured["user"]
    assert "apple pear" in captured["user"]
    assert len(distilled_nodes) == 1
    assert distilled_nodes[0].text.startswith("Task: wordsorting")


def test_llm_distiller_trace_rewrites_wrong_task_prefix(monkeypatch):
    distiller = LLMDistiller(model="mock-model")
    monkeypatch.setattr(
        distiller,
        "_chat",
        lambda **_kwargs: "Task: mgsm\nThought Template:\n1) do thing",
    )

    tree = ReasoningTree(
        tree_id="t3",
        root_id="root",
        nodes=[ReasoningNode(node_id="root", text="x", type="answer", attributes=None)],
        edges=[],
        provenance={"query": "2 4 6 8 -> 24", "solved": True, "task": "game24"},
    )

    distilled_nodes = distiller.distill_trace(tree)
    assert distilled_nodes[0].text.startswith("Task: game24")


def test_insert_solution_template_uses_real_task_and_single_template_node():
    adapter = _RecordingAdapter()
    distiller = _RecordingDistiller()
    metrics = _RecordingMetrics()
    query = UserQuery(
        id="q-1",
        question="Sort these words: pear apple",
        metadata={"task": "wordsorting", "target": "apple pear"},
    )

    stream_loop._insert_solution_template(
        adapter=cast(Any, adapter),
        distiller=cast(Any, distiller),
        metrics=cast(Any, metrics),
        t=1,
        problem_id="p1",
        answer_text="apple pear",
        solved=True,
        query=query,
        retrieval=RetrievalResult(query_id="q-1", paths=[], concatenated_context=""),
    )

    assert distiller.last_tree is not None
    assert distiller.last_tree.provenance is not None
    assert distiller.last_tree.provenance["task"] == "wordsorting"
    assert "steps_summary" in distiller.last_tree.provenance
    assert "distill_input" in distiller.last_tree.provenance

    assert len(adapter.inserted_trees) == 1
    inserted_tree = adapter.inserted_trees[0]
    assert inserted_tree.provenance is not None
    assert inserted_tree.provenance["task"] == "wordsorting"
    assert len(inserted_tree.nodes) == 1
    assert inserted_tree.nodes[0].type == "thought"
    assert inserted_tree.nodes[0].attributes["subtype"] == "template"


def test_llm_distiller_empty_tree(monkeypatch):
    distiller = LLMDistiller(model="mock-model")
    monkeypatch.setattr(
        distiller,
        "_build_client",
        lambda: _StubChatClient("Template: should never be used"),
    )

    empty_tree = ReasoningTree(tree_id="empty", root_id="", nodes=[], edges=[])

    assert distiller.distill_trace(empty_tree) == []
