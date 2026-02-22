from __future__ import annotations

import json

from typer.testing import CliRunner

import graph_bot.adapters.mock_client as mock_client_mod
import graph_bot.cli as cli_mod
import graph_bot.pipelines.stream_loop as stream_loop
from graph_bot.cli import app
from graph_bot.datatypes import MetaGraph, ReasoningNode, UserQuery
from graph_bot.settings import settings
from graph_bot.tasks import registry as task_registry


def _write_game24_problem(tmp_path):
    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(
        json.dumps({"id": "q1", "numbers": [2, 4, 6, 8], "target": 24}) + "\n",
        encoding="utf-8",
    )
    return problems_file


def test_stream_cli_forwards_distiller_mode_and_validator_model(tmp_path, monkeypatch):
    runner = CliRunner()

    problems_file = _write_game24_problem(tmp_path)

    captured: dict[str, object] = {}

    def _fake_run_continual_stream(**kwargs):
        captured.update(kwargs)
        return []

    # Ensure CLI arg takes precedence over settings.
    monkeypatch.setattr(settings, "validator_model", "settings-model")
    monkeypatch.setattr(cli_mod, "run_continual_stream", _fake_run_continual_stream)

    result = runner.invoke(
        app,
        [
            "stream",
            str(problems_file),
            "--validator-mode",
            "weak_llm_judge",
            "--validator-model",
            "cli-model",
            "--distiller-mode",
            "none",
            "--metrics-out-dir",
            str(tmp_path / "metrics"),
            "--run-id",
            "test_cli_stream",
            "--max-problems",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert captured["validator_mode"] == "weak_llm_judge"
    assert captured["validator_model"] == "cli-model"
    assert captured["distiller_mode"] == "none"


def test_stream_cli_defaults_task_and_cross_task_retrieval(tmp_path, monkeypatch):
    runner = CliRunner()
    problems_file = _write_game24_problem(tmp_path)

    captured: dict[str, object] = {}

    def _fake_run_continual_stream(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli_mod, "run_continual_stream", _fake_run_continual_stream)

    result = runner.invoke(app, ["stream", str(problems_file)])

    assert result.exit_code == 0
    assert captured["task"] == "game24"
    assert captured["cross_task_retrieval"] is False


def test_stream_cli_forwards_task_and_cross_task_retrieval(tmp_path, monkeypatch):
    runner = CliRunner()
    problems_file = _write_game24_problem(tmp_path)

    captured: dict[str, object] = {}

    def _fake_run_continual_stream(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli_mod, "run_continual_stream", _fake_run_continual_stream)

    result = runner.invoke(
        app,
        [
            "stream",
            str(problems_file),
            "--task",
            "wordsorting",
            "--cross-task-retrieval",
        ],
    )

    assert result.exit_code == 0
    assert captured["task"] == "wordsorting"
    assert captured["cross_task_retrieval"] is True


def test_stream_cli_help_mentions_domain_jsonl_formats(tmp_path):
    del tmp_path
    runner = CliRunner()

    result = runner.invoke(app, ["stream", "--help"])

    assert result.exit_code == 0
    assert "game24" in result.stdout
    assert "wordsorting" in result.stdout
    assert "mgsm" in result.stdout


def test_run_continual_stream_uses_selected_task_and_metadata(tmp_path, monkeypatch):
    class _TaskStub:
        name = "wordsorting"

        def __init__(self) -> None:
            self.loaded_path = None
            self.prompt_called = False
            self.extract_called = False
            self.oracle_called = False

        def load_problems(self, jsonl_path):
            self.loaded_path = jsonl_path
            return [{"id": "p1", "input": "pear apple", "target": "apple pear"}]

        def to_user_query(self, problem):
            return UserQuery(
                id=str(problem["id"]),
                question=str(problem["input"]),
                metadata={"target": str(problem["target"])},
            )

        def build_solver_prompt(self, mode, query, retrieval):
            del mode
            del query
            del retrieval
            self.prompt_called = True
            return "system", "Input: pear apple\nOutput:"

        def extract_candidate(self, raw_output, query):
            del query
            self.extract_called = True
            return raw_output.strip()

        def oracle_validate(self, candidate, query, problem):
            del query
            del problem
            self.oracle_called = True
            if candidate == "apple pear":
                return True, None
            return False, "mismatch"

        def summarize_steps(self, raw_output, candidate, query):
            del raw_output
            return {
                "task": self.name,
                "query": query.question,
                "candidate": candidate,
                "summary": "Sort words and output one line.",
            }

        def distill_template_input(self, query, steps_summary):
            return (
                f"Task: {self.name}\n"
                f"Problem: {query.question}\n"
                f"Solution Steps Summary: {steps_summary['summary']}\n"
                f"Final Candidate: {steps_summary['candidate']}"
            )

    class _ValidatorStub:
        def get_validator_name(self):
            return "oracle"

        def validate(self, node, problem=None):
            del node
            del problem
            return True

        def failure_reason(self, answer, problem):
            del answer
            del problem
            return None

    class _DistillerStub:
        def distill_query(self, query):
            return query

        def distill_trace(self, tree):
            del tree
            return [
                ReasoningNode(
                    node_id="template-1",
                    text="Task: wordsorting\nThought Template:\n1. Sort words alphabetically.",
                    type="thought",
                    attributes={"subtype": "template", "task": "wordsorting"},
                )
            ]

    adapter_box: dict[str, object] = {}

    class _AdapterStub:
        def __init__(
            self,
            *,
            mode=None,
            use_edges=None,
            policy_id=None,
            cross_task_retrieval=False,
            retrieval_backend=None,
            **kwargs,
        ):
            self.mode = mode
            self.policy_id = policy_id
            self.use_edges = use_edges
            self.cross_task_retrieval = cross_task_retrieval
            # Swallow any new keyword that GraphRAGAdapter might pass
            # (e.g., retrieval_backend) to avoid breaking test intent.
            self.retrieval_backend = retrieval_backend
            if kwargs:
                # Keep any extra kwargs for forward-compatibility without failing.
                pass
            self.retrieval_query = None
            self.inserted_trees = []
            adapter_box["instance"] = self

        def retrieve_paths(self, query, k):
            del k
            self.retrieval_query = query
            return stream_loop.RetrievalResult(
                query_id=query.id,
                paths=[],
                concatenated_context="",
            )

        def register_usage(self, paths):
            del paths

        def update_with_feedback(self, evaluations):
            del evaluations

        def insert_trees(self, trees):
            self.inserted_trees.extend(trees)
            return len(trees)

        def export_graph(self):
            return MetaGraph(graph_id="g-test", nodes=[], edges=[], metadata={})

    class _ManifestStub:
        def log_start(self, run_id, config):
            del run_id
            del config

        def log_end(self, run_id, status, metrics):
            del run_id
            del status
            del metrics

    task_stub = _TaskStub()
    monkeypatch.setattr(task_registry, "get_task", lambda _name: task_stub)
    monkeypatch.setattr(stream_loop, "GraphRAGAdapter", _AdapterStub)
    monkeypatch.setattr(
        stream_loop, "get_validator", lambda _mode, _model: _ValidatorStub()
    )
    monkeypatch.setattr(stream_loop, "get_distiller", lambda _mode: _DistillerStub())
    monkeypatch.setattr(stream_loop, "RunManifest", _ManifestStub)
    monkeypatch.setattr(
        stream_loop, "send_slack_notification", lambda _url, _payload: None
    )
    monkeypatch.setattr(
        stream_loop, "load_pricing_table", lambda _path: {"models": {"mock": {}}}
    )
    monkeypatch.setattr(stream_loop, "calculate_cost", lambda **_kwargs: 0.0)
    monkeypatch.setattr(
        mock_client_mod.MockLLMClient,
        "chat",
        lambda self, system, user, temperature: ("apple pear", None),
    )

    monkeypatch.setattr(settings, "metagraph_path", tmp_path / "metagraph.json")
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "retry_max_attempts", 1)

    problems_file = tmp_path / "task_problems.jsonl"
    problems_file.write_text("{}\n", encoding="utf-8")

    results = stream_loop.run_continual_stream(
        problems_file=problems_file,
        task="wordsorting",
        cross_task_retrieval=True,
        mode="graph_bot",
        max_problems=1,
        run_id="test_stream_task",
        metrics_out_dir=tmp_path / "metrics",
    )

    assert len(results) == 1
    assert task_stub.loaded_path == problems_file
    assert task_stub.prompt_called is True
    assert task_stub.extract_called is True
    assert task_stub.oracle_called is True

    adapter = adapter_box["instance"]
    assert isinstance(adapter, _AdapterStub)
    assert adapter.cross_task_retrieval is True
    assert adapter.retrieval_query is not None
    assert adapter.retrieval_query.metadata is not None
    assert adapter.retrieval_query.metadata["task"] == "wordsorting"
    assert len(adapter.inserted_trees) == 1
    assert adapter.inserted_trees[0].provenance["task"] == "wordsorting"
