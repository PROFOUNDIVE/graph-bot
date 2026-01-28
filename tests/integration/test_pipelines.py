from __future__ import annotations

import json

from graph_bot.pipelines.stream_loop import load_game24_problems, run_continual_stream
from graph_bot.settings import settings


def test_load_game24_problems(tmp_path):
    problems_file = tmp_path / "problems.jsonl"
    problems = [
        {"id": "q1", "numbers": [2, 5, 8, 11]},
        {"id": "q2", "numbers": [1, 3, 4, 6], "target": 24},
    ]
    problems_file.write_text(
        "\n".join(json.dumps(problem) for problem in problems) + "\n",
        encoding="utf-8",
    )

    loaded = load_game24_problems(problems_file)

    assert len(loaded) == 2
    assert loaded[0].id == "q1"
    assert loaded[0].target == 24.0

    query = loaded[0].to_user_query()
    assert query.id == "q1"
    assert query.metadata is not None
    assert query.metadata["task"] == "game24"


def test_run_continual_stream_baseline_io(tmp_path, monkeypatch):
    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(
        json.dumps({"id": "q1", "numbers": [2, 4, 6, 8], "target": 24}) + "\n",
        encoding="utf-8",
    )

    metrics_dir = tmp_path / "metrics"

    monkeypatch.setattr(settings, "metagraph_path", tmp_path / "metagraph.json")
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "retry_max_attempts", 1)

    results = run_continual_stream(
        problems_file=problems_file,
        mode="io",
        max_problems=1,
        metrics_out_dir=metrics_dir,
        run_id="test-io",
    )

    assert len(results) == 1
    assert results[0]["problem_id"] == "q1"
    assert results[0]["solved"] is True

    assert (metrics_dir / "test-io.calls.jsonl").exists()
    assert (metrics_dir / "test-io.problems.jsonl").exists()
    assert (metrics_dir / "test-io.stream.jsonl").exists()
    assert (metrics_dir / "test-io.token_events.jsonl").exists()


def test_run_continual_stream_graph_bot_mode(tmp_path, monkeypatch):
    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(
        json.dumps({"id": "q1", "numbers": [2, 4, 6, 8], "target": 24}) + "\n",
        encoding="utf-8",
    )

    metrics_dir = tmp_path / "metrics"
    metagraph_path = tmp_path / "metagraph.json"

    monkeypatch.setattr(settings, "metagraph_path", metagraph_path)
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "retry_max_attempts", 1)

    run_continual_stream(
        problems_file=problems_file,
        mode="graph_bot",
        max_problems=1,
        metrics_out_dir=metrics_dir,
        run_id="run1",
    )

    results = run_continual_stream(
        problems_file=problems_file,
        mode="graph_bot",
        max_problems=1,
        metrics_out_dir=metrics_dir,
        run_id="run2",
    )

    assert len(results) == 1
    reuse_count = results[0]["reuse_count"]
    assert isinstance(reuse_count, int)
    assert reuse_count >= 1
    assert (metrics_dir / "run2.stream.jsonl").exists()
