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


def test_run_continual_stream_writes_metrics(tmp_path, monkeypatch):
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
        run_id="test",
    )

    assert len(results) == 1
    assert results[0]["problem_id"] == "q1"
    assert results[0]["solved"] is False

    assert (metrics_dir / "test.calls.jsonl").exists()
    assert (metrics_dir / "test.problems.jsonl").exists()
    assert (metrics_dir / "test.stream.jsonl").exists()
    assert (metrics_dir / "test.token_events.jsonl").exists()
