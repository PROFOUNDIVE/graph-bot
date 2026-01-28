from __future__ import annotations

import json

from typer.testing import CliRunner

from graph_bot.cli import app
from graph_bot.settings import settings


def test_trees_insert_cli(tmp_path, mock_settings):
    runner = CliRunner()

    trees_file = tmp_path / "trees.json"
    trees_file.write_text(
        json.dumps(
            [
                {
                    "tree_id": "tree1",
                    "root_id": "root",
                    "nodes": [
                        {"node_id": "root", "text": "root", "type": "thought"},
                        {"node_id": "child", "text": "child", "type": "answer"},
                    ],
                    "edges": [{"src": "root", "dst": "child"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["trees-insert", str(trees_file)])
    assert result.exit_code == 0
    assert "Inserted 1 trees" in result.stdout
    assert mock_settings.metagraph_path.exists()


def test_stream_and_amortize_cli(tmp_path, mock_settings, monkeypatch):
    runner = CliRunner()

    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(
        json.dumps({"id": "q1", "numbers": [2, 4, 6, 8], "target": 24}) + "\n",
        encoding="utf-8",
    )

    metrics_dir = tmp_path / "metrics"
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "retry_max_attempts", 1)

    result = runner.invoke(
        app,
        [
            "stream",
            str(problems_file),
            "--mode",
            "io",
            "--max-problems",
            "1",
            "--metrics-out-dir",
            str(metrics_dir),
            "--run-id",
            "test-run",
        ],
    )
    assert result.exit_code == 0

    stream_path = metrics_dir / "test-run.stream.jsonl"
    assert stream_path.exists()

    out_csv = tmp_path / "amortization.csv"
    result = runner.invoke(app, ["amortize", str(stream_path), "--out", str(out_csv)])
    assert result.exit_code == 0
    assert out_csv.exists()
