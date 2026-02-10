from __future__ import annotations

import json

from typer.testing import CliRunner

import graph_bot.cli as cli_mod
from graph_bot.cli import app
from graph_bot.settings import settings


def test_stream_cli_forwards_distiller_mode_and_validator_model(tmp_path, monkeypatch):
    runner = CliRunner()

    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(
        json.dumps({"id": "q1", "numbers": [2, 4, 6, 8], "target": 24}) + "\n",
        encoding="utf-8",
    )

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
