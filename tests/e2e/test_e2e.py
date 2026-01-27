from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from graph_bot.adapters.vllm_openai_client import LLMUsage
from graph_bot.cli import app


@pytest.fixture
def mock_llm_client():
    with patch("graph_bot.pipelines.main_loop.VLLMOpenAIClient") as mock:
        instance = mock.return_value
        instance.chat.return_value = (
            "(2 + 4) * (6 - 2)",
            LLMUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                latency_ms=100.0,
            ),
        )
        yield instance


def test_full_pipeline_flow(tmp_path, mock_settings, mock_llm_client):
    runner = CliRunner()

    # 1. Create a mock seeds file
    seeds_file = tmp_path / "seeds.jsonl"
    with seeds_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "seed1", "content": "2 4 6 8"}) + "\n")

    trees_file = tmp_path / "trees.json"

    # 2. graph-bot seeds-build
    # Command: seeds-build SEEDS_FILE [--out OUT_FILE]
    result = runner.invoke(
        app, ["seeds-build", str(seeds_file), "--out", str(trees_file)]
    )
    assert result.exit_code == 0
    assert trees_file.exists()
    assert "Wrote 1 trees" in result.stdout

    # 3. graph-bot trees-insert
    # Command: trees-insert TREES_FILE
    result = runner.invoke(app, ["trees-insert", str(trees_file)])
    assert result.exit_code == 0
    assert "Inserted 1 trees" in result.stdout

    # 4. Verify metagraph.json exists in the mocked path
    # mock_settings.metagraph_path is set in conftest.py
    assert mock_settings.metagraph_path.exists()

    # 5. graph-bot postprocess
    result = runner.invoke(app, ["postprocess", "--t", "0"])
    assert result.exit_code == 0
    assert "Postprocess pruned" in result.stdout

    # 6. graph-bot retrieve
    # Command: retrieve QUERY [--show-paths]
    result = runner.invoke(app, ["retrieve", "2 4 6 8", "--show-paths"])
    assert result.exit_code == 0

    # Verify output contains the answer from our mock_llm_client
    assert "(2 + 4) * (6 - 2)" in result.stdout
    # Verify paths were shown
    assert "query_id" in result.stdout
    assert "paths" in result.stdout

    # 7. graph-bot loop-once
    result = runner.invoke(app, ["loop-once", "1 2 3 4"])
    assert result.exit_code == 0
    assert "(2 + 4) * (6 - 2)" in result.stdout
    assert "Generated and inserted" in result.stdout
