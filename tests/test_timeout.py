import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from graph_bot.pipelines.stream_loop import run_continual_stream
from graph_bot.settings import settings


def test_timeout_enforcement():
    problems_dir = Path("outputs/test_timeout")
    problems_dir.mkdir(parents=True, exist_ok=True)
    problems_file = problems_dir / "problems.jsonl"
    with problems_file.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps({"id": "q-timeout", "numbers": [1, 2, 3, 4], "target": 24})
            + "\n"
        )

    old_timeout = settings.execution_timeout_sec
    old_provider = settings.llm_provider
    try:
        settings.execution_timeout_sec = 0.5
        settings.llm_provider = "mock"

        metrics_dir = problems_dir / "metrics"
        if metrics_dir.exists():
            shutil.rmtree(metrics_dir)
        metrics_dir.mkdir(parents=True)

        def slow_chat(*args, **kwargs):
            time.sleep(1.0)
            from graph_bot.adapters.mock_client import LLMUsage

            return "1+2+3+4", LLMUsage(10, 10, 20, 1000.0)

        with patch(
            "graph_bot.adapters.mock_client.MockLLMClient.chat", side_effect=slow_chat
        ):
            results = run_continual_stream(
                problems_file=problems_file,
                metrics_out_dir=metrics_dir,
                run_id="test_timeout",
                max_problems=1,
            )

        print(f"Results: {results}")

        calls_file = metrics_dir / "test_timeout.calls.jsonl"
        problems_file_log = metrics_dir / "test_timeout.problems.jsonl"

        assert calls_file.exists()
        assert problems_file_log.exists()

        found_timeout_call = False
        with calls_file.open("r", encoding="utf-8") as f:
            for line in f:
                call = json.loads(line)
                if (
                    call.get("operation") == "timeout"
                    and call.get("error_type") == "ERR_TIMEOUT"
                ):
                    found_timeout_call = True

        found_timeout_problem = False
        with problems_file_log.open("r", encoding="utf-8") as f:
            for line in f:
                prob = json.loads(line)
                if (
                    prob.get("problem_id") == "q-timeout"
                    and prob.get("solved") is False
                ):
                    found_timeout_problem = True

        assert found_timeout_call, "Timeout call not found in logs"
        assert found_timeout_problem, "Timeout problem not found in logs"
        assert results[0].get("error") == "timeout"
    finally:
        settings.execution_timeout_sec = old_timeout
        settings.llm_provider = old_provider
        if problems_dir.exists():
            shutil.rmtree(problems_dir)
