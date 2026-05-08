from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch


from graph_bot.pipelines.stream_loop import run_continual_stream
from graph_bot.settings import settings


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [cast(dict[str, Any], json.loads(line)) for line in f if line.strip()]


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


def test_timeout_problem_row_emits_g1_fairness_metadata() -> None:
    run_id = "test_timeout_g1_metadata"
    problems_dir = Path("outputs/test_timeout_g1_metadata")
    problems_dir.mkdir(parents=True, exist_ok=True)
    problems_file = problems_dir / "test_timeout_g1_metadata_problems.jsonl"
    problems_file.write_text(
        json.dumps({"id": "test_q_timeout_g1", "numbers": [1, 2, 3, 4], "target": 24})
        + "\n",
        encoding="utf-8",
    )

    old_timeout = settings.execution_timeout_sec
    old_provider = settings.llm_provider
    old_model = settings.llm_model
    old_retrieval_backend = settings.retrieval_backend
    old_distiller_mode = settings.distiller_mode
    try:
        settings.execution_timeout_sec = 0.5
        settings.llm_provider = "mock"
        settings.llm_model = "gpt-4o-mini"
        settings.retrieval_backend = "sparse_jaccard"
        settings.distiller_mode = "rulebased"

        metrics_dir = problems_dir / "metrics"
        if metrics_dir.exists():
            shutil.rmtree(metrics_dir)
        metrics_dir.mkdir(parents=True)

        def slow_chat(*args: object, **kwargs: object):
            del args, kwargs
            from graph_bot.adapters.mock_client import LLMUsage

            time.sleep(1.0)
            return "1+2+3+4", LLMUsage(10, 10, 20, 1000.0)

        with patch(
            "graph_bot.adapters.mock_client.MockLLMClient.chat", side_effect=slow_chat
        ):
            run_continual_stream(
                problems_file=problems_file,
                metrics_out_dir=metrics_dir,
                run_id=run_id,
                max_problems=1,
                mode="graph_bot",
                use_edges=True,
                distiller_mode="rulebased",
            )

        problem_rows = _read_jsonl(metrics_dir / f"{run_id}.problems.jsonl")
        timeout_row = next(
            row for row in problem_rows if row["problem_id"] == "test_q_timeout_g1"
        )

        expected_metadata = {
            "run_id": run_id,
            "mode": "graph_bot",
            "seed": None,
            "resample_id": None,
            "provider": "mock",
            "model": "gpt-4o-mini",
            "validator_mode": "oracle",
            "distiller": "rulebased",
            "retrieval_backend": "sparse_jaccard",
            "cost_scope": "llm_api_cost_usd_problem_row",
            "uses_graph_edges": True,
            "uses_persistent_memory": True,
        }
        for key, expected in expected_metadata.items():
            assert key in timeout_row
            assert timeout_row[key] == expected
    finally:
        settings.execution_timeout_sec = old_timeout
        settings.llm_provider = old_provider
        settings.llm_model = old_model
        settings.retrieval_backend = old_retrieval_backend
        settings.distiller_mode = old_distiller_mode
        if problems_dir.exists():
            shutil.rmtree(problems_dir)
