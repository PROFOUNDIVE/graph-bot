from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import jsonschema
import pytest

from graph_bot.pipelines.stream_loop import (
    run_continual_stream,
    _normalize_candidate_line,
)
from graph_bot.settings import settings


def test_normalize_candidate_line():
    assert _normalize_candidate_line("  \n  (1+2+3)*4  \n  ") == (
        "(1+2+3)*4",
        "fallback_bottom_scan",
    )
    assert _normalize_candidate_line("") == ("", "empty")
    assert _normalize_candidate_line("\n\n") == ("", "empty")
    assert _normalize_candidate_line("Answer: 24\n1+2+3+4") == (
        "1+2+3+4",
        "fallback_bottom_scan",
    )
    assert _normalize_candidate_line("Output: (1+2+3)*4 = 24") == (
        "(1+2+3)*4",
        "got_output_format",
    )


def test_normalize_candidate_priority_and_edge_cases():
    # 1. Conflict: <answer>A</answer> vs Output: B -> Expect A.
    raw_1 = """
    Reasoning...
    Output: (1+2+3)*4 = 24
    <answer>(4-1)*8 = 24</answer>
    """
    assert _normalize_candidate_line(raw_1) == ("(4-1)*8", "answer_block")

    # 2. Malformed: <answer>A vs Output: B -> Expect B.
    raw_2 = """
    Reasoning...
    Output: (1+2+3)*4 = 24
    <answer>(4-1)*8 = 24
    """
    assert _normalize_candidate_line(raw_2) == ("(1+2+3)*4", "got_output_format")

    # 3. Multiple: <answer>A</answer> ... <answer>B</answer> -> Expect A (current regex behavior).
    raw_3 = """
    <answer>(1+2+3)*4 = 24</answer>
    <answer>(4-1)*8 = 24</answer>
    """
    assert _normalize_candidate_line(raw_3) == ("(1+2+3)*4", "answer_block")


def test_stream_contract_and_schema():
    test_dir = Path("outputs/test_contract")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    problems_file = test_dir / "problems.jsonl"
    with problems_file.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps({"id": "q-contract", "numbers": [1, 2, 3, 4], "target": 24})
            + "\n"
        )
        f.write(
            json.dumps({"id": "q-timeout", "numbers": [5, 6, 7, 8], "target": 24})
            + "\n"
        )

    schema_path = Path("docs/specs/token_tracker_schema_v0.json")
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    old_timeout = settings.execution_timeout_sec
    old_provider = settings.llm_provider
    old_model = settings.llm_model

    try:
        settings.execution_timeout_sec = 0.5
        settings.llm_provider = "mock"
        settings.llm_model = "gpt-4o-mini"

        metrics_dir = test_dir / "metrics"

        def mocked_chat(system, user, temperature):
            from graph_bot.adapters.mock_client import LLMUsage

            if "5 6 7 8" in user:
                time.sleep(1.0)

            return "(1+2+3)*4", LLMUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                latency_ms=200.0,
            )

        with patch(
            "graph_bot.adapters.mock_client.MockLLMClient.chat", side_effect=mocked_chat
        ):
            run_continual_stream(
                problems_file=problems_file,
                metrics_out_dir=metrics_dir,
                run_id="contract",
                max_problems=2,
            )

        calls_file = metrics_dir / "contract.calls.jsonl"
        token_events_file = metrics_dir / "contract.token_events.jsonl"
        assert calls_file.exists()
        assert token_events_file.exists()

        with calls_file.open("r", encoding="utf-8") as f:
            calls = [json.loads(line) for line in f]

        retrieve_calls = [c for c in calls if c["operation"] == "retrieve"]
        validate_calls = [c for c in calls if c["operation"] == "validate"]
        solve_calls = [c for c in calls if c["operation"] == "solve"]
        timeout_calls = [c for c in calls if c["operation"] == "timeout"]

        assert len(retrieve_calls) >= 1
        assert "latency_ms" in retrieve_calls[0]
        assert retrieve_calls[0]["latency_ms"] > 0

        assert len(validate_calls) >= 1
        assert "latency_ms" in validate_calls[0]
        assert validate_calls[0]["latency_ms"] >= 0

        assert len(solve_calls) >= 1
        assert "api_cost_usd" in solve_calls[0]
        assert solve_calls[0]["api_cost_usd"] > 0

        assert len(timeout_calls) >= 1
        assert timeout_calls[0]["error_type"] == "ERR_TIMEOUT"
        assert timeout_calls[0]["operation"] == "timeout"

        with token_events_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)

                try:
                    jsonschema.validate(instance=event, schema=schema)
                except jsonschema.ValidationError as e:
                    pytest.fail(f"Schema validation failed: {e}")

                if event["operation"] == "solve":
                    assert event["usage"]["prompt_tokens"] == 100
                    assert event["usage"]["completion_tokens"] == 50
                    assert event["cost_usd"] > 0
                elif event["operation"] == "timeout":
                    assert event["status"] == "timeout"
                    assert event["cost_usd"] == 0.0

    finally:
        settings.execution_timeout_sec = old_timeout
        settings.llm_provider = old_provider
        settings.llm_model = old_model
        if test_dir.exists():
            shutil.rmtree(test_dir)
