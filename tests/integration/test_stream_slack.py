from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from graph_bot.pipelines.stream_loop import run_continual_stream
from graph_bot.settings import settings


def test_run_stream_sends_slack_success(tmp_path, monkeypatch):
    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(
        json.dumps({"id": "q1", "numbers": [2, 4, 6, 8], "target": 24}) + "\n",
        encoding="utf-8",
    )
    metrics_dir = tmp_path / "metrics"

    # Set webhook URL
    monkeypatch.setattr(settings, "slack_webhook_url", "http://slack.com/webhook")
    monkeypatch.setattr(settings, "metagraph_path", tmp_path / "metagraph.json")
    monkeypatch.setattr(settings, "llm_provider", "mock")

    # Patch send_slack_notification in stream_loop module
    with patch("graph_bot.pipelines.stream_loop.send_slack_notification") as mock_send:
        run_continual_stream(
            problems_file=problems_file,
            mode="io",
            max_problems=1,
            metrics_out_dir=metrics_dir,
            run_id="test-slack-success",
        )

        mock_send.assert_called_once()
        args, _ = mock_send.call_args
        url, payload = args
        assert url == "http://slack.com/webhook"
        assert "Completed" in payload["text"]
        assert "Status: completed" in payload["text"]
        assert "Solved: 1" in payload["text"]


def test_run_stream_sends_slack_failure(tmp_path, monkeypatch):
    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(
        json.dumps({"id": "q1", "numbers": [2, 4, 6, 8], "target": 24}) + "\n",
        encoding="utf-8",
    )
    metrics_dir = tmp_path / "metrics"

    monkeypatch.setattr(settings, "slack_webhook_url", "http://slack.com/webhook")
    monkeypatch.setattr(settings, "metagraph_path", tmp_path / "metagraph.json")
    monkeypatch.setattr(settings, "llm_provider", "mock")

    # Raise exception that bubbles up (e.g. from inside the inner exception handler or outside inner try)
    # We patch StreamMetricsLogger.log_problem. If it raises, it might be caught by inner except if called there.
    # But if called in inner except, it bubbles up.
    # Actually, log_problem is called at the end of the try block. If it raises there, it goes to inner except.
    # Then inner except calls log_problem AGAIN. If that raises, it bubbles up.
    with (
        patch(
            "graph_bot.pipelines.metrics_logger.StreamMetricsLogger.log_problem",
            side_effect=ValueError("Fatal Error"),
        ),
        patch("graph_bot.pipelines.stream_loop.send_slack_notification") as mock_send,
    ):
        with pytest.raises(ValueError, match="Fatal Error"):
            run_continual_stream(
                problems_file=problems_file,
                mode="io",
                max_problems=1,
                metrics_out_dir=metrics_dir,
                run_id="test-slack-failure",
            )

        mock_send.assert_called_once()
        args, _ = mock_send.call_args
        url, payload = args
        assert url == "http://slack.com/webhook"
        assert "Failed" in payload["text"]
        assert "Status: failed" in payload["text"]
        assert "Fatal Error" in payload["text"]
