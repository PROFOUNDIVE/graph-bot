from __future__ import annotations

import json
from pathlib import Path

from graph_bot.pipelines.stream_loop import run_continual_stream
from graph_bot.settings import settings


def test_run_continual_stream_accepts_graph_bot_no_edges_alias(tmp_path: Path) -> None:
    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(
        json.dumps({"id": "q1", "numbers": [2, 5, 8, 11], "target": 24}) + "\n",
        encoding="utf-8",
    )
    metrics_dir = tmp_path / "metrics"

    old_provider = settings.llm_provider
    old_model = settings.llm_model
    try:
        settings.llm_provider = "mock"
        settings.llm_model = "mock"
        for alias_mode, run_id in (
            ("graph_bot_no_edges", "test_mode_alias_no_edges"),
            ("graph_bot_no", "test_mode_alias_no"),
        ):
            run_continual_stream(
                problems_file=problems_file,
                mode=alias_mode,
                use_edges=True,
                metrics_out_dir=metrics_dir,
                run_id=run_id,
                max_problems=1,
            )
    finally:
        settings.llm_provider = old_provider
        settings.llm_model = old_model

    assert (metrics_dir / "test_mode_alias_no_edges.stream.jsonl").exists()
    assert (metrics_dir / "test_mode_alias_no.stream.jsonl").exists()
