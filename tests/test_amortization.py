from __future__ import annotations

import json
from pathlib import Path

from graph_bot.utils.amortization import generate_amortization_curve


def test_generate_amortization_curve_metrics(tmp_path: Path):
    # Setup files
    stream_metrics_jsonl = tmp_path / "stream.jsonl"
    problems_jsonl = tmp_path / "problems.jsonl"
    out_csv = tmp_path / "out.csv"

    # Data: 3 data points skewed to ensure Mean != Median.
    # T=1: tokens=1000
    # T=2: tokens=10
    # T=3: tokens=10
    # (Median=10, Mean=(1000+10+10)/3 = 340)

    stream_data = [
        {"t": 1, "cumulative_solved": 1},
        {"t": 2, "cumulative_solved": 2},
        {"t": 3, "cumulative_solved": 3},
    ]

    problems_data = [
        {
            "t": 1,
            "attempt_success_rate": 1.0,
            "tokens_total": 1000,
            "latency_total_ms": 100,
        },
        {
            "t": 2,
            "attempt_success_rate": 0.5,
            "tokens_total": 10,
            "latency_total_ms": 10,
        },
        {
            "t": 3,
            "attempt_success_rate": 0.5,
            "tokens_total": 10,
            "latency_total_ms": 10,
        },
    ]

    with stream_metrics_jsonl.open("w", encoding="utf-8") as f:
        for d in stream_data:
            f.write(json.dumps(d) + "\n")

    with problems_jsonl.open("w", encoding="utf-8") as f:
        for d in problems_data:
            f.write(json.dumps(d) + "\n")

    # Action
    generate_amortization_curve(
        stream_metrics_jsonl=stream_metrics_jsonl,
        out_csv=out_csv,
        problems_jsonl=problems_jsonl,
    )

    # Assertions
    assert out_csv.exists()
    content = out_csv.read_text(encoding="utf-8").splitlines()
    header = content[0].split(",")

    assert "attempt_success_rate_avg" in header
    assert "tokens_total_avg" in header

    last_row = content[-1].split(",")

    asr_avg_idx = header.index("attempt_success_rate_avg")
    tokens_avg_idx = header.index("tokens_total_avg")

    assert float(last_row[tokens_avg_idx]) == 340.0
    assert abs(float(last_row[asr_avg_idx]) - 0.6666666666666666) < 1e-6
