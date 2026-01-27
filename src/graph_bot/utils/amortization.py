from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np


def generate_amortization_curve(
    *,
    stream_metrics_jsonl: Path,
    out_csv: Path,
    problems_jsonl: Path | None = None,
    token_events_jsonl: Path | None = None,
) -> None:
    """Generate EXP1 amortization curve data from stream metrics JSONL.

    Input: JSONL from StreamMetricsLogger ("*.stream.jsonl")
    Output: CSV with columns: t, cum_accuracy, attempt_success_rate_p50, tokens_total_p50, latency_ms_p50, latency_ms_p95, cost_usd_pipeline_avg, cost_usd_total_avg
    """
    stream_rows = []
    with stream_metrics_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                stream_rows.append(json.loads(line))

    problems_rows = []
    if problems_jsonl and problems_jsonl.exists():
        with problems_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    problems_rows.append(json.loads(line))

    events_rows = []
    if token_events_jsonl and token_events_jsonl.exists():
        with token_events_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events_rows.append(json.loads(line))

    header = [
        "t",
        "cum_accuracy",
        "attempt_success_rate_p50",
        "tokens_total_p50",
        "latency_ms_p50",
        "latency_ms_p95",
        "cost_usd_pipeline_avg",
        "cost_usd_total_avg",
    ]

    problems_by_t = {int(p["t"]): p for p in problems_rows if "t" in p}
    events_by_t: dict[int, list[dict]] = {}
    for ev in events_rows:
        t_val = ev.get("t")
        if t_val is not None:
            events_by_t.setdefault(int(t_val), []).append(ev)

    current_problems = []
    total_pipeline_cost = 0.0
    total_cost = 0.0
    rows: list[list[str]] = []

    for stream_row in stream_rows:
        t = int(stream_row["t"])
        cum_solved = int(stream_row["cumulative_solved"])
        cum_accuracy = cum_solved / t

        if t in problems_by_t:
            current_problems.append(problems_by_t[t])

        success_rates = [p["attempt_success_rate"] for p in current_problems]
        tokens_totals = [p["tokens_total"] for p in current_problems]
        latencies = [p["latency_total_ms"] for p in current_problems]

        asr_p50 = statistics.median(success_rates) if success_rates else 0.0
        tokens_p50 = statistics.median(tokens_totals) if tokens_totals else 0.0
        latency_p50 = statistics.median(latencies) if latencies else 0.0
        latency_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

        if t in events_by_t:
            for ev in events_by_t[t]:
                cost = float(ev.get("cost_usd", 0.0))
                total_cost += cost
                if ev.get("component") == "pipeline":
                    total_pipeline_cost += cost

        pipeline_avg = total_pipeline_cost / t
        total_avg = total_cost / t

        rows.append(
            [
                str(t),
                str(cum_accuracy),
                str(asr_p50),
                str(tokens_p50),
                str(latency_p50),
                str(latency_p95),
                str(pipeline_avg),
                str(total_avg),
            ]
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
