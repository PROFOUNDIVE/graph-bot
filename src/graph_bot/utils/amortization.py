from __future__ import annotations

import json
from pathlib import Path


def generate_amortization_curve(
    *,
    stream_metrics_jsonl: Path,
    out_csv: Path,
) -> None:
    """Generate EXP1 amortization curve data from stream metrics JSONL.

    Input: JSONL from StreamMetricsLogger ("*.stream.jsonl")
    Output: CSV with columns: t,cumulative_api_cost_usd,cumulative_solved,cost_per_solved
    """
    rows: list[dict[str, object]] = []

    with stream_metrics_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            t = int(obj["t"])
            cumulative_cost = float(obj["cumulative_api_cost_usd"])
            cumulative_solved = int(obj["cumulative_solved"])
            cost_per_solved = float(obj["cost_per_solved"])
            rows.append(
                {
                    "t": t,
                    "cumulative_api_cost_usd": cumulative_cost,
                    "cumulative_solved": cumulative_solved,
                    "cost_per_solved": cost_per_solved,
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("t,cumulative_api_cost_usd,cumulative_solved,cost_per_solved\n")
        for row in rows:
            f.write(
                f"{row['t']},{row['cumulative_api_cost_usd']},{row['cumulative_solved']},{row['cost_per_solved']}\n"
            )
