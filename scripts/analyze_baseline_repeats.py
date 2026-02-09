from __future__ import annotations

"""Analyze baseline repeat logs (IO vs CoT) for variance and uncertainty.

Generates quantitative report with:
- Per-run metrics table (IO rep01-05, CoT rep01)
- Aggregated statistics with 95% CI (IO: n=5, CoT: n=1 with caveat)
- IO-CoT difference with caveat about asymmetric samples
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import scipy.stats as stats  # type: ignore


RUN_ID_RE = re.compile(r"baseline_(io|cot)_r(\d+)_rep(\d+)")


def _load_last_per_t(path: Path) -> List[Dict[str, Any]]:
    """Load problems.jsonl and return only the last entry per t value."""
    latest: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            t = obj.get("t")
            if t is None:
                continue
            latest[int(t)] = obj
    return [latest[key] for key in sorted(latest.keys())]


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def _ci_95(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Calculate 95% CI using scipy.stats (t-distribution).

    Returns (lower, upper) or (None, None) if n < 2.
    """
    n = len(values)
    if n < 2:
        return (None, None)
    mean = _mean(values)
    sem = stats.sem(values)
    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
    return (ci[0], ci[1])


def _bootstrap_ci(
    values: List[float], n_resamples: int = 9999
) -> Tuple[Optional[float], Optional[float]]:
    """Bootstrap 95% CI using scipy.stats.bootstrap.

    Returns (lower, upper) or (None, None) if n < 2.
    """
    n = len(values)
    if n < 2:
        return (None, None)
    data = (np.array(values),)
    result = stats.bootstrap(data, np.mean, n_resamples=n_resamples, rng=42)
    return (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def _run_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract per-run metrics from a list of problem records.

    Uses p50/p95/avg for tokens and latency per latency policy.
    """
    solved = [float(r.get("solved", 0)) for r in records]
    attempt_success = [float(r.get("attempt_success_rate", 0.0)) for r in records]
    tokens = [float(r.get("tokens_total", 0.0)) for r in records]
    latency = [float(r.get("latency_total_ms", 0.0)) for r in records]
    cost = [float(r.get("api_cost_usd", 0.0)) for r in records]

    return {
        "problems": float(len(records)),
        "solved": float(sum(solved)),
        "accuracy": _mean(solved),
        "attempt_success_rate": _mean(attempt_success),
        # Tokens: p50/p95/avg
        "tokens_p50": _percentile(tokens, 50),
        "tokens_p95": _percentile(tokens, 95),
        "tokens_avg": _mean(tokens),
        # Latency: p50/p95/avg (in ms)
        "latency_p50_ms": _percentile(latency, 50),
        "latency_p95_ms": _percentile(latency, 95),
        "latency_avg_ms": _mean(latency),
        # Cost
        "cost_avg_usd": _mean(cost),
    }


def _collect_runs(log_dir: Path) -> Dict[Tuple[str, str], Dict[str, Dict[str, float]]]:
    """Collect all baseline repeat runs from log directory.

    Returns: {(mode, retry): {rep_id: metrics_dict}}
    """
    runs: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for path in sorted(log_dir.glob("baseline_*_rep*.problems.jsonl")):
        match = RUN_ID_RE.search(path.stem)
        if not match:
            continue
        mode, retry, rep = match.groups()
        key = (mode, f"r{retry}")
        records = _load_last_per_t(path)
        # Only include runs with t=98 (complete runs)
        if len(records) < 98:
            continue
        runs.setdefault(key, {})[rep] = _run_metrics(records)
    return runs


def _write_report(
    out_path: Path, runs: Dict[Tuple[str, str], Dict[str, Dict[str, float]]]
) -> None:
    """Generate markdown report with quantitative analysis."""
    lines: List[str] = []
    lines.append("# IO vs CoT Baseline Repeats: Quantitative Analysis")
    lines.append("")

    if not runs:
        lines.append("No baseline repeat logs found (baseline_*_rep*.problems.jsonl).")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    # Count runs per mode
    io_runs = runs.get(("io", "r3"), {})
    cot_runs = runs.get(("cot", "r3"), {})
    io_n = len(io_runs)
    cot_n = len(cot_runs)

    lines.append("## Sample Sizes")
    lines.append("")
    lines.append(f"- **IO (r3)**: n={io_n} runs")
    lines.append(f"- **CoT (r3)**: n={cot_n} run(s)")
    if cot_n == 1:
        lines.append("")
        lines.append(
            "> ⚠️ **Caveat**: CoT has only n=1 run. No confidence intervals or "
            "statistical comparisons are possible. IO-CoT differences are reported "
            "as point estimates only."
        )
    lines.append("")

    # Per-run metrics table
    lines.append("## Per-Run Metrics")
    lines.append("")
    lines.append(
        "| Method | Rep | Solved/98 | Accuracy | Attempt SR | "
        "Tokens (p50/p95/avg) | Latency ms (p50/p95/avg) | Cost/prob |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for (mode, retry), reps in sorted(runs.items()):
        for rep, m in sorted(reps.items()):
            tokens_str = (
                f"{m['tokens_p50']:.0f} / {m['tokens_p95']:.0f} / {m['tokens_avg']:.0f}"
            )
            latency_str = (
                f"{m['latency_p50_ms']:.0f} / {m['latency_p95_ms']:.0f} / "
                f"{m['latency_avg_ms']:.0f}"
            )
            lines.append(
                f"| {mode.upper()} | {rep} | {m['solved']:.0f}/98 | "
                f"{m['accuracy']:.3f} | {m['attempt_success_rate']:.3f} | "
                f"{tokens_str} | {latency_str} | ${m['cost_avg_usd']:.5f} |"
            )

    lines.append("")

    # Aggregated statistics
    lines.append("## Aggregated Statistics with 95% CI")
    lines.append("")
    lines.append(
        "| Method | n | Metric | Mean | 95% CI Low | 95% CI High | Bootstrap CI Low | Bootstrap CI High |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for mode_key in [("io", "r3"), ("cot", "r3")]:
        if mode_key not in runs:
            continue
        reps = runs[mode_key]
        mode = mode_key[0].upper()
        n = len(reps)

        metrics_to_aggregate = [
            ("accuracy", "Accuracy"),
            ("attempt_success_rate", "Attempt SR"),
            ("tokens_avg", "Tokens (avg)"),
            ("latency_avg_ms", "Latency (avg ms)"),
            ("cost_avg_usd", "Cost/prob"),
        ]

        for metric_key, metric_label in metrics_to_aggregate:
            values = [m[metric_key] for m in reps.values()]
            mean_val = _mean(values)

            if n >= 2:
                ci_lo, ci_hi = _ci_95(values)
                boot_lo, boot_hi = _bootstrap_ci(values)
                ci_lo_str = f"{ci_lo:.4f}" if ci_lo is not None else "—"
                ci_hi_str = f"{ci_hi:.4f}" if ci_hi is not None else "—"
                boot_lo_str = f"{boot_lo:.4f}" if boot_lo is not None else "—"
                boot_hi_str = f"{boot_hi:.4f}" if boot_hi is not None else "—"
            else:
                ci_lo_str = "n=1"
                ci_hi_str = "n=1"
                boot_lo_str = "n=1"
                boot_hi_str = "n=1"

            lines.append(
                f"| {mode} | {n} | {metric_label} | {mean_val:.4f} | "
                f"{ci_lo_str} | {ci_hi_str} | {boot_lo_str} | {boot_hi_str} |"
            )

    lines.append("")

    # IO-CoT difference
    lines.append("## IO vs CoT Difference")
    lines.append("")

    if io_n == 0 or cot_n == 0:
        lines.append("Insufficient data for comparison.")
    else:
        lines.append(
            "> **Note**: With CoT n=1, bootstrap CI for the IO-CoT difference is "
            "infeasible. Only point estimates (IO mean - CoT single value) are reported."
        )
        lines.append("")
        lines.append("| Metric | IO Mean | CoT (n=1) | Δ (IO - CoT) |")
        lines.append("| :--- | :--- | :--- | :--- |")

        # Get CoT single-run values
        cot_rep = list(cot_runs.values())[0]

        metrics_to_compare = [
            ("accuracy", "Accuracy"),
            ("attempt_success_rate", "Attempt SR"),
            ("tokens_avg", "Tokens (avg)"),
            ("latency_avg_ms", "Latency (avg ms)"),
            ("cost_avg_usd", "Cost/prob"),
        ]

        for metric_key, metric_label in metrics_to_compare:
            io_values = [m[metric_key] for m in io_runs.values()]
            io_mean = _mean(io_values)
            cot_val = cot_rep[metric_key]
            delta = io_mean - cot_val

            if metric_key == "cost_avg_usd":
                lines.append(
                    f"| {metric_label} | ${io_mean:.5f} | ${cot_val:.5f} | ${delta:+.5f} |"
                )
            elif metric_key in ("tokens_avg", "latency_avg_ms"):
                lines.append(
                    f"| {metric_label} | {io_mean:.1f} | {cot_val:.1f} | {delta:+.1f} |"
                )
            else:
                lines.append(
                    f"| {metric_label} | {io_mean:.4f} | {cot_val:.4f} | {delta:+.4f} |"
                )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by `scripts/analyze_baseline_repeats.py`*")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze baseline repeat logs (IO vs CoT) with variance and CI."
    )
    parser.add_argument(
        "--log-dir",
        default="/home/hyunwoo/git/graph-bot/outputs/stream_logs",
        help="Directory containing baseline_*_rep*.problems.jsonl logs.",
    )
    parser.add_argument(
        "--out",
        default="/home/hyunwoo/git/graph-bot/outputs/week7_cot_vs_io_report.md",
        help="Output markdown report path.",
    )
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_path = Path(args.out)
    runs = _collect_runs(log_dir)
    _write_report(out_path, runs)
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
