"""Analyze EXP5 repeat runs and produce uncertainty-aware reports.

This script scans EXP5 repeat artifacts in a stream log directory, loads both
`*.problems.jsonl` and `*.token_events.jsonl` for each run, computes per-run
metrics, aggregates per-arm means with numpy-only bootstrap confidence
intervals, and reports paired deltas for:

- Graph vs Flat
- GraphExec vs Graph

Outputs:
- Markdown report
- CSV report
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


RUN_FILE_RE = re.compile(
    r"(?P<base>exp5_[a-zA-Z0-9_-]+)_rep(?P<rep>\d+)\.problems\.jsonl"
)
INFER_VALIDATOR_RE = re.compile(r"(oracle|weak_llm_judge|exec_repair)")

METRICS = (
    "accuracy",
    "attempt_success_rate",
    "tokens_p50",
    "tokens_avg",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_avg_ms",
    "cost_avg_usd",
)

PAIRED_METRICS = (
    "accuracy",
    "attempt_success_rate",
    "tokens_avg",
    "latency_avg_ms",
    "cost_avg_usd",
)


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    arm_id: str
    arm_label: str
    rep: str
    mode: str
    validator_mode: str
    expression_format: str
    retrieval_backend: str
    problems: int
    solved: int
    accuracy: float
    attempt_success_rate: float
    tokens_p50: float
    tokens_avg: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_avg_ms: float
    cost_avg_usd: float


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(val) or np.isinf(val):
        return None
    return val


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def _bootstrap_mean_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    n_resamples: int,
    ci: float = 95.0,
) -> Tuple[Optional[float], Optional[float]]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return (None, None)

    indices = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[indices].mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(means, alpha))
    hi = float(np.percentile(means, 100.0 - alpha))
    return (lo, hi)


def _load_last_per_t(path: Path) -> List[Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("t")
            t_val = _safe_float(t)
            if t_val is None:
                continue
            latest[int(t_val)] = obj
    return [latest[key] for key in sorted(latest.keys())]


def _infer_arm_label(arm_id: str, mode: str) -> str:
    mode_l = mode.lower()
    arm_l = arm_id.lower()
    if "exec" in mode_l or "exec" in arm_l:
        return "graph_exec"
    if "flat" in mode_l or "flat" in arm_l:
        return "flat"
    if "graph" in mode_l or "graph" in arm_l:
        return "graph"
    return arm_l


def _infer_expression_format(mode: str, arm_id: str) -> str:
    mode_l = mode.lower()
    arm_l = arm_id.lower()
    if "graph_bot_exec" in mode_l or "exec" in arm_l:
        return "graph_exec"
    if "flat_template_rag" in mode_l or "flat" in arm_l:
        return "flat_template"
    if "graph_bot" in mode_l or "graph" in arm_l:
        return "graph"
    return "unknown"


def _infer_validator_mode(run_id: str) -> str:
    match = INFER_VALIDATOR_RE.search(run_id.lower())
    if match is not None:
        return match.group(1)
    return "unknown"


def _load_run_modes(manifest_path: Path) -> Dict[str, str]:
    run_modes: Dict[str, str] = {}
    if not manifest_path.exists():
        return run_modes
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = obj.get("run_id")
            if not isinstance(run_id, str):
                continue
            config = obj.get("config")
            if not isinstance(config, dict):
                continue
            mode = config.get("mode")
            if isinstance(mode, str):
                run_modes[run_id] = mode
    return run_modes


def _load_run_metadata(
    run_id: str,
    arm_id: str,
    token_events_path: Path,
    run_modes: Mapping[str, str],
    warnings: List[str],
) -> Tuple[str, str, str, str]:
    mode = run_modes.get(run_id, "unknown")
    retrieval_backend = "unknown"
    validator_mode = _infer_validator_mode(run_id)

    if not token_events_path.exists():
        warnings.append(
            f"Missing token_events for run_id={run_id}: {token_events_path.name}"
        )
    else:
        with token_events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata = obj.get("metadata")
                if not isinstance(metadata, dict):
                    continue
                mode_candidate = metadata.get("mode")
                if isinstance(mode_candidate, str) and mode == "unknown":
                    mode = mode_candidate
                backend_candidate = metadata.get("retrieval_backend")
                if (
                    isinstance(backend_candidate, str)
                    and retrieval_backend == "unknown"
                ):
                    retrieval_backend = backend_candidate
                if mode != "unknown" and retrieval_backend != "unknown":
                    break

    expression_format = _infer_expression_format(mode, arm_id)
    return (
        mode,
        validator_mode,
        expression_format,
        retrieval_backend if retrieval_backend else "unknown",
    )


def _compute_run_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    solved: List[float] = []
    attempt_sr: List[float] = []
    tokens: List[float] = []
    latency: List[float] = []
    cost: List[float] = []

    for row in records:
        solved.append(1.0 if bool(row.get("solved", False)) else 0.0)
        attempt_sr.append(_safe_float(row.get("attempt_success_rate")) or 0.0)
        tokens.append(_safe_float(row.get("tokens_total")) or 0.0)
        latency.append(_safe_float(row.get("latency_total_ms")) or 0.0)
        cost.append(_safe_float(row.get("api_cost_usd")) or 0.0)

    return {
        "problems": float(len(records)),
        "solved": float(sum(solved)),
        "accuracy": _mean(solved),
        "attempt_success_rate": _mean(attempt_sr),
        "tokens_p50": _percentile(tokens, 50),
        "tokens_avg": _mean(tokens),
        "latency_p50_ms": _percentile(latency, 50),
        "latency_p95_ms": _percentile(latency, 95),
        "latency_avg_ms": _mean(latency),
        "cost_avg_usd": _mean(cost),
    }


def _collect_runs(
    log_dir: Path, run_modes: Mapping[str, str]
) -> Tuple[List[RunMetrics], List[str]]:
    warnings: List[str] = []
    runs: List[RunMetrics] = []

    for problems_path in sorted(log_dir.glob("exp5*_rep*.problems.jsonl")):
        match = RUN_FILE_RE.fullmatch(problems_path.name)
        if match is None:
            warnings.append(f"Skipped unmatched run file: {problems_path.name}")
            continue

        arm_id = match.group("base")
        rep = match.group("rep")
        run_id = f"{arm_id}_rep{rep}"
        token_events_path = log_dir / f"{run_id}.token_events.jsonl"

        records = _load_last_per_t(problems_path)
        if not records:
            warnings.append(f"No usable records in {problems_path.name}")
            continue

        mode, validator_mode, expression_format, retrieval_backend = _load_run_metadata(
            run_id=run_id,
            arm_id=arm_id,
            token_events_path=token_events_path,
            run_modes=run_modes,
            warnings=warnings,
        )
        arm_label = _infer_arm_label(arm_id, mode)
        values = _compute_run_metrics(records)

        runs.append(
            RunMetrics(
                run_id=run_id,
                arm_id=arm_id,
                arm_label=arm_label,
                rep=rep,
                mode=mode,
                validator_mode=validator_mode,
                expression_format=expression_format,
                retrieval_backend=retrieval_backend,
                problems=int(values["problems"]),
                solved=int(values["solved"]),
                accuracy=values["accuracy"],
                attempt_success_rate=values["attempt_success_rate"],
                tokens_p50=values["tokens_p50"],
                tokens_avg=values["tokens_avg"],
                latency_p50_ms=values["latency_p50_ms"],
                latency_p95_ms=values["latency_p95_ms"],
                latency_avg_ms=values["latency_avg_ms"],
                cost_avg_usd=values["cost_avg_usd"],
            )
        )

    return (runs, warnings)


def _group_by_arm(runs: Sequence[RunMetrics]) -> Dict[str, List[RunMetrics]]:
    grouped: Dict[str, List[RunMetrics]] = {}
    for run in runs:
        grouped.setdefault(run.arm_id, []).append(run)
    for arm_runs in grouped.values():
        arm_runs.sort(key=lambda x: x.rep)
    return grouped


def _aggregate_by_arm(
    runs_by_arm: Mapping[str, Sequence[RunMetrics]],
    rng: np.random.Generator,
    n_resamples: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for arm_id, arm_runs in sorted(runs_by_arm.items()):
        n = len(arm_runs)
        if n == 0:
            continue
        exemplar = arm_runs[0]

        for metric in METRICS:
            values = [float(getattr(run, metric)) for run in arm_runs]
            mean_val = _mean(values)
            ci_low, ci_high = _bootstrap_mean_ci(
                values=values,
                rng=rng,
                n_resamples=n_resamples,
            )
            rows.append(
                {
                    "arm_id": arm_id,
                    "arm_label": exemplar.arm_label,
                    "mode": exemplar.mode,
                    "validator_mode": exemplar.validator_mode,
                    "expression_format": exemplar.expression_format,
                    "retrieval_backend": exemplar.retrieval_backend,
                    "metric": metric,
                    "n": n,
                    "mean": mean_val,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return rows


def _best_run_per_rep_by_label(
    runs: Sequence[RunMetrics],
) -> Dict[str, Dict[str, RunMetrics]]:
    grouped: Dict[str, Dict[str, RunMetrics]] = {}
    for run in sorted(runs, key=lambda x: x.run_id):
        per_rep = grouped.setdefault(run.arm_label, {})
        if run.rep not in per_rep:
            per_rep[run.rep] = run
    return grouped


def _paired_delta_rows(
    runs: Sequence[RunMetrics],
    rng: np.random.Generator,
    n_resamples: int,
) -> List[Dict[str, Any]]:
    per_label = _best_run_per_rep_by_label(runs)
    comparisons = (
        ("graph_vs_flat", "graph", "flat"),
        ("graph_exec_vs_graph", "graph_exec", "graph"),
    )

    rows: List[Dict[str, Any]] = []
    for comp_name, lhs, rhs in comparisons:
        lhs_runs = per_label.get(lhs, {})
        rhs_runs = per_label.get(rhs, {})
        common_reps = sorted(set(lhs_runs.keys()) & set(rhs_runs.keys()))
        for metric in PAIRED_METRICS:
            deltas = [
                float(getattr(lhs_runs[rep], metric) - getattr(rhs_runs[rep], metric))
                for rep in common_reps
            ]
            mean_val = _mean(deltas)
            ci_low, ci_high = _bootstrap_mean_ci(
                values=deltas,
                rng=rng,
                n_resamples=n_resamples,
            )
            rows.append(
                {
                    "comparison": comp_name,
                    "lhs": lhs,
                    "rhs": rhs,
                    "metric": metric,
                    "n_pairs": len(common_reps),
                    "delta_mean": mean_val,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return rows


def _fmt_ci(low: Optional[float], high: Optional[float], precision: int = 4) -> str:
    if low is None or high is None:
        return "n<2"
    return f"[{low:.{precision}f}, {high:.{precision}f}]"


def _write_markdown(
    out_path: Path,
    runs: Sequence[RunMetrics],
    arm_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
) -> None:
    lines: List[str] = []
    lines.append("# EXP5 Repeats Analysis (v0.7.1)")
    lines.append("")

    if not runs:
        lines.append("No EXP5 repeat runs found (`exp5*_rep*.problems.jsonl`).")
        if warnings:
            lines.append("")
            lines.append("## Warnings")
            lines.append("")
            for w in warnings:
                lines.append(f"- {w}")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append(f"- Runs discovered: **{len(runs)}**")
    lines.append(f"- Arms discovered: **{len({run.arm_id for run in runs})}**")
    lines.append("")

    lines.append("## Per-Run Metrics")
    lines.append("")
    lines.append(
        "| arm | run_id | rep | validator_mode | expression_format | accuracy | "
        "attempt SR | tokens p50/avg | latency p50/p95/avg (ms) | api_cost avg |"
    )
    lines.append(
        "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |"
    )

    for run in sorted(runs, key=lambda x: x.run_id):
        tokens_str = f"{run.tokens_p50:.1f} / {run.tokens_avg:.1f}"
        latency_str = f"{run.latency_p50_ms:.1f} / {run.latency_p95_ms:.1f} / {run.latency_avg_ms:.1f}"
        lines.append(
            f"| {run.arm_id} | {run.run_id} | {run.rep} | {run.validator_mode} | "
            f"{run.expression_format} | {run.accuracy:.4f} | {run.attempt_success_rate:.4f} | "
            f"{tokens_str} | {latency_str} | {run.cost_avg_usd:.6f} |"
        )
    lines.append("")

    lines.append("## Per-Arm Mean with 95% Bootstrap CI")
    lines.append("")
    lines.append("| arm | n | metric | mean | 95% CI |")
    lines.append("| :--- | ---: | :--- | ---: | :--- |")
    for row in sorted(arm_rows, key=lambda x: (str(x["arm_id"]), str(x["metric"]))):
        lines.append(
            f"| {row['arm_id']} | {int(row['n'])} | {row['metric']} | {float(row['mean']):.6f} | "
            f"{_fmt_ci(row['ci_low'], row['ci_high'])} |"
        )
    lines.append("")

    lines.append("## Paired Bootstrap on Repeat-Matched Deltas")
    lines.append("")
    lines.append("| comparison | metric | n_pairs | delta mean | 95% CI |")
    lines.append("| :--- | :--- | ---: | ---: | :--- |")
    for row in paired_rows:
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {int(row['n_pairs'])} | "
            f"{float(row['delta_mean']):.6f} | {_fmt_ci(row['ci_low'], row['ci_high'])} |"
        )

    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.append("")
    lines.append("*Generated by `scripts/analyze_exp5_repeats.py`.*")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(
    out_path: Path,
    runs: Sequence[RunMetrics],
    arm_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "comparison",
        "arm_id",
        "arm_label",
        "run_id",
        "rep",
        "mode",
        "validator_mode",
        "expression_format",
        "retrieval_backend",
        "metric",
        "n",
        "value",
        "ci_low",
        "ci_high",
        "accuracy",
        "attempt_success_rate",
        "tokens_p50",
        "tokens_avg",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_avg_ms",
        "cost_avg_usd",
        "problems",
        "solved",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for run in sorted(runs, key=lambda x: x.run_id):
            writer.writerow(
                {
                    "row_type": "run",
                    "comparison": "",
                    "arm_id": run.arm_id,
                    "arm_label": run.arm_label,
                    "run_id": run.run_id,
                    "rep": run.rep,
                    "mode": run.mode,
                    "validator_mode": run.validator_mode,
                    "expression_format": run.expression_format,
                    "retrieval_backend": run.retrieval_backend,
                    "metric": "",
                    "n": "",
                    "value": "",
                    "ci_low": "",
                    "ci_high": "",
                    "accuracy": run.accuracy,
                    "attempt_success_rate": run.attempt_success_rate,
                    "tokens_p50": run.tokens_p50,
                    "tokens_avg": run.tokens_avg,
                    "latency_p50_ms": run.latency_p50_ms,
                    "latency_p95_ms": run.latency_p95_ms,
                    "latency_avg_ms": run.latency_avg_ms,
                    "cost_avg_usd": run.cost_avg_usd,
                    "problems": run.problems,
                    "solved": run.solved,
                }
            )

        for row in arm_rows:
            writer.writerow(
                {
                    "row_type": "arm_summary",
                    "comparison": "",
                    "arm_id": row["arm_id"],
                    "arm_label": row["arm_label"],
                    "run_id": "",
                    "rep": "",
                    "mode": row["mode"],
                    "validator_mode": row["validator_mode"],
                    "expression_format": row["expression_format"],
                    "retrieval_backend": row["retrieval_backend"],
                    "metric": row["metric"],
                    "n": row["n"],
                    "value": row["mean"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "accuracy": "",
                    "attempt_success_rate": "",
                    "tokens_p50": "",
                    "tokens_avg": "",
                    "latency_p50_ms": "",
                    "latency_p95_ms": "",
                    "latency_avg_ms": "",
                    "cost_avg_usd": "",
                    "problems": "",
                    "solved": "",
                }
            )

        for row in paired_rows:
            writer.writerow(
                {
                    "row_type": "paired_delta",
                    "comparison": row["comparison"],
                    "arm_id": "",
                    "arm_label": "",
                    "run_id": "",
                    "rep": "",
                    "mode": "",
                    "validator_mode": "",
                    "expression_format": "",
                    "retrieval_backend": "",
                    "metric": row["metric"],
                    "n": row["n_pairs"],
                    "value": row["delta_mean"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "accuracy": "",
                    "attempt_success_rate": "",
                    "tokens_p50": "",
                    "tokens_avg": "",
                    "latency_p50_ms": "",
                    "latency_p95_ms": "",
                    "latency_avg_ms": "",
                    "cost_avg_usd": "",
                    "problems": "",
                    "solved": "",
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze EXP5 repeat runs with per-arm bootstrap CIs and paired "
            "bootstrap deltas (Graph vs Flat, GraphExec vs Graph)."
        )
    )
    parser.add_argument(
        "--log-dir",
        default="/home/hyunwoo/git/graph-bot/outputs/stream_logs",
        help="Directory containing exp5*_rep*.problems.jsonl and token_events logs.",
    )
    parser.add_argument(
        "--out-md",
        default="/home/hyunwoo/git/graph-bot/outputs/exp5_report_v0.7.1.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--out-csv",
        default="/home/hyunwoo/git/graph-bot/outputs/exp5_report_v0.7.1.csv",
        help="Output CSV report path.",
    )
    parser.add_argument(
        "--manifest-path",
        default="/home/hyunwoo/git/graph-bot/outputs/run_manifest.jsonl",
        help="Path to run_manifest.jsonl used for mode inference.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Number of bootstrap resamples (numpy-only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic bootstrap.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_dir = Path(args.log_dir)
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    manifest_path = Path(args.manifest_path)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    run_modes = _load_run_modes(manifest_path)
    runs, warnings = _collect_runs(log_dir=log_dir, run_modes=run_modes)

    rng = np.random.default_rng(args.seed)
    runs_by_arm = _group_by_arm(runs)
    arm_rows = _aggregate_by_arm(
        runs_by_arm=runs_by_arm,
        rng=rng,
        n_resamples=args.bootstrap_resamples,
    )
    paired_rows = _paired_delta_rows(
        runs=runs,
        rng=rng,
        n_resamples=args.bootstrap_resamples,
    )

    _write_markdown(
        out_path=out_md,
        runs=runs,
        arm_rows=arm_rows,
        paired_rows=paired_rows,
        warnings=warnings,
    )
    _write_csv(
        out_path=out_csv,
        runs=runs,
        arm_rows=arm_rows,
        paired_rows=paired_rows,
    )

    print(f"Markdown report written to: {out_md}")
    print(f"CSV report written to: {out_csv}")
    if warnings:
        print(f"Warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
