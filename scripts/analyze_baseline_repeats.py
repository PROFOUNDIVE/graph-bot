from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple


RUN_ID_RE = re.compile(r"baseline_(io|cot)_r(\d+)_rep(\d+)")


def _load_last_per_t(path: Path) -> List[Dict[str, float]]:
    latest: Dict[int, Dict[str, float]] = {}
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


def _bootstrap_ci(values: List[float], n_resamples: int = 2000) -> Tuple[float, float]:
    if len(values) < 2:
        return (0.0, 0.0)
    random.seed(0)
    means: List[float] = []
    for _ in range(n_resamples):
        sample = [random.choice(values) for _ in values]
        means.append(_mean(sample))
    return (_percentile(means, 2.5), _percentile(means, 97.5))


def _bootstrap_ci_paired(
    values: List[float], n_resamples: int = 2000
) -> Tuple[float, float]:
    if len(values) < 2:
        return (0.0, 0.0)
    random.seed(0)
    means: List[float] = []
    for _ in range(n_resamples):
        sample = [random.choice(values) for _ in values]
        means.append(_mean(sample))
    return (_percentile(means, 2.5), _percentile(means, 97.5))


def _run_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    solved = [float(r.get("solved", 0)) for r in records]
    attempt_success = [float(r.get("attempt_success_rate", 0.0)) for r in records]
    tokens = [float(r.get("tokens_total", 0.0)) for r in records]
    latency = [float(r.get("latency_total_ms", 0.0)) for r in records]
    return {
        "problems": float(len(records)),
        "solved": float(sum(solved)),
        "accuracy": _mean(solved),
        "attempt_success_rate": _mean(attempt_success),
        "tokens_avg": _mean(tokens),
        "latency_avg_ms": _mean(latency),
    }


def _collect_runs(log_dir: Path) -> Dict[Tuple[str, str], Dict[str, Dict[str, float]]]:
    runs: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for path in sorted(log_dir.glob("baseline_*_rep*.problems.jsonl")):
        match = RUN_ID_RE.search(path.stem)
        if not match:
            continue
        mode, retry, rep = match.groups()
        key = (mode, f"r{retry}")
        records = _load_last_per_t(path)
        runs.setdefault(key, {})[rep] = _run_metrics(records)
    return runs


def _write_report(
    out_path: Path, runs: Dict[Tuple[str, str], Dict[str, Dict[str, float]]]
) -> None:
    lines: List[str] = []
    lines.append("# IO vs CoT Baseline Repeats (Variance + CI)")
    lines.append("")
    if not runs:
        lines.append("No baseline repeat logs found (baseline_*_rep*.problems.jsonl).")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("Per-run metrics (mean per problem)")
    lines.append("")
    lines.append(
        "| Method | Retry | Rep | Problems | Solved | Accuracy | Attempt SR | Tokens avg | Latency avg (ms) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for (mode, retry), reps in sorted(runs.items()):
        for rep, metrics in sorted(reps.items()):
            lines.append(
                "| {method} | {retry} | {rep} | {problems:.0f} | {solved:.0f} | {accuracy:.3f} | {asr:.3f} | {tokens:.1f} | {latency:.1f} |".format(
                    method=mode.upper(),
                    retry=retry,
                    rep=rep,
                    problems=metrics["problems"],
                    solved=metrics["solved"],
                    accuracy=metrics["accuracy"],
                    asr=metrics["attempt_success_rate"],
                    tokens=metrics["tokens_avg"],
                    latency=metrics["latency_avg_ms"],
                )
            )

    lines.append("")
    lines.append("Mean and bootstrap 95% CI (across repeats)")
    lines.append("")
    lines.append("| Method | Retry | Metric | Mean | CI 2.5% | CI 97.5% |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for (mode, retry), reps in sorted(runs.items()):
        for metric in [
            "accuracy",
            "attempt_success_rate",
            "tokens_avg",
            "latency_avg_ms",
        ]:
            values = [m[metric] for m in reps.values()]
            mean_val = _mean(values)
            lo, hi = _bootstrap_ci(values)
            lines.append(
                f"| {mode.upper()} | {retry} | {metric} | {mean_val:.3f} | {lo:.3f} | {hi:.3f} |"
            )

    lines.append("")
    lines.append("Paired IO–CoT differences (retry-matched reps)")
    lines.append("")
    lines.append("| Retry | Metric | Mean diff (IO-COT) | CI 2.5% | CI 97.5% | Pairs |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for retry in sorted({key[1] for key in runs}):
        io = runs.get(("io", retry), {})
        cot = runs.get(("cot", retry), {})
        shared = sorted(set(io.keys()) & set(cot.keys()))
        if not shared:
            continue
        for metric in [
            "accuracy",
            "attempt_success_rate",
            "tokens_avg",
            "latency_avg_ms",
        ]:
            diffs = [io[rep][metric] - cot[rep][metric] for rep in shared]
            mean_diff = _mean(diffs)
            lo, hi = _bootstrap_ci_paired(diffs)
            lines.append(
                f"| {retry} | {metric} | {mean_diff:.3f} | {lo:.3f} | {hi:.3f} | {len(shared)} |"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze baseline repeat logs (IO vs CoT)."
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
    print(out_path)


if __name__ == "__main__":
    main()
