import json
import numpy as np
import glob


def calculate_stats():
    run_files = glob.glob("outputs/stream_logs/exp1_run*.problems.jsonl")

    all_solved = []
    all_costs = []
    all_latencies = []
    all_tokens = []

    total_problems = 0
    total_solved_count = 0

    for fpath in run_files:
        with open(fpath, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        total_problems += len(lines)
        for p in lines:
            solved = p.get("solved", False)
            cost = p.get("api_cost_usd", 0.0)
            latency = p.get("latency_total_ms", 0.0)
            # tokens might be in 'usage' dict or separate fields depending on schema
            # Checking metrics_definitions.md, usually total_tokens
            tokens = p.get("tokens_total", 0)

            if solved:
                total_solved_count += 1

            all_solved.append(1 if solved else 0)
            all_costs.append(cost)
            all_latencies.append(latency)
            all_tokens.append(tokens)

    # Averages across runs
    accuracy = (total_solved_count / total_problems) * 100 if total_problems else 0
    avg_cost = np.mean(all_costs) if all_costs else 0
    total_cost_per_run = sum(all_costs) / len(run_files) if run_files else 0  # approx

    p50_latency = np.percentile(all_latencies, 50) / 1000.0 if all_latencies else 0
    p95_latency = np.percentile(all_latencies, 95) / 1000.0 if all_latencies else 0

    p50_tokens = np.percentile(all_tokens, 50) if all_tokens else 0
    avg_tokens = np.mean(all_tokens) if all_tokens else 0

    print(f"Accuracy: {accuracy:.2f}% ({total_solved_count}/{total_problems})")
    print(f"Avg Cost per Problem: ${avg_cost:.6f}")
    print(f"Total Cost (approx per run N=98): ${avg_cost * 98:.4f}")
    print(f"Latency (s): p50 {p50_latency:.2f}, p95 {p95_latency:.2f}")
    print(f"Tokens: p50 {p50_tokens:.1f}, avg {avg_tokens:.1f}")
    print(f"Runs included: {len(run_files)}")


if __name__ == "__main__":
    calculate_stats()
