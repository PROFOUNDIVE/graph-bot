import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def load_stream_log(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Handle appended logs: split by resets (t=1) and find the longest one
    if "t" in df.columns:
        reset_indices = df.index[df["t"] == 1].tolist()
        if not reset_indices:
            return df

        # Add end index to simplify loop
        reset_indices.append(len(df))

        longest_df = pd.DataFrame()

        for i in range(len(reset_indices) - 1):
            start = reset_indices[i]
            # The segment goes until the next reset index
            end = reset_indices[i + 1]
            sub_df = df.iloc[start:end]
            if len(sub_df) > len(longest_df):
                longest_df = sub_df

        return longest_df

    return df


def analyze_exp1():
    print("Analyzing EXP1...")
    log_files = sorted(glob.glob("outputs/stream_logs/exp1_run*.stream.jsonl"))
    if not log_files:
        print("No EXP1 logs found.")
        return

    plt.figure(figsize=(10, 6))

    for log_file in log_files:
        run_id = os.path.basename(log_file).replace(".stream.jsonl", "")
        try:
            df = load_stream_log(log_file)
            if df.empty:
                print(f"Skipping empty log: {log_file}")
                continue
            # Plot cumulative solved vs cumulative cost
            plt.plot(
                df["cumulative_api_cost_usd"], df["cumulative_solved"], label=run_id
            )
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    plt.xlabel("Cumulative API Cost (USD)")
    plt.ylabel("Cumulative Solved Problems")
    plt.title("EXP1: Cost Efficiency")
    plt.legend()
    plt.grid(True)

    # Add baseline reference lines for IO/CoT (retry=3)
    # From outputs/week7_midweek_report.md:
    # - IO (retry=3): accuracy=17.35%, cost/run=$0.00915, N=98 -> 17 solved @ $0.00915
    # - CoT (retry=3): accuracy=16.33%, cost/run=$0.02544, N=98 -> 16 solved @ $0.02544
    IO_SOLVED = 98 * 0.1735
    IO_COST = 0.00915
    COT_SOLVED = 98 * 0.1633
    COT_COST = 0.02544

    plt.axhline(
        y=IO_SOLVED,
        color="#E74C3C",
        linestyle=":",
        linewidth=2,
        label=f"IO (retry=3): {IO_SOLVED:.0f} solved @ ${IO_COST:.4f}",
    )
    plt.axvline(
        x=IO_COST,
        color="#E74C3C",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
    )
    plt.scatter([IO_COST], [IO_SOLVED], color="#E74C3C", s=80, zorder=5, marker="x")

    plt.axhline(
        y=COT_SOLVED,
        color="#F39C12",
        linestyle=":",
        linewidth=2,
        label=f"CoT (retry=3): {COT_SOLVED:.0f} solved @ ${COT_COST:.4f}",
    )
    plt.axvline(
        x=COT_COST,
        color="#F39C12",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
    )
    plt.scatter([COT_COST], [COT_SOLVED], color="#F39C12", s=80, zorder=5, marker="x")

    plt.legend(loc="lower right")

    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/exp1_cost_vs_solved.png")
    print("Saved outputs/figures/exp1_cost_vs_solved.png")


def analyze_exp3():
    print("Analyzing EXP3...")
    log_files = sorted(glob.glob("outputs/stream_logs/exp3_noval_run*.stream.jsonl"))
    if not log_files:
        print("No EXP3 logs found.")
        return

    results = []
    for log_file in log_files:
        run_id = os.path.basename(log_file).replace(".stream.jsonl", "")
        try:
            df = load_stream_log(log_file)
            if df.empty:
                continue

            # Calculate metrics
            avg_contamination = df["contamination_rate"].mean()
            final_contamination = df["contamination_rate"].iloc[-1]
            total_solved = df["cumulative_solved"].iloc[-1]

            # Simple "collapse" detection: when solved count stops increasing for N steps?
            # Or just report raw stats.

            results.append(
                {
                    "run_id": run_id,
                    "avg_contamination_rate": avg_contamination,
                    "final_contamination_rate": final_contamination,
                    "total_solved": total_solved,
                }
            )
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    df_res = pd.DataFrame(results)
    os.makedirs("outputs/tables", exist_ok=True)
    df_res.to_csv("outputs/tables/exp3_contamination_table.csv", index=False)
    print("Saved outputs/tables/exp3_contamination_table.csv")
    print(df_res)


def analyze_exp2():
    print("Analyzing EXP2...")
    try:
        # Condition A
        if os.path.exists("outputs/stream_logs/exp2a.stream.jsonl"):
            df_a = load_stream_log("outputs/stream_logs/exp2a.stream.jsonl")
            cost_a = df_a["cumulative_api_cost_usd"].iloc[-1] if not df_a.empty else 0
            solved_a = df_a["cumulative_solved"].iloc[-1] if not df_a.empty else 0
        else:
            cost_a, solved_a = 0, 0

        # Condition B
        if os.path.exists("outputs/stream_logs/exp2b_seed.stream.jsonl"):
            df_b_seed = load_stream_log("outputs/stream_logs/exp2b_seed.stream.jsonl")
            cost_b_seed = (
                df_b_seed["cumulative_api_cost_usd"].iloc[-1]
                if not df_b_seed.empty
                else 0
            )
            solved_b_seed = (
                df_b_seed["cumulative_solved"].iloc[-1] if not df_b_seed.empty else 0
            )
        else:
            cost_b_seed, solved_b_seed = 0, 0

        if os.path.exists("outputs/stream_logs/exp2b_online.stream.jsonl"):
            df_b_online = load_stream_log(
                "outputs/stream_logs/exp2b_online.stream.jsonl"
            )
            cost_b_online = (
                df_b_online["cumulative_api_cost_usd"].iloc[-1]
                if not df_b_online.empty
                else 0
            )
            solved_b_online = (
                df_b_online["cumulative_solved"].iloc[-1]
                if not df_b_online.empty
                else 0
            )
        else:
            cost_b_online, solved_b_online = 0, 0

        results = [
            {
                "condition": "A (No Seed)",
                "phase": "Online",
                "cost": cost_a,
                "solved": solved_a,
            },
            {
                "condition": "B (Warm Start)",
                "phase": "Seed",
                "cost": cost_b_seed,
                "solved": solved_b_seed,
            },
            {
                "condition": "B (Warm Start)",
                "phase": "Online",
                "cost": cost_b_online,
                "solved": solved_b_online,
            },
            {
                "condition": "B (Warm Start)",
                "phase": "Total",
                "cost": cost_b_seed + cost_b_online,
                "solved": solved_b_seed + solved_b_online,
            },
        ]

        df_res = pd.DataFrame(results)
        os.makedirs("outputs/tables", exist_ok=True)
        df_res.to_csv("outputs/tables/exp2_cost_summary.csv", index=False)
        print("Saved outputs/tables/exp2_cost_summary.csv")
        print(df_res)
    except Exception as e:
        print(f"Error analyzing EXP2: {e}")


if __name__ == "__main__":
    analyze_exp1()
    analyze_exp2()
    analyze_exp3()
