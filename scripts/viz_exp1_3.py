from typing import cast

import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def load_stream_logs(pattern):
    # Absolute path pattern is expected
    files = glob.glob(pattern)
    dfs = []
    for f in files:
        data = []
        with open(f, "r") as fh:
            for line in fh:
                data.append(json.loads(line))
        if data:
            df = pd.DataFrame(data)
            df["run_source"] = os.path.basename(f)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def viz_exp1_amortization():
    df = load_stream_logs(
        "/home/hyunwoo/git/graph-bot/outputs/stream_logs/exp1_run*.stream.jsonl"
    )
    if df.empty:
        print("No EXP1 data found.")
        return

    plt.figure(figsize=(10, 6))

    run_sources = sorted(df["run_source"].unique())
    run_colors = ["#7FB3D5", "#5499C7", "#2E86C1"]
    avg_df = pd.DataFrame()

    for i, run_src in enumerate(run_sources):
        subset = (
            cast(pd.DataFrame, df.loc[df["run_source"] == run_src])
            .set_index("t")
            .sort_index()
            .reset_index()
        )
        color = run_colors[i % len(run_colors)]
        plt.plot(
            subset["t"],
            subset["cost_per_solved"],
            label=run_src,
            color=color,
            alpha=0.6,
        )

    if len(run_sources) > 1:
        avg_df = df.groupby("t")["cost_per_solved"].mean().reset_index()
        plt.plot(
            avg_df["t"],
            avg_df["cost_per_solved"],
            label="graph-bot: avg. of 3 runs",
            color="#1A5276",
            linewidth=2.5,
            linestyle="--",
        )

    # Baseline reference lines (cost per solved for fair comparison)
    # From outputs/week7_midweek_report.md (retry=3 to match EXP1 retry budget):
    # - IO (retry=3): accuracy=17.35%, cost/problem=$0.00009 -> cost/solved=$0.00052
    # - CoT (retry=3): accuracy=16.33%, cost/problem=$0.00026 -> cost/solved=$0.00159
    IO_COST_PER_SOLVED = 0.00009 / 0.1735
    COT_COST_PER_SOLVED = 0.00026 / 0.1633
    plt.axhline(
        y=IO_COST_PER_SOLVED,
        color="#E74C3C",
        linestyle=":",
        linewidth=2,
        label=f"IO (retry=3; 2 runs): ${IO_COST_PER_SOLVED:.5f} (avg.)",
    )
    plt.axhline(
        y=COT_COST_PER_SOLVED,
        color="#008000",
        linestyle=":",
        linewidth=2,
        label=f"CoT (retry=3; 2 runs): ${COT_COST_PER_SOLVED:.5f} (avg.)",
    )

    if len(run_sources) > 1:
        crossings = []
        for i in range(len(avg_df) - 1):
            c1 = avg_df.iloc[i]["cost_per_solved"]
            c2 = avg_df.iloc[i + 1]["cost_per_solved"]

            if (c1 >= COT_COST_PER_SOLVED > c2) or (c1 <= COT_COST_PER_SOLVED < c2):
                t1 = avg_df.iloc[i]["t"]
                t2 = avg_df.iloc[i + 1]["t"]
                alpha = (COT_COST_PER_SOLVED - c1) / (c2 - c1)
                cross_t = t1 + alpha * (t2 - t1)
                crossings.append(cross_t)

        if crossings:
            target_t = crossings[1] if len(crossings) >= 2 else crossings[0]

            plt.plot(target_t, COT_COST_PER_SOLVED, "ko", zorder=5)
            plt.annotate(
                "break-even point",
                xy=(target_t, COT_COST_PER_SOLVED),
                xytext=(target_t + 15, COT_COST_PER_SOLVED + 0.0006),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9
                ),
            )

    plt.xlabel("Problem Index (t)")
    plt.ylabel("Cost per Solved ($)")
    plt.title("EXP1: Amortization Curve (Cost per Solved)")
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/hyunwoo/git/graph-bot/outputs/figures/exp1_amortization_viz.png")
    print("Saved /home/hyunwoo/git/graph-bot/outputs/figures/exp1_amortization_viz.png")


def viz_exp3_contamination():
    df_noval = load_stream_logs(
        "/home/hyunwoo/git/graph-bot/outputs/stream_logs/exp3_noval_run*.stream.jsonl"
    )
    df_val = load_stream_logs(
        "/home/hyunwoo/git/graph-bot/outputs/stream_logs/exp1_run*.stream.jsonl"
    )

    if df_noval.empty:
        print("No EXP3 data found.")
        return

    plt.figure(figsize=(10, 6))

    noval_sources = sorted(df_noval["run_source"].unique())
    noval_colors = ["#F1948A", "#E74C3C", "#B03A2E"]

    for i, run_src in enumerate(noval_sources):
        subset = (
            cast(pd.DataFrame, df_noval.loc[df_noval["run_source"] == run_src])
            .set_index("t")
            .sort_index()
            .reset_index()
        )
        color = noval_colors[i % len(noval_colors)]
        plt.plot(
            subset["t"],
            subset["contamination_rate"],
            label=f"No-Val: {run_src}",
            linestyle="-",
            color=color,
            alpha=0.6,
        )

    avg_noval = df_noval.groupby("t")["contamination_rate"].mean().reset_index()
    plt.plot(
        avg_noval["t"],
        avg_noval["contamination_rate"],
        label="No-Val (Avg)",
        color="#922B21",
        linewidth=2.5,
        linestyle="--",
    )

    if not df_val.empty:
        df_val["contamination_rate"] = df_val["contamination_rate"].fillna(0)

        val_sources = sorted(df_val["run_source"].unique())
        val_colors = ["#85C1E9", "#5DADE2", "#2E86C1"]

        for i, run_src in enumerate(val_sources):
            subset = (
                cast(pd.DataFrame, df_val.loc[df_val["run_source"] == run_src])
                .set_index("t")
                .sort_index()
                .reset_index()
            )
            color = val_colors[i % len(val_colors)]
            plt.plot(
                subset["t"],
                subset["contamination_rate"],
                label=f"With-Val: {run_src}",
                linestyle="-",
                color=color,
                alpha=0.6,
            )

        avg_val = df_val.groupby("t")["contamination_rate"].mean().reset_index()
        plt.plot(
            avg_val["t"],
            avg_val["contamination_rate"],
            label="With Validator (Avg)",
            color="#1A5276",
            linewidth=2.5,
            linestyle="--",
        )

    plt.xlabel("Time (t)")
    plt.ylabel("Contamination Rate")
    plt.title("EXP3: Memory Contamination")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        "/home/hyunwoo/git/graph-bot/outputs/figures/exp3_contamination_viz.png"
    )
    print(
        "Saved /home/hyunwoo/git/graph-bot/outputs/figures/exp3_contamination_viz.png"
    )


if __name__ == "__main__":
    os.makedirs("/home/hyunwoo/git/graph-bot/outputs/figures", exist_ok=True)
    viz_exp1_amortization()
    viz_exp3_contamination()
