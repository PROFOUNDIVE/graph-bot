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
    for run_src in df["run_source"].unique():
        subset = (
            cast(pd.DataFrame, df.loc[df["run_source"] == run_src])
            .set_index("t")
            .sort_index()
            .reset_index()
        )
        plt.plot(subset["t"], subset["cost_per_solved"], label=run_src, alpha=0.5)

    if len(df["run_source"].unique()) > 1:
        avg_df = df.groupby("t")["cost_per_solved"].mean().reset_index()
        plt.plot(
            avg_df["t"],
            avg_df["cost_per_solved"],
            label="Average",
            color="black",
            linewidth=2,
            linestyle="--",
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
    for run_src in df_noval["run_source"].unique():
        subset = (
            cast(pd.DataFrame, df_noval.loc[df_noval["run_source"] == run_src])
            .set_index("t")
            .sort_index()
            .reset_index()
        )
        plt.plot(
            subset["t"],
            subset["contamination_rate"],
            label=f"No-Val: {run_src}",
            linestyle="--",
            color="red",
            alpha=0.5,
        )

    avg_noval = df_noval.groupby("t")["contamination_rate"].mean().reset_index()
    plt.plot(
        avg_noval["t"],
        avg_noval["contamination_rate"],
        label="No-Val (Avg)",
        color="red",
        linewidth=2,
    )

    if not df_val.empty:
        df_val["contamination_rate"] = df_val["contamination_rate"].fillna(0)
        avg_val = df_val.groupby("t")["contamination_rate"].mean().reset_index()
        plt.plot(
            avg_val["t"],
            avg_val["contamination_rate"],
            label="With Validator (Avg)",
            color="blue",
            linewidth=2,
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
