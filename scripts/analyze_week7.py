from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_base_dir(base_dir: Path | None) -> Path:
    return base_dir.resolve() if base_dir is not None else ROOT_DIR


def load_stream_log_with_longest(path: Path) -> pd.DataFrame:
    df = load_stream_log(path)
    if df.empty:
        return df
    if "t" in df.columns:
        reset_indices = df.index[df["t"] == 1].tolist()
        if not reset_indices:
            return df
        reset_indices.append(len(df))
        longest_df = pd.DataFrame()
        for i in range(len(reset_indices) - 1):
            start = reset_indices[i]
            end = reset_indices[i + 1]
            sub_df = df.iloc[start:end]
            if len(sub_df) > len(longest_df):
                longest_df = sub_df
        return longest_df
    return df


def load_stream_log(path: Path) -> pd.DataFrame:
    data: list[dict] = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)


def load_stream_logs(pattern: str) -> pd.DataFrame:
    pattern_path = Path(pattern)
    base_dir = pattern_path.parent
    files = sorted(base_dir.glob(pattern_path.name))
    dfs: list[pd.DataFrame] = []
    for file_path in files:
        df = load_stream_log(file_path)
        if not df.empty:
            df["run_source"] = file_path.name
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def plot_exp1_amortization(base_dir: Path | None = None) -> None:
    base = resolve_base_dir(base_dir)
    df = load_stream_logs(
        str(base / "outputs" / "stream_logs" / "exp1_run*.stream.jsonl")
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
    (base / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    output_path = base / "outputs" / "figures" / "exp1_amortization_viz.png"
    plt.savefig(output_path)
    print(f"Saved {output_path}")


def plot_exp3_contamination(base_dir: Path | None = None) -> None:
    base = resolve_base_dir(base_dir)
    df_noval = load_stream_logs(
        str(base / "outputs" / "stream_logs" / "exp3_noval_run*.stream.jsonl")
    )
    df_val = load_stream_logs(
        str(base / "outputs" / "stream_logs" / "exp1_run*.stream.jsonl")
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
    (base / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    output_path = base / "outputs" / "figures" / "exp3_contamination_viz.png"
    plt.savefig(output_path)
    print(f"Saved {output_path}")


def plot_exp4_memory_growth(base_dir: Path | None = None) -> None:
    base = resolve_base_dir(base_dir)
    problems_path = base / "outputs" / "stream_logs" / "exp4_dryrun.problems.jsonl"
    stream_path = base / "outputs" / "stream_logs" / "exp4_dryrun.stream.jsonl"

    if not problems_path.exists() and not stream_path.exists():
        print("No EXP4 data found.")
        return

    # Prefer per-problem metrics: memory_n_nodes/memory_n_edges are emitted in problems.jsonl.
    df = load_stream_log(problems_path) if problems_path.exists() else pd.DataFrame()
    x_col = "t"
    nodes_col = "memory_n_nodes"
    edges_col = "memory_n_edges"

    if df.empty or not {x_col, nodes_col, edges_col}.issubset(df.columns):
        # Fallback: some runs may only log metagraph growth in stream.jsonl.
        df = load_stream_log(stream_path) if stream_path.exists() else pd.DataFrame()
        nodes_col = "metagraph_nodes"
        edges_col = "metagraph_edges"
        if df.empty or not {x_col, nodes_col, edges_col}.issubset(df.columns):
            print("EXP4 log missing required columns.")
            return

    df_sorted = df.sort_values(x_col)

    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_left.plot(
        df_sorted[x_col],
        df_sorted[nodes_col],
        color="blue",
        label="Metagraph Nodes",
    )
    ax_left.set_xlabel("Time (t)")
    ax_left.set_ylabel("Metagraph Nodes")

    ax_right = ax_left.twinx()
    ax_right.plot(
        df_sorted[x_col],
        df_sorted[edges_col],
        color="green",
        label="Metagraph Edges",
    )
    ax_right.set_ylabel("Metagraph Edges")

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        lines_left + lines_right, labels_left + labels_right, loc="upper left"
    )
    ax_left.grid(True)
    fig.suptitle("EXP4: Memory Growth (Nodes/Edges)")

    (base / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    output_path = base / "outputs" / "figures" / "exp4_memory_growth.png"
    output_path_viz = base / "outputs" / "figures" / "exp4_memory_growth_viz.png"
    plt.savefig(output_path)
    print(f"Saved {output_path}")
    plt.savefig(output_path_viz)
    print(f"Saved {output_path_viz}")


def report_exp2_formal_logs(base_dir: Path | None = None) -> None:
    base = resolve_base_dir(base_dir)
    pattern = base / "outputs" / "stream_logs" / "exp2_formal_*.stream.jsonl"
    files = sorted(pattern.parent.glob(pattern.name))
    if not files:
        print("No EXP2 formal logs found.")
        return
    print(f"Found {len(files)} EXP2 formal log(s).")


def plot_exp2_efficiency(base_dir: Path | None = None) -> None:
    base = resolve_base_dir(base_dir)
    cold_path = base / "outputs" / "stream_logs" / "exp2_formal_cold.stream.jsonl"
    seed_path = base / "outputs" / "stream_logs" / "exp2_formal_warm_seed.stream.jsonl"
    online_path = (
        base / "outputs" / "stream_logs" / "exp2_formal_warm_online.stream.jsonl"
    )

    df_cold = (
        load_stream_log_with_longest(cold_path)
        if cold_path.exists()
        else pd.DataFrame()
    )
    df_seed = (
        load_stream_log_with_longest(seed_path)
        if seed_path.exists()
        else pd.DataFrame()
    )
    df_online = (
        load_stream_log_with_longest(online_path)
        if online_path.exists()
        else pd.DataFrame()
    )

    plotted = 0
    required = {"cumulative_api_cost_usd", "cumulative_solved"}

    plt.figure(figsize=(10, 6))

    # Cold
    if not df_cold.empty:
        if required.issubset(df_cold.columns):
            x = df_cold["cumulative_api_cost_usd"].astype(float)
            y = df_cold["cumulative_solved"].astype(float)
            plt.plot(x, y, label="EXP2 Cold (Formal)", alpha=0.8)
            plotted += 1
        else:
            print(
                "EXP2 Cold (Formal): missing required columns: {}".format(
                    sorted(list(required))
                )
            )

    # Seed
    if not df_seed.empty:
        if required.issubset(df_seed.columns):
            x = df_seed["cumulative_api_cost_usd"].astype(float)
            y = df_seed["cumulative_solved"].astype(float)
            plt.plot(x, y, label="EXP2 Warm Seed", alpha=0.8)
            plotted += 1
        else:
            print(
                "EXP2 Warm Seed: missing required columns: {}".format(
                    sorted(list(required))
                )
            )

    # Online
    if not df_online.empty:
        if required.issubset(df_online.columns):
            seed_cost = (
                float(df_seed["cumulative_api_cost_usd"].iloc[-1])
                if not df_seed.empty
                else 0.0
            )
            seed_solved = (
                float(df_seed["cumulative_solved"].iloc[-1])
                if not df_seed.empty
                else 0.0
            )
            x = df_online["cumulative_api_cost_usd"].astype(float) + seed_cost
            y = df_online["cumulative_solved"].astype(float) + seed_solved
            plt.plot(x, y, label="EXP2 Warm Online (Offset)", alpha=0.8)
            plotted += 1
        else:
            print(
                "EXP2 Warm Online: missing required columns: {}".format(
                    sorted(list(required))
                )
            )

    if plotted == 0:
        print("EXP2: All EXP2 series skipped due to missing required columns.")
        return

    plt.xlabel("Cumulative API Cost (USD)")
    plt.ylabel("Cumulative Solved")
    plt.title("EXP2: Efficiency across configurations")
    plt.legend()
    plt.grid(True)
    (base / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    output_path = base / "outputs" / "figures" / "exp2_efficiency_viz.png"
    plt.savefig(output_path)
    print(f"Saved {output_path}")


def plot_exp2_cost_per_solved(base_dir: Path | None = None) -> None:
    base = resolve_base_dir(base_dir)
    cold_path = base / "outputs" / "stream_logs" / "exp2_formal_cold.stream.jsonl"
    seed_path = base / "outputs" / "stream_logs" / "exp2_formal_warm_seed.stream.jsonl"
    online_path = (
        base / "outputs" / "stream_logs" / "exp2_formal_warm_online.stream.jsonl"
    )

    df_cold = (
        load_stream_log_with_longest(cold_path)
        if cold_path.exists()
        else pd.DataFrame()
    )
    df_seed = (
        load_stream_log_with_longest(seed_path)
        if seed_path.exists()
        else pd.DataFrame()
    )
    df_online = (
        load_stream_log_with_longest(online_path)
        if online_path.exists()
        else pd.DataFrame()
    )

    plotted_any = 0
    plt.figure(figsize=(10, 6))
    required = {"t", "cumulative_api_cost_usd", "cumulative_solved"}

    # Cold
    if not df_cold.empty:
        if required.issubset(df_cold.columns):
            t = df_cold["t"].astype(float)
            cost = df_cold["cumulative_api_cost_usd"].astype(float)
            solved = df_cold["cumulative_solved"].astype(float).clip(lower=1.0)
            cps = cost / solved
            plt.plot(t, cps, label="EXP2 Cold (Formal)", alpha=0.8)
            plotted_any += 1
        else:
            print(
                "EXP2 Cold (Formal): missing required columns: {cols}".format(
                    cols=sorted(required)
                )
            )

    # Seed
    if not df_seed.empty:
        if required.issubset(df_seed.columns):
            t = df_seed["t"].astype(float)
            cost = df_seed["cumulative_api_cost_usd"].astype(float)
            solved = df_seed["cumulative_solved"].astype(float).clip(lower=1.0)
            cps = cost / solved
            plt.plot(t, cps, label="EXP2 Warm Seed", alpha=0.8)
            plotted_any += 1
        else:
            print(
                "EXP2 Warm Seed: missing required columns: {cols}".format(
                    cols=sorted(required)
                )
            )

    # Online
    if not df_online.empty:
        if required.issubset(df_online.columns):
            seed_cost = (
                float(df_seed["cumulative_api_cost_usd"].iloc[-1])
                if not df_seed.empty
                else 0.0
            )
            seed_solved = (
                float(df_seed["cumulative_solved"].iloc[-1])
                if not df_seed.empty
                else 0.0
            )
            t = df_online["t"].astype(float)
            denom = df_online["cumulative_solved"].astype(float) + seed_solved
            denom = denom.clip(lower=1.0)
            cps = (
                df_online["cumulative_api_cost_usd"].astype(float) + seed_cost
            ) / denom
            plt.plot(t, cps, label="EXP2 Warm Online (Offset)", alpha=0.8)
            plotted_any += 1
        else:
            print(
                "EXP2 Warm Online: missing required columns: {cols}".format(
                    cols=sorted(required)
                )
            )

    if plotted_any == 0:
        print("EXP2: All EXP2 series skipped due to missing required columns.")
        return

    plt.xlabel("Time (t)")
    plt.ylabel("Cost per Solved ($)")
    plt.title("EXP2: Cost per Solved")
    plt.legend()
    plt.grid(True)
    (base / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    output_path = base / "outputs" / "figures" / "exp2_cost_per_solved_viz.png"
    plt.savefig(output_path)
    print(f"Saved {output_path}")


def analyze_exp2_formal_cost_summary(base_dir: Path | None = None) -> None:
    base = resolve_base_dir(base_dir)
    cold_path = base / "outputs" / "stream_logs" / "exp2_formal_cold.stream.jsonl"
    seed_path = base / "outputs" / "stream_logs" / "exp2_formal_warm_seed.stream.jsonl"
    online_path = (
        base / "outputs" / "stream_logs" / "exp2_formal_warm_online.stream.jsonl"
    )

    df_cold = load_stream_log(cold_path) if cold_path.exists() else pd.DataFrame()
    df_seed = load_stream_log(seed_path) if seed_path.exists() else pd.DataFrame()
    df_online = load_stream_log(online_path) if online_path.exists() else pd.DataFrame()

    cold_cost = (
        float(df_cold["cumulative_api_cost_usd"].iloc[-1])
        if not df_cold.empty and "cumulative_api_cost_usd" in df_cold.columns
        else 0.0
    )
    cold_solved = (
        float(df_cold["cumulative_solved"].iloc[-1])
        if not df_cold.empty and "cumulative_solved" in df_cold.columns
        else 0.0
    )

    seed_cost = (
        float(df_seed["cumulative_api_cost_usd"].iloc[-1])
        if not df_seed.empty and "cumulative_api_cost_usd" in df_seed.columns
        else 0.0
    )
    seed_solved = (
        float(df_seed["cumulative_solved"].iloc[-1])
        if not df_seed.empty and "cumulative_solved" in df_seed.columns
        else 0.0
    )

    online_cost = (
        float(df_online["cumulative_api_cost_usd"].iloc[-1])
        if not df_online.empty and "cumulative_api_cost_usd" in df_online.columns
        else 0.0
    )
    online_solved = (
        float(df_online["cumulative_solved"].iloc[-1])
        if not df_online.empty and "cumulative_solved" in df_online.columns
        else 0.0
    )

    results = [
        {
            "condition": "Cold (Online)",
            "phase": "Online",
            "cost": cold_cost,
            "solved": cold_solved,
        },
        {
            "condition": "Warm (Seed)",
            "phase": "Seed",
            "cost": seed_cost,
            "solved": seed_solved,
        },
        {
            "condition": "Warm (Online)",
            "phase": "Online",
            "cost": online_cost,
            "solved": online_solved,
        },
        {
            "condition": "Warm (Total)",
            "phase": "Total",
            "cost": seed_cost + online_cost,
            "solved": seed_solved + online_solved,
        },
    ]

    df_res = pd.DataFrame(results)
    (base / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
    output_csv = base / "outputs" / "tables" / "exp2_formal_cost_summary.csv"
    df_res.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")
    print(df_res)


def main() -> None:
    report_exp2_formal_logs()
    analyze_exp2_formal_cost_summary()
    plot_exp1_amortization()
    plot_exp3_contamination()
    plot_exp4_memory_growth()
    plot_exp2_efficiency()
    plot_exp2_cost_per_solved()
    # No runtime notepad logging per instructions


if __name__ == "__main__":
    main()
