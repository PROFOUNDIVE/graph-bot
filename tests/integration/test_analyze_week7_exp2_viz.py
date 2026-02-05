import json
import os
import runpy
from pathlib import Path


def test_exp2_viz_with_synthetic_logs(tmp_path: Path):
    # Ensure matplotlib runs headless
    os.environ["MPLBACKEND"] = "Agg"

    # Create synthetic logs directory structure expected by the script
    logs_dir = tmp_path / "outputs" / "stream_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Minimal JSONL records required by the visualization plotting
    sample_lines = [
        {"t": 0, "cumulative_api_cost_usd": 0.0, "cumulative_solved": 0},
        {"t": 1, "cumulative_api_cost_usd": 0.5, "cumulative_solved": 1},
        {"t": 2, "cumulative_api_cost_usd": 1.0, "cumulative_solved": 1},
    ]

    (logs_dir / "exp2_formal_cold.stream.jsonl").write_text(
        "\n".join(json.dumps(x) for x in sample_lines),
        encoding="utf-8",
    )
    (logs_dir / "exp2_formal_warm_seed.stream.jsonl").write_text(
        "\n".join(json.dumps(x) for x in sample_lines),
        encoding="utf-8",
    )
    (logs_dir / "exp2_formal_warm_online.stream.jsonl").write_text(
        "\n".join(json.dumps(x) for x in sample_lines),
        encoding="utf-8",
    )

    script_path = Path("scripts/analyze_week7.py").resolve()
    assert script_path.exists(), f"Script not found: {script_path}"

    # Load script via run_path and ensure we do NOT run __main__
    scope = runpy.run_path(str(script_path), run_name="not_main")

    plot_eff = scope.get("plot_exp2_efficiency")
    plot_cost = scope.get("plot_exp2_cost_per_solved")

    assert callable(plot_eff) and callable(
        plot_cost
    ), "Expected plotting functions to be defined."

    # Execute plotting functions
    plot_eff(base_dir=tmp_path)
    plot_cost(base_dir=tmp_path)

    # Validate that the figures were produced
    eff_path = tmp_path / "outputs" / "figures" / "exp2_efficiency_viz.png"
    cost_path = tmp_path / "outputs" / "figures" / "exp2_cost_per_solved_viz.png"

    assert (
        eff_path.exists() and eff_path.stat().st_size > 0
    ), f"Missing or empty: {eff_path}"
    assert (
        cost_path.exists() and cost_path.stat().st_size > 0
    ), f"Missing or empty: {cost_path}"
