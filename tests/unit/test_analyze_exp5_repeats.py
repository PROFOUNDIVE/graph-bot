from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _base_problem_row(*, t: int) -> dict[str, object]:
    return {
        "t": t,
        "problem_id": f"problem-{t}",
        "solved": True,
        "attempts": 1,
        "solved_attempt": 1,
        "attempt_success_rate": 1.0,
        "llm_calls": 1,
        "tokens_total": 100,
        "latency_total_ms": 10.0,
        "api_cost_usd": 0.01,
        "retrieval_hit": True,
        "reuse_count": 1,
        "memory_n_nodes": 4,
        "memory_n_edges": 0,
    }


def _base_retrieval_event(
    run_id: str, *, t: int, retrieval_metadata: dict[str, object]
) -> dict[str, object]:
    return {
        "timestamp": "2026-05-01T00:00:00Z",
        "stream_run_id": run_id,
        "problem_id": f"problem-{t}",
        "t": t,
        "event_type": "rag_retrieval",
        "operation": "retrieve",
        "status": "success",
        "model": "sparse_jaccard",
        "latency_ms": 0,
        "run_id": f"{run_id}:problem-{t}",
        "span_id": f"span-{run_id}-{t}",
        "component": "rag_infra",
        "metadata": {
            "pricing_version": "v0",
            "mode": "graph_bot",
            "retrieval_backend": "sparse_jaccard",
            "task": "game24",
            **retrieval_metadata,
        },
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "cost_usd": 0.0,
    }


def _base_memory_update_event(
    run_id: str, *, t: int, edges_added_count: int
) -> dict[str, object]:
    return {
        "timestamp": "2026-05-01T00:00:01Z",
        "stream_run_id": run_id,
        "problem_id": f"problem-{t}",
        "t": t,
        "event_type": "memory_update",
        "operation": "insert_trees",
        "status": "success",
        "model": "cpu",
        "latency_ms": 0,
        "run_id": f"{run_id}:problem-{t}",
        "span_id": f"memory-{run_id}-{t}",
        "component": "metagraph",
        "metadata": {"edges_added_count": edges_added_count},
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "cost_usd": 0.0,
    }


def _write_fixture_run(
    log_dir: Path,
    *,
    base_name: str,
    rep: str,
    problem_rows: list[dict[str, object]],
    token_events: list[dict[str, object]],
) -> str:
    run_id = f"{base_name}_rep{rep}"
    _write_jsonl(log_dir / f"{run_id}.problems.jsonl", problem_rows)
    _write_jsonl(log_dir / f"{run_id}.token_events.jsonl", token_events)
    return run_id


def _run_analyzer(
    tmp_path: Path,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "analyze_exp5_repeats.py"
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "run_manifest.jsonl"
    out_md = tmp_path / "report.md"
    out_csv = tmp_path / "report.csv"

    return (
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--log-dir",
                str(log_dir),
                "--out-md",
                str(out_md),
                "--out-csv",
                str(out_csv),
                "--manifest-path",
                str(manifest_path),
                "--bootstrap-resamples",
                "10",
                "--seed",
                "1",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        ),
        out_md,
        out_csv,
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_diagnostic_row(
    rows: list[dict[str, str]], *, row_type: str, diagnostic_key: str
) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row["row_type"] == row_type and row["diagnostic_key"] == diagnostic_key
    )


def test_analyzer_downgrades_graph_gate_when_traversal_telemetry_is_absent(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    edge_problem = _base_problem_row(t=1)
    edge_problem["retrieval_used_stored_edges"] = False
    edge_problem["retrieval_path_node_cardinality"] = 1
    edge_problem["memory_n_edges"] = 7
    no_edge_problem = _base_problem_row(t=1)
    no_edge_problem["retrieval_used_stored_edges"] = False
    no_edge_problem["retrieval_path_node_cardinality"] = 1
    no_edge_problem["memory_n_edges"] = 5

    edge_run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_edges",
        rep="01",
        problem_rows=[edge_problem],
        token_events=[
            _base_retrieval_event(
                "exp5_graph_bot_edges_rep01",
                t=1,
                retrieval_metadata={
                    "retrieval_used_stored_edges": False,
                    "retrieval_path_node_cardinality": 1,
                },
            ),
            _base_memory_update_event(
                "exp5_graph_bot_edges_rep01", t=1, edges_added_count=2
            ),
        ],
    )
    no_edge_run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_no_edges",
        rep="01",
        problem_rows=[no_edge_problem],
        token_events=[
            _base_retrieval_event(
                "exp5_graph_bot_no_edges_rep01",
                t=1,
                retrieval_metadata={
                    "retrieval_used_stored_edges": False,
                    "retrieval_path_node_cardinality": 1,
                },
            ),
            _base_memory_update_event(
                "exp5_graph_bot_no_edges_rep01", t=1, edges_added_count=3
            ),
        ],
    )

    manifest_rows = [
        {"run_id": edge_run_id, "config": {"mode": "graph_bot"}},
        {"run_id": no_edge_run_id, "config": {"mode": "graph_bot"}},
    ]
    _write_jsonl(tmp_path / "run_manifest.jsonl", manifest_rows)

    result, out_md, out_csv = _run_analyzer(tmp_path)

    assert result.returncode == 0, result.stderr
    rows = _read_csv_rows(out_csv)
    graph_gate_row = _find_diagnostic_row(
        rows,
        row_type="claim_gate",
        diagnostic_key="graph_structure_used",
    )
    edge_growth_row = _find_diagnostic_row(
        rows,
        row_type="graph_diagnostic_aggregate",
        diagnostic_key="edge_enabled_persisted_edge_growth_any",
    )
    no_edge_growth_row = _find_diagnostic_row(
        rows,
        row_type="graph_diagnostic_aggregate",
        diagnostic_key="no_edge_persisted_edge_growth_any",
    )

    assert graph_gate_row["diagnostic_status"] == "FAIL"
    assert graph_gate_row["value_text"] == "no_traversal_evidence"
    assert edge_growth_row["value_text"] == "True"
    assert no_edge_growth_row["value_text"] == "True"

    markdown = out_md.read_text(encoding="utf-8")
    assert "Graph-structure-used gate=FAIL source=no_traversal_evidence" in markdown
    assert (
        "Persisted edge growth is reported separately from traversal evidence."
        in markdown
    )


def test_analyzer_passes_gate0_when_edge_enabled_has_exclusive_traversal(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    edge_problem = _base_problem_row(t=1)
    edge_problem["retrieval_used_stored_edges"] = True
    edge_problem["retrieval_path_node_cardinality"] = 2
    edge_problem["memory_n_edges"] = 1

    no_edge_problem = _base_problem_row(t=1)
    no_edge_problem["retrieval_used_stored_edges"] = False
    no_edge_problem["retrieval_path_node_cardinality"] = 1
    no_edge_problem["memory_n_edges"] = 0

    edge_run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_edges",
        rep="01",
        problem_rows=[edge_problem],
        token_events=[
            _base_retrieval_event(
                "exp5_graph_bot_edges_rep01",
                t=1,
                retrieval_metadata={
                    "retrieval_used_stored_edges": True,
                    "retrieval_path_node_cardinality": 2,
                },
            ),
            _base_memory_update_event(
                "exp5_graph_bot_edges_rep01", t=1, edges_added_count=1
            ),
        ],
    )
    no_edge_run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_no_edges",
        rep="01",
        problem_rows=[no_edge_problem],
        token_events=[
            _base_retrieval_event(
                "exp5_graph_bot_no_edges_rep01",
                t=1,
                retrieval_metadata={
                    "retrieval_used_stored_edges": False,
                    "retrieval_path_node_cardinality": 1,
                },
            )
        ],
    )

    manifest_rows = [
        {"run_id": edge_run_id, "config": {"mode": "graph_bot"}},
        {"run_id": no_edge_run_id, "config": {"mode": "graph_bot"}},
    ]
    _write_jsonl(tmp_path / "run_manifest.jsonl", manifest_rows)

    result, out_md, out_csv = _run_analyzer(tmp_path)

    assert result.returncode == 0, result.stderr
    rows = _read_csv_rows(out_csv)
    graph_gate_row = _find_diagnostic_row(
        rows,
        row_type="claim_gate",
        diagnostic_key="graph_structure_used",
    )
    edge_growth_row = _find_diagnostic_row(
        rows,
        row_type="graph_diagnostic_aggregate",
        diagnostic_key="edge_enabled_persisted_edge_growth_any",
    )
    no_edge_growth_row = _find_diagnostic_row(
        rows,
        row_type="graph_diagnostic_aggregate",
        diagnostic_key="no_edge_persisted_edge_growth_any",
    )

    assert graph_gate_row["diagnostic_status"] == "PASS"
    assert graph_gate_row["value_text"] == "traversal_confirmed"
    assert edge_growth_row["value_text"] == "True"
    assert no_edge_growth_row["value_text"] == "False"

    markdown = out_md.read_text(encoding="utf-8")
    assert "Graph-structure-used gate=PASS source=traversal_confirmed" in markdown


def test_analyzer_fails_gate0_when_no_edge_reports_stored_edge_traversal(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    edge_problem = _base_problem_row(t=1)
    edge_problem["retrieval_used_stored_edges"] = True
    edge_problem["retrieval_path_node_cardinality"] = 2
    edge_problem["memory_n_edges"] = 1

    no_edge_problem = _base_problem_row(t=1)
    no_edge_problem["retrieval_used_stored_edges"] = True
    no_edge_problem["retrieval_path_node_cardinality"] = 2
    no_edge_problem["memory_n_edges"] = 0

    no_edge_event = _base_retrieval_event(
        "exp5_graph_bot_no_edges_rep01",
        t=1,
        retrieval_metadata={
            "retrieval_used_stored_edges": True,
            "retrieval_path_node_cardinality": 2,
        },
    )

    edge_run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_edges",
        rep="01",
        problem_rows=[edge_problem],
        token_events=[
            _base_retrieval_event(
                "exp5_graph_bot_edges_rep01",
                t=1,
                retrieval_metadata={
                    "retrieval_used_stored_edges": True,
                    "retrieval_path_node_cardinality": 2,
                },
            )
        ],
    )
    no_edge_run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_no_edges",
        rep="01",
        problem_rows=[no_edge_problem],
        token_events=[no_edge_event],
    )

    manifest_rows = [
        {"run_id": edge_run_id, "config": {"mode": "graph_bot"}},
        {"run_id": no_edge_run_id, "config": {"mode": "graph_bot"}},
    ]
    _write_jsonl(tmp_path / "run_manifest.jsonl", manifest_rows)

    result, out_md, out_csv = _run_analyzer(tmp_path)

    assert result.returncode == 0, result.stderr
    rows = _read_csv_rows(out_csv)
    graph_gate_row = _find_diagnostic_row(
        rows,
        row_type="claim_gate",
        diagnostic_key="graph_structure_used",
    )

    assert graph_gate_row["diagnostic_status"] == "FAIL"
    assert graph_gate_row["value_text"] == "shared_traversal_evidence"

    markdown = out_md.read_text(encoding="utf-8")
    assert "Graph-structure-used gate=FAIL source=shared_traversal_evidence" in markdown


def test_analyzer_marks_legacy_only_graph_telemetry_as_ambiguous(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    edge_problem = _base_problem_row(t=1)
    edge_problem["reuse_count"] = 4
    edge_problem["memory_n_edges"] = 8
    no_edge_problem = _base_problem_row(t=1)
    no_edge_problem["reuse_count"] = 3
    no_edge_problem["memory_n_edges"] = 6

    edge_run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_edges",
        rep="01",
        problem_rows=[edge_problem],
        token_events=[
            _base_retrieval_event(
                "exp5_graph_bot_edges_rep01",
                t=1,
                retrieval_metadata={"packed_context_tokens": 44},
            ),
            _base_memory_update_event(
                "exp5_graph_bot_edges_rep01", t=1, edges_added_count=2
            ),
        ],
    )
    no_edge_run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_no_edges",
        rep="01",
        problem_rows=[no_edge_problem],
        token_events=[
            _base_retrieval_event(
                "exp5_graph_bot_no_edges_rep01",
                t=1,
                retrieval_metadata={"packed_context_tokens": 41},
            ),
            _base_memory_update_event(
                "exp5_graph_bot_no_edges_rep01", t=1, edges_added_count=1
            ),
        ],
    )

    manifest_rows = [
        {"run_id": edge_run_id, "config": {"mode": "graph_bot"}},
        {"run_id": no_edge_run_id, "config": {"mode": "graph_bot"}},
    ]
    _write_jsonl(tmp_path / "run_manifest.jsonl", manifest_rows)

    result, out_md, out_csv = _run_analyzer(tmp_path)

    assert result.returncode == 0, result.stderr
    rows = _read_csv_rows(out_csv)
    graph_gate_row = _find_diagnostic_row(
        rows,
        row_type="claim_gate",
        diagnostic_key="graph_structure_used",
    )

    assert graph_gate_row["diagnostic_status"] == "FAIL"
    assert graph_gate_row["value_text"] == "telemetry_ambiguous"

    markdown = out_md.read_text(encoding="utf-8")
    assert "Graph-structure-used gate=FAIL source=telemetry_ambiguous" in markdown


def test_analyzer_report_uses_conservative_week19_claim_wording(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    edge_problem = _base_problem_row(t=1)
    edge_problem["reuse_count"] = 4
    edge_problem["memory_n_edges"] = 8
    no_edge_problem = _base_problem_row(t=1)
    no_edge_problem["reuse_count"] = 3
    no_edge_problem["memory_n_edges"] = 6

    edge_run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_edges",
        rep="01",
        problem_rows=[edge_problem],
        token_events=[
            _base_retrieval_event(
                "exp5_graph_bot_edges_rep01",
                t=1,
                retrieval_metadata={"packed_context_tokens": 44},
            ),
            _base_memory_update_event(
                "exp5_graph_bot_edges_rep01", t=1, edges_added_count=2
            ),
        ],
    )
    no_edge_run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_no_edges",
        rep="01",
        problem_rows=[no_edge_problem],
        token_events=[
            _base_retrieval_event(
                "exp5_graph_bot_no_edges_rep01",
                t=1,
                retrieval_metadata={"packed_context_tokens": 41},
            ),
            _base_memory_update_event(
                "exp5_graph_bot_no_edges_rep01", t=1, edges_added_count=1
            ),
        ],
    )

    manifest_rows = [
        {"run_id": edge_run_id, "config": {"mode": "graph_bot"}},
        {"run_id": no_edge_run_id, "config": {"mode": "graph_bot"}},
    ]
    _write_jsonl(tmp_path / "run_manifest.jsonl", manifest_rows)

    result, out_md, out_csv = _run_analyzer(tmp_path)

    assert result.returncode == 0, result.stderr
    markdown = out_md.read_text(encoding="utf-8")
    combined_outputs = markdown + "\n" + out_csv.read_text(encoding="utf-8")

    assert (
        "Historical shared edge-count failures are treated as source-mixed / "
        "mismeasured, not as proof of no-edge traversal leakage." in markdown
    )
    assert "Graph structure effect remains unconfirmed." in markdown
    assert (
        "Execution-readiness must be interpreted separately from graph-structure "
        "and performance outcomes." in markdown
    )
    assert "failed because no-edge also had edges" not in combined_outputs
    assert "graph structure effect is confirmed" not in combined_outputs
    assert "graph performance advantage is confirmed" not in combined_outputs
    assert "execution benefit is confirmed" not in combined_outputs


def test_analyzer_report_keeps_conservative_week19_wording_for_no_traversal_path(
    tmp_path,
):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    edge_problem = _base_problem_row(t=1)
    edge_problem["retrieval_used_stored_edges"] = False
    edge_problem["retrieval_path_node_cardinality"] = 1
    edge_problem["memory_n_edges"] = 7
    no_edge_problem = _base_problem_row(t=1)
    no_edge_problem["retrieval_used_stored_edges"] = False
    no_edge_problem["retrieval_path_node_cardinality"] = 1
    no_edge_problem["memory_n_edges"] = 5

    edge_run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_edges",
        rep="01",
        problem_rows=[edge_problem],
        token_events=[
            _base_retrieval_event(
                "exp5_graph_bot_edges_rep01",
                t=1,
                retrieval_metadata={
                    "retrieval_used_stored_edges": False,
                    "retrieval_path_node_cardinality": 1,
                },
            ),
            _base_memory_update_event(
                "exp5_graph_bot_edges_rep01", t=1, edges_added_count=2
            ),
        ],
    )
    no_edge_run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_no_edges",
        rep="01",
        problem_rows=[no_edge_problem],
        token_events=[
            _base_retrieval_event(
                "exp5_graph_bot_no_edges_rep01",
                t=1,
                retrieval_metadata={
                    "retrieval_used_stored_edges": False,
                    "retrieval_path_node_cardinality": 1,
                },
            ),
            _base_memory_update_event(
                "exp5_graph_bot_no_edges_rep01", t=1, edges_added_count=3
            ),
        ],
    )

    manifest_rows = [
        {"run_id": edge_run_id, "config": {"mode": "graph_bot"}},
        {"run_id": no_edge_run_id, "config": {"mode": "graph_bot"}},
    ]
    _write_jsonl(tmp_path / "run_manifest.jsonl", manifest_rows)

    result, out_md, out_csv = _run_analyzer(tmp_path)

    assert result.returncode == 0, result.stderr
    markdown = out_md.read_text(encoding="utf-8")
    combined_outputs = markdown + "\n" + out_csv.read_text(encoding="utf-8")

    assert "Graph-structure-used gate=FAIL source=no_traversal_evidence" in markdown
    assert (
        "Historical shared edge-count failures are treated as source-mixed / "
        "mismeasured, not as proof of no-edge traversal leakage." in markdown
    )
    assert "Graph structure effect remains unconfirmed." in markdown
    assert (
        "Execution-readiness must be interpreted separately from graph-structure "
        "and performance outcomes." in markdown
    )
    assert "failed because no-edge also had edges" not in combined_outputs
    assert "graph structure effect is confirmed" not in combined_outputs
    assert "graph performance advantage is confirmed" not in combined_outputs


def test_analyzer_infers_validator_mode_from_validate_token_events(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = _write_fixture_run(
        log_dir,
        base_name="exp5_graph_bot_edges",
        rep="01",
        problem_rows=[_base_problem_row(t=1)],
        token_events=[
            _base_retrieval_event(
                "exp5_graph_bot_edges_rep01",
                t=1,
                retrieval_metadata={
                    "retrieval_used_stored_edges": False,
                    "retrieval_path_node_cardinality": 1,
                },
            ),
            {
                "timestamp": "2026-05-01T00:00:02Z",
                "stream_run_id": "exp5_graph_bot_edges_rep01",
                "problem_id": "problem-1",
                "t": 1,
                "event_type": "tool_call",
                "operation": "validate",
                "status": "success",
                "model": "oracle",
                "latency_ms": 0,
                "run_id": "exp5_graph_bot_edges_rep01:problem-1",
                "span_id": "validate-exp5_graph_bot_edges_rep01-1",
                "component": "evaluator",
                "metadata": {"pricing_version": "v0"},
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "cost_usd": 0.0,
            },
        ],
    )
    _write_jsonl(
        tmp_path / "run_manifest.jsonl",
        [{"run_id": run_id, "config": {"mode": "graph_bot"}}],
    )

    result, out_md, out_csv = _run_analyzer(tmp_path)

    assert result.returncode == 0, result.stderr
    run_rows = [row for row in _read_csv_rows(out_csv) if row["row_type"] == "run"]

    assert len(run_rows) == 1
    assert run_rows[0]["run_id"] == run_id
    assert run_rows[0]["validator_mode"] == "oracle"
    assert (
        "| exp5_graph_bot_edges | exp5_graph_bot_edges_rep01 | 01 | oracle |"
        in out_md.read_text(encoding="utf-8")
    )
