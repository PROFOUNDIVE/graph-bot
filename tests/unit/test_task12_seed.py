from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from graph_bot.interfaces import AbstractDistiller
from graph_bot.pipelines.stream_loop import run_continual_stream
from graph_bot.settings import settings
from graph_bot.utils.task12_seed import write_task12_seeded_metagraph


class _PassthroughDistiller(AbstractDistiller):
    def distill_query(self, query: str) -> str:
        return query

    def distill_trace(self, tree) -> list[Any]:
        del tree
        return []


def _run_seeded_stream(
    tmp_path: Path, *, use_edges: bool
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    problems_file = tmp_path / "task12_seeded_problem.jsonl"
    problems_file.write_text(
        json.dumps({"id": "q-seeded", "numbers": [1, 2, 3, 4], "target": 24}) + "\n",
        encoding="utf-8",
    )

    metrics_dir = tmp_path / ("metrics_edges" if use_edges else "metrics_no_edges")
    metagraph_path = tmp_path / (
        "seeded_edges.json" if use_edges else "seeded_no_edges.json"
    )
    write_task12_seeded_metagraph(metagraph_path)

    old_provider = settings.llm_provider
    old_model = settings.llm_model
    old_retrieval_backend = settings.retrieval_backend
    old_top_k_paths = settings.top_k_paths
    old_timeout = settings.execution_timeout_sec

    try:
        settings.llm_provider = "mock"
        settings.llm_model = "gpt-4o-mini"
        settings.retrieval_backend = "sparse_jaccard"
        settings.top_k_paths = 1
        settings.execution_timeout_sec = 1.0

        def mocked_chat(system, user, temperature):
            del system, user, temperature
            from graph_bot.adapters.mock_client import LLMUsage

            return "(1+2+3)*4", LLMUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                latency_ms=10.0,
            )

        with (
            patch(
                "graph_bot.adapters.graphrag.settings.metagraph_path", metagraph_path
            ),
            patch(
                "graph_bot.adapters.mock_client.MockLLMClient.chat",
                side_effect=mocked_chat,
            ),
        ):
            run_continual_stream(
                problems_file=problems_file,
                metrics_out_dir=metrics_dir,
                run_id=(
                    "test_task12_seeded_edges"
                    if use_edges
                    else "test_task12_seeded_no_edges"
                ),
                max_problems=1,
                use_edges=use_edges,
                mode="graph_bot",
                distiller=_PassthroughDistiller(),
            )
    finally:
        settings.llm_provider = old_provider
        settings.llm_model = old_model
        settings.retrieval_backend = old_retrieval_backend
        settings.top_k_paths = old_top_k_paths
        settings.execution_timeout_sec = old_timeout

    run_id = "test_task12_seeded_edges" if use_edges else "test_task12_seeded_no_edges"
    problems_path = metrics_dir / f"{run_id}.problems.jsonl"
    token_events_path = metrics_dir / f"{run_id}.token_events.jsonl"

    with problems_path.open("r", encoding="utf-8") as file_obj:
        problems = [
            cast(dict[str, Any], json.loads(line)) for line in file_obj if line.strip()
        ]
    with token_events_path.open("r", encoding="utf-8") as file_obj:
        token_events = [
            cast(dict[str, Any], json.loads(line)) for line in file_obj if line.strip()
        ]

    return problems, token_events


def test_write_task12_seeded_metagraph_supports_gate0_traversal_split(tmp_path: Path):
    edge_problems, edge_events = _run_seeded_stream(tmp_path, use_edges=True)
    no_edge_problems, no_edge_events = _run_seeded_stream(tmp_path, use_edges=False)

    edge_problem = edge_problems[0]
    no_edge_problem = no_edge_problems[0]

    assert edge_problem["retrieval_hit"] is True
    assert edge_problem["retrieval_path_node_cardinality"] == 2
    assert edge_problem["retrieval_used_stored_edges"] is True

    assert no_edge_problem["retrieval_hit"] is True
    assert no_edge_problem["retrieval_path_node_cardinality"] == 1
    assert no_edge_problem["retrieval_used_stored_edges"] is False

    edge_retrieval_event = next(
        event for event in edge_events if event["event_type"] == "rag_retrieval"
    )
    no_edge_retrieval_event = next(
        event for event in no_edge_events if event["event_type"] == "rag_retrieval"
    )

    edge_metadata = cast(dict[str, Any], edge_retrieval_event["metadata"])
    no_edge_metadata = cast(dict[str, Any], no_edge_retrieval_event["metadata"])

    assert edge_metadata["retrieval_path_node_cardinality"] == 2
    assert edge_metadata["retrieval_used_stored_edges"] is True
    assert no_edge_metadata["retrieval_path_node_cardinality"] == 1
    assert no_edge_metadata["retrieval_used_stored_edges"] is False
