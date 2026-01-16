from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..adapters.graphrag import GraphRAGAdapter
from ..eval.validators import BaseValidator, get_validator
from ..pipelines.metrics_logger import StreamMetricsLogger
from ..types import PathEvaluation, StreamCallMetrics, StreamProblemMetrics, UserQuery


class Game24Problem(BaseModel):
    """Game of 24 problem format."""

    id: str
    numbers: List[float]
    target: float = Field(default=24.0)
    metadata: Dict[str, Any] | None = None

    def to_user_query(self) -> UserQuery:
        """Convert to UserQuery format."""
        numbers_str = " ".join(str(int(n)) for n in self.numbers)
        question = f"{numbers_str} → {int(self.target)}"
        metadata = self.metadata or {}
        metadata["task"] = "game24"
        return UserQuery(id=self.id, question=question, metadata=metadata)


def load_game24_problems(file_path: Path) -> List[Game24Problem]:
    """Load Game24 problems from JSONL file.

    Expected format: {"id": "q-001", "numbers": [2, 5, 8, 11], "target": 24}
    """
    problems: List[Game24Problem] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            problems.append(Game24Problem.model_validate(obj))
    return problems


def run_continual_stream(
    *,
    problems_file: Path,
    mode: str | None = None,
    use_edges: bool | None = None,
    policy_id: str | None = None,
    validator_mode: str = "oracle",
    max_problems: int | None = None,
    metrics_out_dir: Path = Path("outputs/stream_logs"),
    run_id: str = "run",
):
    """Run continual stream loop for Game of 24."""
    from ..settings import settings

    problems = load_game24_problems(problems_file)
    if max_problems:
        problems = problems[:max_problems]

    adapter = GraphRAGAdapter(
        mode=mode,
        use_edges=use_edges,
        policy_id=policy_id,
    )
    validator = get_validator(validator_mode)
    metrics = StreamMetricsLogger(out_dir=metrics_out_dir, run_id=run_id)

    results: list[dict[str, object]] = []

    for t, problem in enumerate(problems, 1):
        problem_id = problem.id
        query = problem.to_user_query()

        call_id_retrieve = metrics.new_call_id()
        metrics.log_call(
            StreamCallMetrics(
                call_id=call_id_retrieve,
                parent_id=None,
                t=t,
                problem_id=problem_id,
                operation="retrieve",
            )
        )

        active_mode = mode or settings.mode
        active_policy_id = policy_id or settings.policy_id

        adapter.mode = active_mode
        adapter.policy_id = active_policy_id

        retrieval = adapter.retrieve_paths(query, k=settings.top_k_paths)
        adapter.register_usage(retrieval.paths)

        reuse_count = sum(len(p.node_ids) for p in retrieval.paths)
        contaminated = _count_contaminated_templates(
            adapter.export_graph(), retrieval.paths
        )
        contamination_rate = None
        if reuse_count > 0:
            contamination_rate = contaminated / reuse_count

        answer_text, aggregate_usage, solved, solved_attempt, last_reason = (
            _solve_with_retries(
                query=query,
                retrieval=retrieval,
                validator=validator,
                metrics=metrics,
                parent_call_id=call_id_retrieve,
                t=t,
                problem_id=problem_id,
            )
        )

        solve_tokens_total = _maybe_int(aggregate_usage.get("total_tokens"))
        solve_latency_ms = _maybe_float(aggregate_usage.get("latency_ms"))

        evaluations = []
        for path in retrieval.paths:
            evaluations.append(
                PathEvaluation(
                    path_id=path.path_id,
                    node_ids=path.node_ids,
                    success=solved,
                    tokens=solve_tokens_total,
                    latency_ms=solve_latency_ms,
                    cost_usd=None,
                )
            )
        adapter.update_with_feedback(evaluations)

        if solved:
            _insert_solution_template(
                adapter=adapter,
                problem_id=problem_id,
                answer_text=answer_text,
            )

        memory = adapter.export_graph()

        tokens_total = solve_tokens_total or 0
        latency_total_ms = solve_latency_ms or 0.0
        api_cost_usd = 0.0
        if settings.llm_provider == "vllm" and settings.llm_token_cost_usd_per_1k > 0:
            api_cost_usd = (tokens_total / 1000.0) * settings.llm_token_cost_usd_per_1k

        problem_metrics = StreamProblemMetrics(
            t=t,
            problem_id=problem_id,
            solved=solved,
            attempts=settings.retry_max_attempts,
            solved_attempt=solved_attempt,
            attempt_success_rate=1.0 if solved else 0.0,
            llm_calls=settings.retry_max_attempts,
            tokens_total=tokens_total,
            latency_total_ms=latency_total_ms,
            api_cost_usd=api_cost_usd,
            retrieval_hit=len(retrieval.paths) > 0,
            reuse_count=reuse_count,
            memory_n_nodes=len(memory.nodes),
            memory_n_edges=len(memory.edges),
            contamination_rate=contamination_rate,
        )
        cumulative = metrics.log_problem(problem_metrics)

        results.append(
            {
                "t": t,
                "problem_id": problem_id,
                "solved": solved,
                "reuse_count": reuse_count,
                "contamination_rate": contamination_rate,
                "cumulative_cost": cumulative.cumulative_api_cost_usd,
                "cost_per_solved": cumulative.cost_per_solved,
            }
        )

    return results


def _maybe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _maybe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _count_contaminated_templates(graph, paths: list) -> int:
    node_index = {node.node_id: node for node in graph.nodes}

    contaminated = 0
    for path in paths:
        for node_id in path.node_ids:
            node = node_index.get(node_id)
            if not node or not node.attributes:
                continue
            quality = node.attributes.get("quality")
            if isinstance(quality, dict) and quality.get("validator_passed") is False:
                contaminated += 1

    return contaminated


def _insert_solution_template(
    *,
    adapter: GraphRAGAdapter,
    problem_id: str,
    answer_text: str,
) -> None:
    from ..types import ReasoningNode, ReasoningTree

    tree = ReasoningTree(
        tree_id=f"episode-{problem_id}",
        root_id=f"episode-{problem_id}-solution",
        nodes=[
            ReasoningNode(
                node_id=f"episode-{problem_id}-solution",
                text=answer_text,
                type="thought",
                attributes={"subtype": "template"},
            )
        ],
        edges=[],
        provenance={"task": "game24", "source": "stream"},
    )
    adapter.insert_trees([tree])


def _solve_with_retries(
    *,
    query: UserQuery,
    retrieval,
    validator: BaseValidator,
    metrics: StreamMetricsLogger,
    parent_call_id: str,
    t: int,
    problem_id: str,
) -> tuple[str, dict[str, object], bool, int | None, str | None]:
    from ..pipelines.main_loop import answer_with_retrieval
    from ..settings import settings

    temperatures = [
        settings.retry_temperature_1,
        settings.retry_temperature_2,
        settings.retry_temperature_3,
    ]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_latency_ms = 0.0

    last_reason: str | None = None
    last_answer_text = ""

    for attempt_index in range(1, settings.retry_max_attempts + 1):
        temperature = temperatures[min(attempt_index - 1, len(temperatures) - 1)]

        call_id_solve = metrics.new_call_id()
        answer_text = ""
        solve_usage: dict[str, object] = {}
        solve_error_type: str | None = None
        try:
            answer = answer_with_retrieval(
                query,
                retrieval=retrieval,
            )
            answer_text = answer.answer
            solve_usage = dict(answer.metadata or {})
        except Exception as exc:
            solve_error_type = type(exc).__name__

        prompt_tokens = _maybe_int(solve_usage.get("prompt_tokens"))
        completion_tokens = _maybe_int(solve_usage.get("completion_tokens"))
        attempt_total_tokens = _maybe_int(solve_usage.get("total_tokens"))
        latency_ms = _maybe_float(solve_usage.get("latency_ms"))

        total_prompt_tokens += prompt_tokens or 0
        total_completion_tokens += completion_tokens or 0
        total_tokens += attempt_total_tokens or 0
        total_latency_ms += latency_ms or 0.0

        attempt_passed = False
        reason = None
        if solve_error_type is None:
            attempt_passed = validator.validate(answer_text, query.question)
            if not attempt_passed:
                reason = validator.failure_reason(answer_text, query.question)
        else:
            reason = "solve_error"

        last_reason = reason

        metrics.log_call(
            StreamCallMetrics(
                call_id=call_id_solve,
                parent_id=parent_call_id,
                t=t,
                problem_id=problem_id,
                operation="solve",
                attempt_index=attempt_index,
                temperature=temperature,
                validator_passed=attempt_passed if solve_error_type is None else False,
                failure_reason=reason,
                prompt_variant="retry" if attempt_index > 1 else "base",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=attempt_total_tokens,
                latency_ms=latency_ms,
                error_type=solve_error_type,
            )
        )

        last_answer_text = answer_text

        if attempt_passed:
            return (
                answer_text,
                {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                    "latency_ms": total_latency_ms,
                },
                True,
                attempt_index,
                None,
            )

    return (
        last_answer_text,
        {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": total_latency_ms,
        },
        False,
        None,
        last_reason,
    )


def _solve_with_retrieval(query: UserQuery, retrieval) -> tuple[str, dict[str, object]]:
    from ..pipelines.main_loop import answer_with_retrieval

    answer = answer_with_retrieval(query, retrieval=retrieval)
    return answer.answer, dict(answer.metadata or {})


def _update_metagraph_with_success(
    adapter: GraphRAGAdapter,
    query: UserQuery,
    retrieval,
    answer: str,
) -> None:
    """Update MetaGraph with successful solution.

    Mark retrieved nodes as used and update their success stats.
    """
    for path in retrieval.paths:
        for node_id in path.node_ids:
            try:
                node = adapter._graph.nodes[
                    next(
                        i
                        for i, n in enumerate(adapter._graph.nodes)
                        if n.node_id == node_id
                    )
                ]
                if node.attributes and "stats" in node.attributes:
                    stats = node.attributes["stats"]
                    stats["n_used"] = int(stats.get("n_used", 0)) + 1
                    stats["n_success"] = int(stats.get("n_success", 0)) + 1
            except (StopIteration, IndexError, KeyError):
                pass

    adapter._save_graph()


def _update_metagraph_with_failure(
    adapter: GraphRAGAdapter,
    query: UserQuery,
    retrieval,
    answer: str,
) -> None:
    """Update MetaGraph with failed solution.

    Mark retrieved nodes as used and update their failure stats.
    """
    for path in retrieval.paths:
        for node_id in path.node_ids:
            try:
                node = adapter._graph.nodes[
                    next(
                        i
                        for i, n in enumerate(adapter._graph.nodes)
                        if n.node_id == node_id
                    )
                ]
                if node.attributes and "stats" in node.attributes:
                    stats = node.attributes["stats"]
                    stats["n_used"] = int(stats.get("n_used", 0)) + 1
                    stats["n_fail"] = int(stats.get("n_fail", 0)) + 1
            except (StopIteration, IndexError, KeyError):
                pass

    adapter._save_graph()
