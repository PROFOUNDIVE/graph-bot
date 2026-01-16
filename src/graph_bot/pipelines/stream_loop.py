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

        (
            answer_text,
            aggregate_usage,
            solved,
            solved_attempt,
            _last_reason,
            attempts_used,
        ) = _solve_with_retries(
            query=query,
            retrieval=retrieval,
            validator=validator,
            metrics=metrics,
            parent_call_id=call_id_retrieve,
            t=t,
            problem_id=problem_id,
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

        attempts_used = max(1, attempts_used)
        success_rate = (1.0 / attempts_used) if solved else 0.0

        problem_metrics = StreamProblemMetrics(
            t=t,
            problem_id=problem_id,
            solved=solved,
            attempts=attempts_used,
            solved_attempt=solved_attempt,
            attempt_success_rate=success_rate,
            llm_calls=attempts_used,
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


def _normalize_candidate_line(raw_output: str) -> tuple[str, str]:
    raw = raw_output.strip()
    lines = [line.strip() for line in raw.splitlines()]
    for line in lines:
        if line:
            return line, "first_non_empty_line"
    return "", "empty"


def _precheck_candidate(
    *,
    candidate_line: str,
    allowed_numbers: list[int],
) -> str | None:
    import re

    if not candidate_line:
        return "empty_output"

    if "→" in candidate_line or "=" in candidate_line:
        return "format_error"

    if re.search(r"[^0-9\s\(\)\+\-\*/]", candidate_line):
        return "illegal_tokens"

    number_tokens = [int(x) for x in re.findall(r"\d+", candidate_line)]
    if sorted(number_tokens) != sorted(allowed_numbers):
        return "wrong_numbers"

    return None


def _solve_with_retries(
    *,
    query: UserQuery,
    retrieval,
    validator: BaseValidator,
    metrics: StreamMetricsLogger,
    parent_call_id: str,
    t: int,
    problem_id: str,
) -> tuple[str, dict[str, object], bool, int | None, str | None, int]:
    import re

    from ..adapters.vllm_openai_client import VLLMOpenAIClient
    from ..settings import settings

    temperatures = [
        settings.retry_temperature_1,
        settings.retry_temperature_2,
        settings.retry_temperature_3,
    ]

    pattern = re.compile(r"(-?\d+\.?\d*)")
    numbers = [int(float(x)) for x in pattern.findall(query.question)[:4]]
    numbers_str = " ".join(str(x) for x in numbers)

    system = "You solve Game of 24. Output only an arithmetic expression."

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_latency_ms = 0.0

    client = VLLMOpenAIClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )

    last_reason: str | None = None
    last_raw_output: str | None = None
    last_answer_text = ""
    attempts_used = 0

    for attempt_index in range(1, settings.retry_max_attempts + 1):
        attempts_used = attempt_index
        temperature = temperatures[min(attempt_index - 1, len(temperatures) - 1)]

        call_id_solve = metrics.new_call_id()

        prompt_variant = "base" if attempt_index == 1 else "retry"

        retry_line = ""
        if attempt_index > 1 and last_reason:
            prev = (last_raw_output or "").strip()
            retry_line = (
                "Previous attempt was invalid.\n"
                f"Reason: {last_reason}\n"
                f"Previous raw output (verbatim):\n{prev}\n\n"
                "Fix it and return ONLY a corrected expression.\n"
                "Checklist:\n"
                "- One line only (no explanation).\n"
                "- Use each given number exactly once as a standalone number token.\n"
                "- Never concatenate digits (e.g., 9 and 9 must not become 99).\n"
                "- Only + - * / and parentheses.\n"
                "- Must evaluate to 24.\n\n"
            )

        user = (
            f"Numbers: {numbers_str}\n\n"
            "Rules:\n"
            "- Use each given number exactly once.\n"
            "- Use only + - * / and parentheses.\n"
            "- Do NOT output '= 24' or '→ 24'.\n"
            "- Output MUST be a single line containing only the expression.\n\n"
            + retry_line
            + f"Retrieved templates/context:\n{retrieval.concatenated_context}\n"
        )

        answer_text = ""
        solve_error_type: str | None = None
        usage = None
        try:
            answer_text, usage = client.chat(
                system=system,
                user=user,
                temperature=temperature,
            )
        except Exception as exc:
            solve_error_type = type(exc).__name__

        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        attempt_total_tokens = usage.total_tokens if usage else None
        latency_ms = usage.latency_ms if usage else None

        total_prompt_tokens += prompt_tokens or 0
        total_completion_tokens += completion_tokens or 0
        total_tokens += attempt_total_tokens or 0
        total_latency_ms += latency_ms or 0.0

        raw_output = answer_text or ""
        candidate_line, normalization = _normalize_candidate_line(raw_output)

        attempt_passed = False
        reason = None
        precheck_failure_reason = None

        if solve_error_type is None:
            precheck_failure_reason = _precheck_candidate(
                candidate_line=candidate_line,
                allowed_numbers=numbers,
            )
            if precheck_failure_reason is None:
                attempt_passed = validator.validate(candidate_line, query.question)
                if attempt_passed:
                    answer_text = candidate_line
                else:
                    reason = validator.failure_reason(candidate_line, query.question)
                    answer_text = candidate_line
            else:
                reason = precheck_failure_reason
                answer_text = candidate_line
        else:
            reason = "solve_error"

        last_reason = reason
        last_raw_output = raw_output
        last_answer_text = answer_text

        metrics.log_call(
            StreamCallMetrics(
                call_id=call_id_solve,
                parent_id=parent_call_id,
                t=t,
                problem_id=problem_id,
                operation="solve",
                attempt_index=attempt_index,
                temperature=temperature,
                validator_passed=attempt_passed,
                failure_reason=reason,
                prompt_variant=prompt_variant,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=attempt_total_tokens,
                latency_ms=latency_ms,
                error_type=solve_error_type,
                raw_output=raw_output,
                candidate_line=candidate_line,
                normalization=normalization,
                precheck_failure_reason=precheck_failure_reason,
            )
        )

        call_id_validate = metrics.new_call_id()
        validate_error_type: str | None
        if solve_error_type is not None:
            validate_error_type = "skipped_solve_error"
        elif precheck_failure_reason is not None:
            validate_error_type = "skipped_precheck"
        else:
            validate_error_type = None

        metrics.log_call(
            StreamCallMetrics(
                call_id=call_id_validate,
                parent_id=call_id_solve,
                t=t,
                problem_id=problem_id,
                operation="validate",
                attempt_index=attempt_index,
                temperature=temperature,
                validator_passed=attempt_passed,
                failure_reason=reason,
                prompt_variant=prompt_variant,
                error_type=validate_error_type,
                raw_output=raw_output,
                candidate_line=candidate_line,
                normalization=normalization,
                precheck_failure_reason=precheck_failure_reason,
            )
        )

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
                attempts_used,
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
        attempts_used,
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
