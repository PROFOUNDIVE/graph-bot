from __future__ import annotations

import datetime
import json
import re
import signal
import time
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ..adapters.distiller import GraphRAGDistiller, get_distiller
from ..adapters.graphrag import GraphRAGAdapter
from ..datatypes import (
    PathEvaluation,
    RetrievalResult,
    StreamCallMetrics,
    StreamProblemMetrics,
    UserQuery,
)
from ..eval.validators import BaseValidator, get_validator
from ..interfaces import AbstractDistiller
from ..pipelines.metrics_logger import StreamMetricsLogger
from ..utils.pricing import calculate_cost, load_pricing_table
from ..utils.slack import send_slack_notification
from ..utils.manifest import RunManifest


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


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Problem processing timed out")


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
    validator_model: str | None = None,
    validator_gated_update: bool = True,
    distiller: AbstractDistiller | None = None,
    distiller_mode: str | None = None,
    max_problems: int | None = None,
    metrics_out_dir: Path = Path("outputs/stream_logs"),
    run_id: str = "run",
):
    """Run continual stream loop for Game of 24."""
    from ..settings import settings

    start_run = time.perf_counter()

    problems = load_game24_problems(problems_file)
    if max_problems:
        problems = problems[:max_problems]

    adapter = GraphRAGAdapter(
        mode=mode,
        use_edges=use_edges,
        policy_id=policy_id,
    )
    validator = get_validator(validator_mode, validator_model)
    if distiller is None:
        actual_mode = (distiller_mode or settings.distiller_mode or "graphrag").lower()
        distiller = get_distiller(actual_mode)
    metrics = StreamMetricsLogger(out_dir=metrics_out_dir, run_id=run_id)

    # Manifest Integration
    manifest = RunManifest()
    manifest.log_start(run_id, config={"mode": mode, "model": settings.llm_model})

    pricing_table = load_pricing_table(settings.pricing_path)
    if settings.llm_model not in pricing_table.get("models", {}):
        raise ValueError(
            f"Model {settings.llm_model} not found in pricing table at {settings.pricing_path}. "
            f"Available models: {list(pricing_table.get('models', {}).keys())}"
        )

    results: list[dict[str, object]] = []
    run_status = "FAILED"

    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        try:
            for t, problem in enumerate(problems, 1):
                problem_id = problem.id
                query = problem.to_user_query()
                start_problem = time.perf_counter()
                poisoned_update_rate: float | None = None

                active_mode = mode or settings.mode
                active_policy_id = policy_id or settings.policy_id

                adapter.mode = active_mode
                adapter.policy_id = active_policy_id

                is_baseline = active_mode in {"io", "cot"}

                try:
                    signal.setitimer(signal.ITIMER_REAL, settings.execution_timeout_sec)

                    distilled_query = query
                    if not is_baseline:
                        start_distill = time.perf_counter()
                        distilled_question = query.question
                        if query.metadata and query.metadata.get("task") == "game24":
                            distilled_question = distiller.distill_query(query.question)
                        distilled_query = query.model_copy(
                            update={"question": distilled_question}
                        )
                        latency_distill = (time.perf_counter() - start_distill) * 1000.0
                        metrics.log_token_event(
                            {
                                "timestamp": datetime.datetime.utcnow().isoformat()
                                + "Z",
                                "stream_run_id": metrics.run_id,
                                "problem_id": problem_id,
                                "t": t,
                                "event_type": "cpu_op",
                                "operation": "distill_query",
                                "status": "success",
                                "model": "cpu",
                                "latency_ms": int(round(latency_distill)),
                                "run_id": f"{metrics.run_id}:{problem_id}",
                                "span_id": metrics.new_call_id(),
                                "component": "pipeline",
                                "metadata": {},
                                "usage": {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0,
                                },
                                "cost_usd": 0.0,
                            }
                        )

                    if is_baseline:
                        retrieval = RetrievalResult(
                            query_id=query.id, paths=[], concatenated_context=""
                        )
                        latency_ret_ms = 0.0
                        ret_error_type = "skipped_mode"
                    else:
                        start_ret = time.perf_counter()
                        retrieval = adapter.retrieve_paths(
                            distilled_query, k=settings.top_k_paths
                        )
                        latency_ret_ms = (time.perf_counter() - start_ret) * 1000.0
                        ret_error_type = None

                    call_id_retrieve = metrics.new_call_id()
                    metrics.log_call(
                        StreamCallMetrics(
                            call_id=call_id_retrieve,
                            parent_id=None,
                            t=t,
                            problem_id=problem_id,
                            operation="retrieve",
                            latency_ms=latency_ret_ms,
                            api_cost_usd=0.0,
                            error_type=ret_error_type,
                        )
                    )
                    metrics.log_token_event(
                        {
                            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                            "stream_run_id": metrics.run_id,
                            "problem_id": problem_id,
                            "t": t,
                            "event_type": "rag_retrieval",
                            "operation": "retrieve",
                            "status": "success" if not ret_error_type else "skipped",
                            "model": settings.embedding_model,
                            "latency_ms": int(round(latency_ret_ms)),
                            "run_id": f"{metrics.run_id}:{problem_id}",
                            "span_id": call_id_retrieve,
                            "component": "rag_infra",
                            "metadata": {
                                "pricing_version": "v0",
                                "mode": active_mode,
                                "distilled_question": distilled_query.question,
                                "packed_context_tokens": (
                                    len(
                                        re.findall(
                                            r"\w+", retrieval.concatenated_context
                                        )
                                    )
                                    if retrieval and retrieval.concatenated_context
                                    else 0
                                ),
                            },
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                            "cost_usd": 0.0,
                        }
                    )

                    if not is_baseline:
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
                        pricing_table=pricing_table,
                        parent_call_id=call_id_retrieve,
                        t=t,
                        problem_id=problem_id,
                        mode=active_mode,
                    )

                    solve_prompt_tokens = _maybe_int(
                        aggregate_usage.get("prompt_tokens")
                    )
                    solve_completion_tokens = _maybe_int(
                        aggregate_usage.get("completion_tokens")
                    )
                    solve_tokens_total = _maybe_int(aggregate_usage.get("total_tokens"))
                    solve_latency_ms = _maybe_float(aggregate_usage.get("latency_ms"))

                    if not is_baseline:
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

                        should_update = solved or (not validator_gated_update)
                        if should_update:
                            _insert_solution_template(
                                adapter=adapter,
                                distiller=distiller,
                                metrics=metrics,
                                t=t,
                                problem_id=problem_id,
                                answer_text=answer_text,
                                solved=solved,
                                query=query,
                                retrieval=retrieval,
                            )
                            poisoned_update_rate = 1.0 if not solved else 0.0

                    signal.setitimer(signal.ITIMER_REAL, 0)

                    memory = adapter.export_graph()

                    tokens_total = solve_tokens_total or 0
                    latency_total_ms = solve_latency_ms or 0.0
                    api_cost_usd = calculate_cost(
                        pricing_table=pricing_table,
                        model_name=settings.llm_model,
                        prompt_tokens=solve_prompt_tokens or 0,
                        completion_tokens=solve_completion_tokens or 0,
                    )

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
                        poisoned_update_rate=poisoned_update_rate,
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

                except TimeoutException:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    latency_ms = (time.perf_counter() - start_problem) * 1000.0

                    metrics.log_call(
                        StreamCallMetrics(
                            call_id=metrics.new_call_id(),
                            parent_id=None,
                            t=t,
                            problem_id=problem_id,
                            operation="timeout",
                            latency_ms=latency_ms,
                            api_cost_usd=0.0,
                            error_type="ERR_TIMEOUT",
                        )
                    )
                    metrics.log_token_event(
                        {
                            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                            "stream_run_id": metrics.run_id,
                            "problem_id": problem_id,
                            "t": t,
                            "event_type": "llm_completion",
                            "operation": "timeout",
                            "status": "timeout",
                            "model": settings.llm_model,
                            "latency_ms": int(round(latency_ms)),
                            "run_id": f"{metrics.run_id}:{problem_id}",
                            "span_id": metrics.new_call_id(),
                            "component": "pipeline",
                            "metadata": {"pricing_version": "v0"},
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                            "cost_usd": 0.0,
                        }
                    )
                    problem_metrics = StreamProblemMetrics(
                        t=t,
                        problem_id=problem_id,
                        solved=False,
                        attempts=0,
                        solved_attempt=None,
                        attempt_success_rate=0.0,
                        llm_calls=0,
                        tokens_total=0,
                        latency_total_ms=latency_ms,
                        api_cost_usd=0.0,
                        retrieval_hit=False,
                        reuse_count=0,
                        memory_n_nodes=0,
                        memory_n_edges=0,
                        contamination_rate=None,
                        poisoned_update_rate=None,
                    )
                    metrics.log_problem(problem_metrics)
                    results.append(
                        {
                            "t": t,
                            "problem_id": problem_id,
                            "solved": False,
                            "error": "timeout",
                        }
                    )

                except Exception as e:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    latency_ms = (time.perf_counter() - start_problem) * 1000.0
                    error_type = type(e).__name__

                    metrics.log_token_event(
                        {
                            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                            "stream_run_id": metrics.run_id,
                            "problem_id": problem_id,
                            "t": t,
                            "event_type": "llm_completion",
                            "operation": "error",
                            "status": "error",
                            "model": settings.llm_model,
                            "latency_ms": int(round(latency_ms)),
                            "run_id": f"{metrics.run_id}:{problem_id}",
                            "span_id": metrics.new_call_id(),
                            "component": "pipeline",
                            "metadata": {"pricing_version": "v0", "error": str(e)},
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                            "cost_usd": 0.0,
                        }
                    )
                    problem_metrics = StreamProblemMetrics(
                        t=t,
                        problem_id=problem_id,
                        solved=False,
                        attempts=0,
                        solved_attempt=None,
                        attempt_success_rate=0.0,
                        llm_calls=0,
                        tokens_total=0,
                        latency_total_ms=latency_ms,
                        api_cost_usd=0.0,
                        retrieval_hit=False,
                        reuse_count=0,
                        memory_n_nodes=0,
                        memory_n_edges=0,
                        contamination_rate=None,
                        poisoned_update_rate=None,
                    )
                    metrics.log_problem(problem_metrics)
                    results.append(
                        {
                            "t": t,
                            "problem_id": problem_id,
                            "solved": False,
                            "error": error_type,
                        }
                    )

                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)

        except Exception as e:
            elapsed = time.perf_counter() - start_run
            n_solved = sum(1 for r in results if r.get("solved"))
            total_cost = 0.0
            if results:
                total_cost = results[-1].get("cumulative_cost", 0.0)

            memory_nodes = 0
            try:
                if adapter:
                    memory_nodes = len(adapter.export_graph().nodes)
            except Exception:
                pass

            payload = {
                "text": (
                    f"*Graph-Bot Stream Failed*\n"
                    f"• Run ID: `{run_id}`\n"
                    f"• Status: failed\n"
                    f"• Elapsed: {elapsed:.2f}s\n"
                    f"• Solved: {n_solved}\n"
                    f"• Memory Nodes: {memory_nodes}\n"
                    f"• Cost: ${total_cost:.4f}\n"
                    f"• Error: `{str(e)}`"
                )
            }
            send_slack_notification(settings.slack_webhook_url, payload)
            raise e

        finally:
            signal.signal(signal.SIGALRM, old_handler)

        elapsed = time.perf_counter() - start_run
        n_solved = sum(1 for r in results if r.get("solved"))
        total_cost = 0.0
        if results:
            total_cost = results[-1].get("cumulative_cost", 0.0)
        memory_nodes = len(adapter.export_graph().nodes)

        payload = {
            "text": (
                f"*Graph-Bot Stream Completed*\n"
                f"• Run ID: `{run_id}`\n"
                f"• Status: completed\n"
                f"• Elapsed: {elapsed:.2f}s\n"
                f"• Solved: {n_solved}\n"
                f"• Memory Nodes: {memory_nodes}\n"
                f"• Cost: ${total_cost:.4f}\n"
            )
        }
        send_slack_notification(settings.slack_webhook_url, payload)

        run_status = "COMPLETED"
        return results

    finally:
        cumulative_cost = 0.0
        if results:
            cumulative_cost = results[-1].get("cumulative_cost", 0.0)

        manifest.log_end(
            run_id, status=run_status, metrics={"cumulative_cost": cumulative_cost}
        )


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
    distiller: AbstractDistiller,
    metrics: StreamMetricsLogger,
    t: int,
    problem_id: str,
    answer_text: str,
    solved: bool,
    query: UserQuery,
    retrieval: RetrievalResult | None = None,
) -> None:
    from ..datatypes import ReasoningEdge, ReasoningNode, ReasoningTree

    # 1. Distill trace into a compact template
    start_distill = time.perf_counter()
    distill_tree = ReasoningTree(
        tree_id=f"episode-{problem_id}",
        root_id=f"episode-{problem_id}-solution",
        nodes=[
            ReasoningNode(
                node_id=f"episode-{problem_id}-solution",
                text=answer_text,
                type="answer",
                attributes={"query": query.question},
            )
        ],
        edges=[],
        provenance={
            "task": "game24",
            "source": "stream",
            "query": query.question,
            "solved": solved,
        },
    )
    distilled_nodes = distiller.distill_trace(distill_tree)
    latency_distill = (time.perf_counter() - start_distill) * 1000.0

    metrics.log_token_event(
        {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "stream_run_id": metrics.run_id,
            "problem_id": problem_id,
            "t": t,
            "event_type": "cpu_op",
            "operation": "distill_trace",
            "status": "success",
            "model": "cpu",
            "latency_ms": int(round(latency_distill)),
            "run_id": f"{metrics.run_id}:{problem_id}",
            "span_id": metrics.new_call_id(),
            "component": "pipeline",
            "metadata": {},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
        }
    )

    # 2. Identify source edges if retrieval was used
    edges: list[ReasoningEdge] = []
    if retrieval and retrieval.paths:
        for path in retrieval.paths:
            if not path.node_ids:
                continue
            # The template used is effectively the last node in the retrieved path
            source_id = path.node_ids[-1]
            edges.append(
                ReasoningEdge(
                    src=source_id,
                    dst=f"episode-{problem_id}-solution",
                    relation="used_for",
                    attributes={
                        "weight": 1.0,
                        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                    },
                )
            )

    tree = ReasoningTree(
        tree_id=f"episode-{problem_id}",
        root_id=f"episode-{problem_id}-solution",
        nodes=distilled_nodes,
        edges=edges,
        provenance={"task": "game24", "source": "stream"},
    )
    adapter.insert_trees([tree])

    metrics.log_token_event(
        {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "stream_run_id": metrics.run_id,
            "problem_id": problem_id,
            "t": t,
            "event_type": "memory_update",
            "operation": "insert_trees",
            "status": "success",
            "model": "cpu",
            "latency_ms": 0,
            "run_id": f"{metrics.run_id}:{problem_id}",
            "span_id": metrics.new_call_id(),
            "component": "metagraph",
            "metadata": {"edges_added_count": len(edges)},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cost_usd": 0.0,
        }
    )


def _normalize_candidate_line(
    raw_output: str, allowed_numbers: list[int] | None = None
) -> tuple[str, str]:
    import re

    # 1. <answer> block extraction (Priority 1 per Spec)
    match = re.search(r"<answer>(.*?)</answer>", raw_output, re.DOTALL)
    if match:
        content = match.group(1).strip()
        for line in content.splitlines():
            line = line.strip()
            if line:
                # Strip potential "= 24" suffix
                if "=" in line:
                    line = line.split("=")[0].strip()
                return line, "answer_block"

    # 2. Check for explicit "Output: <expr> = 24" format (GoT style)
    # We look for the LAST occurrence of "Output:" if multiple exist (CoT)
    output_matches = list(re.finditer(r"^\s*Output:\s*(.*)$", raw_output, re.MULTILINE))
    if output_matches:
        candidate = output_matches[-1].group(1).strip()
        # Strip potential "= 24" suffix
        if "=" in candidate:
            candidate = candidate.split("=")[0].strip()
        return candidate, "got_output_format"

    # 3. Fallback: Bottom-to-top scan
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    for line in reversed(lines):
        candidate = line
        # Strip "Output:" if present
        if candidate.startswith("Output:"):
            candidate = candidate[len("Output:") :].strip()

        # Strip "= 24" if present
        if "=" in candidate:
            candidate = candidate.split("=")[0].strip()

        if allowed_numbers is not None:
            # Check plausibility (precheck)
            if (
                _precheck_candidate(
                    candidate_line=candidate, allowed_numbers=allowed_numbers
                )
                is None
            ):
                return candidate, "fallback_bottom_scan"
        else:
            # If no numbers provided, just return the last non-empty line
            return candidate, "fallback_bottom_scan"

    return "", "empty"


def _precheck_candidate(
    *,
    candidate_line: str,
    allowed_numbers: list[int],
) -> str | None:
    import re

    from ..eval.validators import extract_game24_expression_number_literals

    if not candidate_line:
        return "empty_output"

    # GoT style might leave some artifacts, but we stripped '=' in normalize.
    # So we strictly forbid '=' and '->' here to ensure clean expression.
    if "→" in candidate_line or "=" in candidate_line:
        return "format_error"

    if re.search(r"[^0-9\s\(\)\+\-\*/]", candidate_line):
        return "illegal_tokens"

    number_tokens = extract_game24_expression_number_literals(candidate_line)
    if number_tokens is None:
        return "format_error"

    if sorted(number_tokens) != sorted(allowed_numbers):
        return "wrong_numbers"

    return None


def _solve_with_retries(
    *,
    query: UserQuery,
    retrieval,
    validator: BaseValidator,
    metrics: StreamMetricsLogger,
    pricing_table: dict[str, Any],
    parent_call_id: str,
    t: int,
    problem_id: str,
    mode: str = "graph_bot",
) -> tuple[str, dict[str, object], bool, int | None, str | None, int]:
    import re

    from ..adapters.mock_client import MockLLMClient
    from ..adapters.vllm_openai_client import VLLMOpenAIClient
    from ..settings import settings

    temperatures = [
        settings.retry_temperature_1,
        settings.retry_temperature_2,
        settings.retry_temperature_3,
    ]

    # GoT Prompts (Source: graph-of-thoughts/tasks/gameof24.py)
    got_io_system = """You are playing the 24 Game.

Given four numbers, use each number exactly once (in any order) and only the operations +, -, *, /.
Parentheses are allowed. Create one valid Python expression that evaluates to 24.

Output must follow the example format as closely as possible: a single line of the form
Output: <python_expression>
Do not include any additional explanation or text."""

    got_io_user_template = """<Example>
Input: 4 9 10 13
Output: (10 - 4) * (13 - 9)
</Example>

Input: {input}
Output:"""

    got_cot_system = """You are playing the 24 Game.

Given four numbers, use each number exactly once (in any order) and only the operations +, -, *, /.
Parentheses are allowed. Find one valid Python expression that evaluates to 24.

You may show intermediate steps, but you MUST enclose the final Python expression in <answer> tags:
<answer>(1 + 2) * 8</answer>
Do not include any additional text inside the tags."""

    got_cot_user_template = """<Examples>
Input: 4 9 10 13
Work:
(10 - 4) = 6
(13 - 9) = 4
6 * 4 = 24
<answer>(10 - 4) * (13 - 9)</answer>

Input: 1 3 4 6
Work:
(6 / 3) = 2
4 * 2 = 8
8 * 3 = 24  (using 1 and 3? not allowed) -> backtrack
(4 - 1) = 3
6 * 3 = 18
18 + 3 = 21 -> backtrack
(6 - 1) = 5
5 * 4 = 20
20 + 3 = 23 -> backtrack
(6 - 4) = 2
3 * 2 = 6
6 * 4 = 24 (uses 4 twice) -> backtrack
(6 / (1 - 3/4)) = 24
<answer>6 / (1 - 3/4)</answer>
</Examples>

Input: {input}
"""

    pattern = re.compile(r"(-?\d+\.?\d*)")
    numbers = [int(float(x)) for x in pattern.findall(query.question)[:4]]
    numbers_str = " ".join(str(x) for x in numbers)

    if mode == "io":
        system = got_io_system
        user_template = got_io_user_template.format(input=numbers_str)
    elif mode == "cot":
        system = got_cot_system
        user_template = got_cot_user_template.format(input=numbers_str)
    else:
        # graph_bot mode now uses CoT style (multi-line reasoning)
        system = got_cot_system
        base_user = got_cot_user_template.format(input=numbers_str)
        user_template = f"{base_user}\nRetrieved templates/context:\n{retrieval.concatenated_context}\n"

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_latency_ms = 0.0

    if settings.llm_provider == "mock":
        client = MockLLMClient(model=settings.llm_model)
    else:
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

        prompt_variant = f"{mode}:base" if attempt_index == 1 else f"{mode}:retry"

        current_user = user_template
        if attempt_index > 1 and last_reason:
            prev = (last_raw_output or "").strip()
            retry_line = (
                "\n\nPrevious attempt was invalid.\n"
                f"Reason: {last_reason}\n"
                f"Previous output: {prev}\n"
                "Please fix it. Output ONLY the valid Python expression in the format:\n"
                "Output: <expression>"
            )
            current_user += retry_line

        answer_text = ""
        solve_error_type: str | None = None
        usage = None
        try:
            answer_text, usage = client.chat(
                system=system,
                user=current_user,
                temperature=temperature,
            )

            # Audit Logic
            if usage and getattr(usage, "audit_prompt_tokens", None) is not None:
                audit_prompt_tokens = usage.audit_prompt_tokens
                prompt_tokens = usage.prompt_tokens or 0
                if prompt_tokens > 0 and audit_prompt_tokens is not None:
                    gap = abs(audit_prompt_tokens - prompt_tokens) / prompt_tokens
                    if gap > 0.05:
                        metrics.log_token_event(
                            {
                                "timestamp": datetime.datetime.utcnow().isoformat()
                                + "Z",
                                "stream_run_id": metrics.run_id,
                                "problem_id": problem_id,
                                "t": t,
                                "event_type": "token_audit_gap",
                                "operation": "solve_audit",
                                "status": "warning",
                                "model": settings.llm_model,
                                "latency_ms": 0,
                                "run_id": f"{metrics.run_id}:{problem_id}",
                                "span_id": metrics.new_call_id(),
                                "component": "audit",
                                "metadata": {
                                    "local_tokens": prompt_tokens,
                                    "remote_tokens": audit_prompt_tokens,
                                    "gap_ratio": gap,
                                },
                                "usage": {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0,
                                },
                                "cost_usd": 0.0,
                            }
                        )

        except TimeoutException:
            raise
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
        candidate_line, normalization = _normalize_candidate_line(
            raw_output, allowed_numbers=numbers
        )

        attempt_passed: bool = False
        reason = None
        precheck_failure_reason = None

        if solve_error_type is None:
            precheck_failure_reason = _precheck_candidate(
                candidate_line=candidate_line,
                allowed_numbers=numbers,
            )
            start_val = time.perf_counter()
            if precheck_failure_reason is None:
                attempt_passed = bool(
                    validator.validate(candidate_line, query.question)
                )
                if attempt_passed:
                    answer_text = candidate_line
                else:
                    reason = validator.failure_reason(candidate_line, query.question)
                    answer_text = candidate_line
            else:
                reason = precheck_failure_reason
                answer_text = candidate_line
            latency_val_ms = (time.perf_counter() - start_val) * 1000.0
        else:
            reason = "solve_error"
            latency_val_ms = 0.0

        last_reason = reason
        last_raw_output = raw_output
        last_answer_text = answer_text

        attempt_api_cost_usd = calculate_cost(
            pricing_table=pricing_table,
            model_name=settings.llm_model,
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
        )

        metrics.log_call(
            StreamCallMetrics(
                call_id=call_id_solve,
                parent_id=parent_call_id,
                t=t,
                problem_id=problem_id,
                operation="solve",
                attempt_index=attempt_index,
                temperature=temperature,
                validator_passed=bool(attempt_passed),
                failure_reason=reason,
                prompt_variant=prompt_variant,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=attempt_total_tokens,
                latency_ms=latency_ms or 0.0,
                api_cost_usd=attempt_api_cost_usd,
                error_type=solve_error_type,
                raw_output=raw_output,
                candidate_line=candidate_line,
                normalization=normalization,
                precheck_failure_reason=precheck_failure_reason,
            )
        )
        metrics.log_token_event(
            {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "stream_run_id": metrics.run_id,
                "problem_id": problem_id,
                "t": t,
                "event_type": "llm_completion",
                "operation": "solve",
                "status": "failed" if solve_error_type else "success",
                "model": settings.llm_model,
                "latency_ms": int(round(latency_ms or 0.0)),
                "run_id": f"{metrics.run_id}:{problem_id}",
                "span_id": call_id_solve,
                "component": "pipeline",
                "metadata": {"pricing_version": "v0"},
                "usage": {
                    "prompt_tokens": prompt_tokens or 0,
                    "completion_tokens": completion_tokens or 0,
                    "total_tokens": attempt_total_tokens or 0,
                },
                "cost_usd": float(attempt_api_cost_usd),
            }
        )

        call_id_validate = metrics.new_call_id()
        validate_error_type: str | None
        if solve_error_type is not None:
            validate_error_type = "skipped_solve_error"
        elif precheck_failure_reason is not None:
            validate_error_type = "skipped_precheck"
        else:
            validate_error_type = None
        validator_name = validator.get_validator_name()
        validate_operation = (
            "validate_llm_judge" if validator_name == "weak_llm_judge" else "validate"
        )

        metrics.log_call(
            StreamCallMetrics(
                call_id=call_id_validate,
                parent_id=call_id_solve,
                t=t,
                problem_id=problem_id,
                operation="validate",
                attempt_index=attempt_index,
                temperature=temperature,
                validator_passed=bool(attempt_passed),
                failure_reason=reason,
                prompt_variant=prompt_variant,
                latency_ms=latency_val_ms,
                api_cost_usd=0.0,
                error_type=validate_error_type,
                raw_output=raw_output,
                candidate_line=candidate_line,
                normalization=normalization,
                precheck_failure_reason=precheck_failure_reason,
            )
        )
        metrics.log_token_event(
            {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "stream_run_id": metrics.run_id,
                "problem_id": problem_id,
                "t": t,
                "event_type": "tool_call",
                "operation": validate_operation,
                "status": "failed" if validate_error_type else "success",
                "model": validator_name,
                "latency_ms": int(round(latency_val_ms)),
                "run_id": f"{metrics.run_id}:{problem_id}",
                "span_id": call_id_validate,
                "component": "evaluator",
                "metadata": {"pricing_version": "v0"},
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "cost_usd": 0.0,
            }
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


def _solve_with_retrieval(query: UserQuery, retrieval) -> str:
    """Solve problem using retrieved templates.

    Currently stubbed - returns placeholder answer.
    Future: Integrate with actual LLM API.
    """
    _ = (query, retrieval)
    return "RETRIEVAL_PLACEHOLDER"
