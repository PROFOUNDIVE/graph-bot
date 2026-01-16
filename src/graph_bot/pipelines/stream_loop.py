from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from pydantic import BaseModel, Field

from ..types import UserQuery
from ..adapters.graphrag import GraphRAGAdapter
from ..eval.validators import get_validator
from ..logsetting import logger


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
):
    """Run continual stream loop for Game of 24.

    For each problem:
    1. Retrieve templates from MetaGraph
    2. Solve problem using LLM
    3. Validate answer with validator
    4. Update MetaGraph with validation feedback

    Args:
        problems_file: Path to JSONL file with Game24 problems
        mode: Execution mode (graph_bot or flat_template_rag)
        use_edges: Whether to use graph edges for path construction
        policy_id: Selection policy (semantic_only or semantic_topK_stats_rerank)
        validator_mode: Validator mode (oracle, exec_repair, weak_llm_judge)
        max_problems: Optional limit on number of problems to process
    """
    from ..settings import settings

    logger.info(f"Starting continual stream from {problems_file}")
    logger.info(f"Mode: {mode or settings.mode}")
    logger.info(
        f"Use edges: {use_edges if use_edges is not None else settings.use_edges}"
    )
    logger.info(f"Policy: {policy_id or settings.policy_id}")
    logger.info(f"Validator: {validator_mode}")

    problems = load_game24_problems(problems_file)
    if max_problems:
        problems = problems[:max_problems]

    logger.info(f"Loaded {len(problems)} problems")

    adapter = GraphRAGAdapter(
        mode=mode,
        use_edges=use_edges,
        policy_id=policy_id,
    )

    validator = get_validator(validator_mode)

    results = []
    for idx, problem in enumerate(problems, 1):
        logger.info(f"[{idx}/{len(problems)}] Processing {problem.id}")

        query = problem.to_user_query()

        try:
            retrieval = adapter.retrieve_paths(query, k=settings.top_k_paths)
            logger.debug(f"Retrieved {len(retrieval.paths)} paths")

            answer_text = _solve_with_retrieval(query, retrieval)
            logger.debug(f"Answer: {answer_text}")

            solved = validator.validate(answer_text, query.question)
            logger.info(f"Validation: {'PASS' if solved else 'FAIL'}")

            results.append(
                {
                    "problem_id": problem.id,
                    "solved": solved,
                    "answer": answer_text,
                    "n_retrieved_paths": len(retrieval.paths),
                }
            )

            if solved:
                _update_metagraph_with_success(adapter, query, retrieval, answer_text)
            else:
                _update_metagraph_with_failure(adapter, query, retrieval, answer_text)

        except Exception as e:
            logger.error(f"Error processing {problem.id}: {e}")
            results.append(
                {
                    "problem_id": problem.id,
                    "solved": False,
                    "answer": "",
                    "error": str(e),
                }
            )

    logger.info(
        f"Stream complete. Solved: {sum(r['solved'] for r in results)}/{len(results)}"
    )

    return results


def _solve_with_retrieval(query: UserQuery, retrieval) -> str:
    """Solve problem using retrieved templates.

    Currently stubbed - returns placeholder answer.
    Future: Integrate with actual LLM API.
    """
    from ..pipelines.main_loop import answer_with_retrieval

    answer = answer_with_retrieval(query, retrieval=retrieval)
    return answer.answer


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
