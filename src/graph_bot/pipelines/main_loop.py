from __future__ import annotations
from __future__ import annotations

from typing import Iterable

from ..adapters.graphrag import GraphRAGAdapter
from ..adapters.vllm_openai_client import VLLMOpenAIClient
from ..settings import settings
from ..types import LLMAnswer, RetrievalResult, SeedData, UserQuery
from .build_trees import build_reasoning_trees_from_seeds
from .retrieve import retrieve_k_optimal_paths


def process_seeds_and_write(seeds: Iterable[SeedData]) -> int:
    _adapter = GraphRAGAdapter()
    trees = build_reasoning_trees_from_seeds(seeds)
    written = _adapter.insert_trees(trees)
    return written


def postprocess_after_T_inputs(trees_count: int | None = None) -> int:
    del trees_count
    adapter = GraphRAGAdapter()
    return adapter.prune_graph()


def answer_with_retrieval(
    query: UserQuery,
    *,
    retrieval: RetrievalResult | None = None,
    adapter: GraphRAGAdapter | None = None,
) -> LLMAnswer:
    active_adapter = adapter or GraphRAGAdapter()
    active_retrieval = retrieval or retrieve_k_optimal_paths(
        query, adapter=active_adapter
    )

    system = "You solve Game of 24. Output only an arithmetic expression."
    user = (
        f"Numbers: {query.question}\n\n"
        "Rules:\n"
        "- Use each given number exactly once.\n"
        "- Use only + - * / and parentheses.\n"
        "- Do NOT output '= 24' or '→ 24'.\n"
        "- Output MUST be a single line containing only the expression.\n\n"
        f"Retrieved templates/context:\n{active_retrieval.concatenated_context}\n"
    )

    client = VLLMOpenAIClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
    text, usage = client.chat(
        system=system,
        user=user,
        temperature=settings.llm_temperature,
    )

    active_adapter.register_usage(active_retrieval.paths)

    return LLMAnswer(
        query_id=query.id,
        answer=text.strip(),
        metadata={
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "latency_ms": usage.latency_ms,
        },
    )
