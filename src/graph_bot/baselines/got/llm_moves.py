from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class MoveCallResult:
    operation: str
    prompt: str
    raw_output: str
    moves: List[Dict[str, Any]]
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float | None


def build_llm_client() -> Any:
    from ...settings import settings

    if settings.llm_provider == "mock":
        from ...adapters.mock_client import MockLLMClient

        return MockLLMClient(model=settings.llm_model)

    from ...adapters.vllm_openai_client import VLLMOpenAIClient

    return VLLMOpenAIClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )


def parse_jsonl_moves(raw_output: str) -> List[Dict[str, Any]]:
    moves: List[Dict[str, Any]] = []
    dedup: set[tuple[int, int, str]] = set()

    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if "{" not in line or "}" not in line:
            continue

        try:
            line_json = line[line.index("{") : line.rindex("}") + 1]
            obj = json.loads(line_json)
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue
        pick = obj.get("pick")
        op = obj.get("op")
        if not isinstance(pick, list) or len(pick) != 2:
            continue
        if op not in {"+", "-", "*", "/"}:
            continue
        try:
            id1 = int(pick[0])
            id2 = int(pick[1])
        except Exception:
            continue
        if id1 == id2:
            continue

        key = (id1, id2, str(op))
        if key in dedup:
            continue
        dedup.add(key)
        moves.append({"pick": [id1, id2], "op": str(op)})

    return moves


def call_llm_for_moves(
    *,
    client: Any,
    prompt: str,
    operation: str,
    temperature: float,
) -> MoveCallResult:
    system = (
        "You are a 24-game search assistant. "
        "Return only JSONL move objects that follow the requested schema."
    )
    raw_output, usage = client.chat(
        system=system,
        user=prompt,
        temperature=temperature,
    )
    return MoveCallResult(
        operation=operation,
        prompt=prompt,
        raw_output=raw_output,
        moves=parse_jsonl_moves(raw_output),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        latency_ms=getattr(usage, "latency_ms", None),
    )
