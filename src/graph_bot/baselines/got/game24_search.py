from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .game24_prompts import build_improve_prompt, build_next_move_prompt
from .game24_utils import game24_score, test_game24
from .llm_moves import MoveCallResult, call_llm_for_moves


State = Dict[str, Any]


@dataclass(frozen=True)
class SearchVariantConfig:
    variant: str
    max_depth: int
    num_branches: int
    beam_width: int
    refine_width: int
    num_tries: int


@dataclass(frozen=True)
class SearchResult:
    variant: str
    best_state: State | None
    best_terminal_state: State | None
    solved: bool
    llm_calls: List[MoveCallResult]
    aggregate_usage: Dict[str, object]


SEARCH_VARIANTS: dict[str, SearchVariantConfig] = {
    "got": SearchVariantConfig(
        variant="got",
        max_depth=3,
        num_branches=30,
        beam_width=3,
        refine_width=10,
        num_tries=1,
    ),
    "tot": SearchVariantConfig(
        variant="tot",
        max_depth=3,
        num_branches=30,
        beam_width=3,
        refine_width=0,
        num_tries=0,
    ),
}


def _derive_next_id(items: list[dict[str, Any]]) -> int:
    if not items:
        return 0
    try:
        return max(int(item.get("id", -1)) for item in items) + 1
    except Exception:
        return len(items)


def _normalize_numbers(numbers: Sequence[float | int]) -> list[float]:
    return [float(num) for num in numbers]


def init_state(numbers: Sequence[float | int]) -> State:
    normalized = _normalize_numbers(numbers)
    items = []
    for idx, value in enumerate(normalized):
        expr = str(int(value)) if float(value).is_integer() else str(value)
        items.append({"id": idx, "value": float(value), "expr": expr})
    items_json = json.dumps(items)
    return {
        "items": items,
        "next_id": len(items),
        "depth": 0,
        "items_json": items_json,
        "prev_items_json": items_json,
        "last_move": None,
    }


def _find_item(items: list[dict[str, Any]], item_id: int) -> dict[str, Any] | None:
    for item in items:
        try:
            if int(item.get("id", -1)) == int(item_id):
                return item
        except Exception:
            continue
    return None


def _apply_op(a: float, b: float, op: str) -> float | None:
    try:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            if abs(b) < 1e-12:
                return None
            return a / b
    except Exception:
        return None
    return None


def _invalid_next_state(state: State, move: dict[str, Any] | None) -> State:
    current_items = state.get("items", [])
    items_json = state.get("items_json", "[]")
    invalid = dict(state)
    invalid["invalid_move"] = True
    invalid["items"] = current_items
    invalid["items_json"] = items_json
    invalid["prev_items_json"] = items_json
    invalid["prev_next_id"] = int(state.get("next_id", _derive_next_id(current_items)))
    invalid["last_move"] = move
    invalid["depth"] = int(state.get("depth", 0)) + 1
    invalid["next_id"] = int(state.get("next_id", _derive_next_id(current_items)))
    return invalid


def apply_move(state: State, move: dict[str, Any]) -> State:
    try:
        pick = move.get("pick")
        op = move.get("op")
        if not isinstance(pick, list) or len(pick) != 2:
            return _invalid_next_state(state, move)
        if op not in {"+", "-", "*", "/"}:
            return _invalid_next_state(state, move)

        id1, id2 = int(pick[0]), int(pick[1])
        if id1 == id2:
            return _invalid_next_state(state, move)

        items = state.get("items", [])
        if not isinstance(items, list) or len(items) == 0:
            return _invalid_next_state(state, move)

        item1 = _find_item(items, id1)
        item2 = _find_item(items, id2)
        if item1 is None or item2 is None:
            return _invalid_next_state(state, move)

        a = float(item1.get("value", 0.0))
        b = float(item2.get("value", 0.0))
        result = _apply_op(a, b, str(op))
        if result is None:
            return _invalid_next_state(state, move)

        expr1 = str(item1.get("expr", item1.get("value", "")))
        expr2 = str(item2.get("expr", item2.get("value", "")))
        new_expr = f"({expr1}{op}{expr2})"

        next_items = []
        for item in items:
            item_id = int(item.get("id", -1))
            if item_id in {id1, id2}:
                continue
            next_items.append(item)

        new_id = int(state.get("next_id", _derive_next_id(items)))
        next_items.append({"id": new_id, "value": float(result), "expr": new_expr})
        next_items_json = json.dumps(next_items)
        prev_items_json = state.get("items_json", json.dumps(items))

        next_state = dict(state)
        next_state.pop("invalid_move", None)
        next_state["items"] = next_items
        next_state["next_id"] = new_id + 1
        next_state["depth"] = int(state.get("depth", 0)) + 1
        next_state["items_json"] = next_items_json
        next_state["prev_items_json"] = prev_items_json
        next_state["prev_next_id"] = int(state.get("next_id", _derive_next_id(items)))
        next_state["last_move"] = {"pick": [id1, id2], "op": str(op)}
        next_state["current"] = new_expr
        return next_state
    except Exception:
        return _invalid_next_state(state, move)


def _restore_previous_state(state: State) -> State:
    prev_items_json = state.get("prev_items_json", state.get("items_json", "[]"))
    try:
        prev_items = json.loads(prev_items_json)
    except Exception:
        prev_items = state.get("items", [])

    if not isinstance(prev_items, list):
        prev_items = []
    next_id = int(state.get("prev_next_id", _derive_next_id(prev_items)))
    depth = max(0, int(state.get("depth", 0)) - 1)
    return {
        "items": prev_items,
        "next_id": next_id,
        "depth": depth,
        "items_json": prev_items_json,
        "prev_items_json": prev_items_json,
        "last_move": state.get("last_move"),
    }


def _score_then_keep(states: list[State], keep_n: int) -> list[State]:
    if keep_n <= 0:
        return []
    scored = sorted(states, key=game24_score)
    return scored[:keep_n]


def _best_terminal(states: list[State]) -> State | None:
    terminals = [state for state in states if test_game24(state)]
    if not terminals:
        return None
    return min(terminals, key=game24_score)


def _aggregate_usage(llm_calls: list[MoveCallResult]) -> dict[str, object]:
    prompt_tokens = sum((call.prompt_tokens or 0) for call in llm_calls)
    completion_tokens = sum((call.completion_tokens or 0) for call in llm_calls)
    total_tokens = sum((call.total_tokens or 0) for call in llm_calls)
    latency_ms = sum((call.latency_ms or 0.0) for call in llm_calls)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
    }


def run_search_variant(
    *,
    variant: str,
    numbers: Sequence[float | int],
    client: Any,
    temperature: float = 0.0,
) -> SearchResult:
    config = SEARCH_VARIANTS.get(variant)
    if config is None:
        supported = ", ".join(sorted(SEARCH_VARIANTS))
        raise ValueError(f"Unsupported variant '{variant}'. Supported: {supported}")

    frontier = [init_state(numbers)]
    best_state: State | None = frontier[0]
    best_terminal_state: State | None = None
    llm_calls: list[MoveCallResult] = []

    for _ in range(config.max_depth):
        generated: list[State] = []
        for state in frontier:
            prompt = build_next_move_prompt(
                items_json=state.get("items_json", "[]"),
                num_branches=config.num_branches,
            )
            call = call_llm_for_moves(
                client=client,
                prompt=prompt,
                operation="generate",
                temperature=temperature,
            )
            llm_calls.append(call)

            if not call.moves:
                generated.append(_invalid_next_state(state, None))
                continue

            for move in call.moves[: config.num_branches]:
                generated.append(apply_move(state, move))

        if not generated:
            break

        best_terminal_state = _best_terminal(generated)
        if best_terminal_state is not None:
            best_state = best_terminal_state
            break

        if config.variant == "got":
            refine_pool = _score_then_keep(generated, config.refine_width)
            refined: list[State] = []
            for state in refine_pool:
                if not state.get("invalid_move"):
                    refined.append(state)
                    continue

                repaired = state
                for _ in range(max(1, config.num_tries)):
                    improve_prompt = build_improve_prompt(
                        prev_items_json=state.get(
                            "prev_items_json", state.get("items_json", "[]")
                        ),
                        last_move=state.get("last_move"),
                    )
                    improve_call = call_llm_for_moves(
                        client=client,
                        prompt=improve_prompt,
                        operation="improve",
                        temperature=temperature,
                    )
                    llm_calls.append(improve_call)
                    if not improve_call.moves:
                        continue

                    base_state = _restore_previous_state(state)
                    repaired = apply_move(base_state, improve_call.moves[0])
                    if not repaired.get("invalid_move"):
                        break
                refined.append(repaired)

            if not refined:
                break

            best_terminal_state = _best_terminal(refined)
            if best_terminal_state is not None:
                best_state = best_terminal_state
                break

            frontier = _score_then_keep(refined, config.beam_width)
        else:
            frontier = _score_then_keep(generated, config.beam_width)

        if not frontier:
            break
        best_state = frontier[0]

    aggregate_usage = _aggregate_usage(llm_calls)
    return SearchResult(
        variant=variant,
        best_state=best_state,
        best_terminal_state=best_terminal_state,
        solved=best_terminal_state is not None,
        llm_calls=llm_calls,
        aggregate_usage=aggregate_usage,
    )
