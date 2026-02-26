from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any, Dict, List, Tuple

TARGET = 24.0
EPS = 1e-6


def _round_key(vals: List[float], ndigits: int = 6) -> Tuple[float, ...]:
    return tuple(sorted(round(v, ndigits) for v in vals))


@lru_cache(maxsize=200_000)
def _best_residual(key: Tuple[float, ...]) -> float:
    vals = list(key)
    n = len(vals)

    if n == 0:
        return 1e9
    if n == 1:
        v = vals[0]
        if not math.isfinite(v):
            return 1e9
        return abs(v - TARGET)

    best = 1e9
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vals[i], vals[j]
            rest = [vals[k] for k in range(n) if k != i and k != j]
            candidates = [a + b, a * b, a - b, b - a]
            if abs(b) > EPS:
                candidates.append(a / b)
            if abs(a) > EPS:
                candidates.append(b / a)

            for c in candidates:
                if not math.isfinite(c):
                    continue
                if abs(c) > 1e6:
                    continue

                nxt = rest + [c]
                residual = _best_residual(_round_key(nxt))
                if residual < best:
                    best = residual
                    if best <= 1e-9:
                        return 0.0

    return best


def game24_score(state: Dict[str, Any]) -> float:
    logging.debug("game24_utils > game24_score called")
    try:
        if state.get("invalid_move"):
            return 1e9

        items = state.get("items", [])
        if not isinstance(items, list) or len(items) == 0:
            return 1e9

        vals: List[float] = []
        mag_penalty = 0.0
        for item in items:
            value = float(item.get("value", 0.0))
            if not math.isfinite(value):
                return 1e9
            vals.append(value)
            mag_penalty += max(0.0, abs(value) - 100.0)

        residual = _best_residual(_round_key(vals))
        depth_tiebreak = 0.05 * (len(items) - 1)
        return float(residual) + depth_tiebreak + 0.001 * float(mag_penalty)
    except Exception:
        return 1e9


def test_game24(state: Dict[str, Any]) -> bool:
    logging.debug("game24_utils > test_game24 called")
    try:
        if state.get("invalid_move"):
            return False
        items = state.get("items", [])
        if not isinstance(items, list) or len(items) != 1:
            return False
        value = float(items[0].get("value", 0.0))
        return abs(value - 24.0) < 1e-6
    except Exception:
        return False
