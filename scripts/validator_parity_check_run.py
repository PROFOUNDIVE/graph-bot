from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_OF_THOUGHTS_ROOT = REPO_ROOT.parent / "graph-of-thoughts"
GRAPH_OF_THOUGHTS_TASKS = GRAPH_OF_THOUGHTS_ROOT / "tasks"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GRAPH_OF_THOUGHTS_ROOT))
sys.path.insert(0, str(GRAPH_OF_THOUGHTS_TASKS))

if "backoff" not in sys.modules:
    backoff_stub = types.ModuleType("backoff")

    def _expo(*_args, **_kwargs):
        return None

    def _on_exception(*_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

    setattr(backoff_stub, "expo", _expo)
    setattr(backoff_stub, "on_exception", _on_exception)
    sys.modules["backoff"] = backoff_stub

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class OpenAI:  # noqa: N801
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("OpenAI client is not available in parity script")

    class OpenAIError(Exception):
        pass

    setattr(openai_stub, "OpenAI", OpenAI)
    setattr(openai_stub, "OpenAIError", OpenAIError)
    sys.modules["openai"] = openai_stub

    types_stub = types.ModuleType("openai.types")
    chat_stub = types.ModuleType("openai.types.chat")
    chat_completion_stub = types.ModuleType("openai.types.chat.chat_completion")

    class ChatCompletion:  # noqa: N801
        pass

    setattr(chat_completion_stub, "ChatCompletion", ChatCompletion)
    sys.modules["openai.types"] = types_stub
    sys.modules["openai.types.chat"] = chat_stub
    sys.modules["openai.types.chat.chat_completion"] = chat_completion_stub

from graph_bot.eval.validators import get_validator  # noqa: E402


class UtilsModule(Protocol):
    def test_game24(self, state: Dict) -> bool: ...


def _load_got_modules() -> Tuple[type, UtilsModule]:
    gameof24_path = GRAPH_OF_THOUGHTS_TASKS / "gameof24.py"
    utils_path = GRAPH_OF_THOUGHTS_TASKS / "utils.py"

    gameof24_spec = importlib.util.spec_from_file_location(
        "got_gameof24", gameof24_path
    )
    utils_spec = importlib.util.spec_from_file_location("got_utils", utils_path)
    if gameof24_spec is None or gameof24_spec.loader is None:
        raise RuntimeError("Failed to load gameof24 module")
    if utils_spec is None or utils_spec.loader is None:
        raise RuntimeError("Failed to load utils module")

    gameof24_module = importlib.util.module_from_spec(gameof24_spec)
    utils_module = importlib.util.module_from_spec(utils_spec)
    sys.modules["got_gameof24"] = gameof24_module
    sys.modules["got_utils"] = utils_module
    gameof24_spec.loader.exec_module(gameof24_module)
    utils_spec.loader.exec_module(utils_module)

    parser_class = getattr(gameof24_module, "Gameof24ExpressionParser")
    return parser_class, cast(UtilsModule, utils_module)


class OracleValidate(Protocol):
    def __call__(self, answer: str, problem: str) -> bool: ...


def _load_cases(path: Path, limit: int) -> List[Dict]:
    cases: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            cases.append(json.loads(line))
            if len(cases) >= limit:
                break
    if len(cases) < limit:
        raise ValueError(f"Expected {limit} cases, found {len(cases)}")
    return cases


def _solve_24(numbers: List[int]) -> Optional[str]:
    target = Fraction(24, 1)
    terms: List[Tuple[Fraction, str]] = [(Fraction(n), str(n)) for n in numbers]

    def _search(items: List[Tuple[Fraction, str]]) -> Optional[str]:
        if len(items) == 1:
            return items[0][1] if items[0][0] == target else None

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a_val, a_expr = items[i]
                b_val, b_expr = items[j]
                rest = [items[k] for k in range(len(items)) if k not in (i, j)]

                candidates: List[Tuple[Fraction, str]] = [
                    (a_val + b_val, f"({a_expr}+{b_expr})"),
                    (a_val * b_val, f"({a_expr}*{b_expr})"),
                    (a_val - b_val, f"({a_expr}-{b_expr})"),
                    (b_val - a_val, f"({b_expr}-{a_expr})"),
                ]
                if b_val != 0:
                    candidates.append((a_val / b_val, f"({a_expr}/{b_expr})"))
                if a_val != 0:
                    candidates.append((b_val / a_val, f"({b_expr}/{a_expr})"))

                for value, expr in candidates:
                    result = _search(rest + [(value, expr)])
                    if result is not None:
                        return result
        return None

    return _search(terms)


def _validate_with_got(expression: str, numbers: List[int]) -> bool:
    parser_class, utils_module = _load_got_modules()
    parser = parser_class()
    base_state = {
        "original": " ".join(str(n) for n in numbers),
        "current": "",
    }
    text = f"Output: {expression} = 24"
    states = parser.parse_generate_answer(base_state, [text])
    if not states:
        return False
    return utils_module.test_game24(states[0])


def _pick_invalid_expression(
    numbers: List[int],
    oracle_validate: OracleValidate,
    problem_str: str,
) -> str:
    a, b, c, d = numbers
    templates = [
        f"{a}+{b}+{c}+{d}",
        f"{a}*{b}+{c}+{d}",
        f"{a}+{b}*{c}+{d}",
        f"{a}+{b}+{c}*{d}",
        f"({a}+{b}+{c})*{d}",
        f"({a}+{b})*({c}+{d})",
        f"({a}-{b})+{c}+{d}",
    ]
    if b != 0:
        templates.append(f"({a}/{b})+{c}+{d}")
    if d != 0:
        templates.append(f"({a}+{b}+{c})/{d}")

    for expr in templates:
        if not oracle_validate(expr, problem_str):
            return expr
    return f"{a}+{a}+{b}+{c}"


def _problem_string(numbers: List[int]) -> str:
    return f"{' '.join(str(n) for n in numbers)} -> 24"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validator parity check")
    parser.add_argument(
        "--data-path",
        default="data/game24_buffer_98.shuffle_1.jsonl",
        help="Path to Game24 JSONL data",
    )
    parser.add_argument(
        "--out-path",
        default="outputs/validator_parity_report.json",
        help="Path to write report JSON",
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    data_path = REPO_ROOT / args.data_path
    out_path = REPO_ROOT / args.out_path
    cases = _load_cases(data_path, args.limit)

    oracle = get_validator("oracle")
    report_cases = []
    mismatches = 0

    for case in cases:
        numbers = case.get("numbers")
        if not isinstance(numbers, list) or len(numbers) != 4:
            raise ValueError(f"Invalid numbers for case: {case}")
        numbers_int = [int(n) for n in numbers]
        problem_str = _problem_string(numbers_int)

        valid_expr = _solve_24(numbers_int)
        invalid_expr = _pick_invalid_expression(
            numbers_int, oracle.validate, problem_str
        )

        oracle_valid = oracle.validate(valid_expr, problem_str) if valid_expr else None
        oracle_invalid = oracle.validate(invalid_expr, problem_str)
        got_valid = _validate_with_got(valid_expr, numbers_int) if valid_expr else None
        got_invalid = _validate_with_got(invalid_expr, numbers_int)

        mismatch = (oracle_valid != got_valid) or (oracle_invalid != got_invalid)
        if mismatch:
            mismatches += 1

        report_cases.append(
            {
                "id": case.get("id"),
                "numbers": numbers_int,
                "valid_expression": valid_expr,
                "invalid_expression": invalid_expr,
                "oracle": {"valid": oracle_valid, "invalid": oracle_invalid},
                "graph_of_thoughts": {"valid": got_valid, "invalid": got_invalid},
                "mismatch": mismatch,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "total": len(report_cases),
        "mismatches": mismatches,
        "cases": report_cases,
    }
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
