from __future__ import annotations

import ast
import operator
from abc import ABC, abstractmethod
from typing import Dict

from ..logsetting import logger


def extract_game24_problem_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for token in text.replace("→", " ").split():
        try:
            value = int(float(token))
        except ValueError:
            continue
        numbers.append(abs(value))
    return numbers


def extract_game24_expression_number_literals(expr: str) -> list[int] | None:
    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return None

    allowed_binop_types = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    numbers: list[int] = []

    def _walk(n: ast.AST) -> None:
        if isinstance(n, ast.Expression):
            _walk(n.body)
            return

        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool):
                raise ValueError("invalid constant")
            if isinstance(n.value, int):
                numbers.append(abs(n.value))
                return
            if isinstance(n.value, float):
                if float(n.value).is_integer():
                    numbers.append(abs(int(n.value)))
                    return
                raise ValueError("non-integer literal")
            raise ValueError("invalid constant")

        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub):
                _walk(n.operand)
                return
            raise ValueError("invalid unary")

        if isinstance(n, ast.BinOp):
            if not isinstance(n.op, allowed_binop_types):
                raise ValueError("invalid op")
            _walk(n.left)
            _walk(n.right)
            return

        raise ValueError("invalid node")

    try:
        _walk(node)
    except Exception:
        return None

    return numbers


class BaseValidator(ABC):
    """Base class for answer validation."""

    @abstractmethod
    def validate(self, answer: str, problem: str) -> bool:
        """Validate answer for given problem."""
        pass

    def failure_reason(self, answer: str, problem: str) -> str | None:
        del answer
        del problem
        return None

    @abstractmethod
    def get_validator_name(self) -> str:
        """Return validator identifier for logging."""
        pass


class Game24Validator(BaseValidator):
    """Validator for Game of 24 problems.

    Validates that:
    1. Expression evaluates to 24
    2. All numbers from problem are used exactly once
    3. Only valid operations (+, -, *, /, parentheses) are used
    4. Division by zero is avoided
    """

    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def failure_reason(self, answer: str, problem: str) -> str | None:
        try:
            problem_numbers = extract_game24_problem_numbers(problem)
            if problem_numbers and problem_numbers[-1] == 24:
                problem_numbers = problem_numbers[:-1]

            answer_numbers = extract_game24_expression_number_literals(answer)
            if answer_numbers is None:
                return "format_error"

            if sorted(problem_numbers) != sorted(answer_numbers):
                return "wrong_numbers"

            result = self._safe_eval(answer)
            if result is None:
                return "format_error"

            if abs(result - 24.0) >= 1e-6:
                return "math_error"

            return None
        except Exception:
            return "format_error"

    def validate(self, answer: str, problem: str) -> bool:
        try:
            problem_numbers = extract_game24_problem_numbers(problem)
            if problem_numbers and problem_numbers[-1] == 24:
                problem_numbers = problem_numbers[:-1]

            answer_numbers = extract_game24_expression_number_literals(answer)
            if answer_numbers is None:
                logger.debug(f"Validation failed: cannot parse literals: {answer}")
                return False

            if sorted(problem_numbers) != sorted(answer_numbers):
                logger.debug(
                    f"Validation failed: numbers mismatch. "
                    f"Problem: {problem_numbers}, Answer: {answer_numbers}"
                )
                return False

            result = self._safe_eval(answer)
            if result is None:
                logger.debug(f"Validation failed: cannot evaluate answer: {answer}")
                return False

            is_valid = abs(result - 24.0) < 1e-6
            if not is_valid:
                logger.debug(f"Validation failed: result {result} != 24")
            return is_valid

        except Exception as e:
            logger.warning(f"Validation error for answer '{answer}': {e}")
            return False

    def get_validator_name(self) -> str:
        return "oracle"

    def _safe_eval(self, expr: str) -> float | None:
        """Safely evaluate arithmetic expression.

        Only allows: +, -, *, /, parentheses, and numbers.
        """
        try:
            node = ast.parse(expr, mode="eval")

            def _eval(node):
                if isinstance(node, ast.Constant):
                    if isinstance(node.value, (int, float)):
                        return float(node.value)
                    raise ValueError(f"Invalid constant: {node.value}")

                if isinstance(node, ast.BinOp):
                    left = _eval(node.left)
                    right = _eval(node.right)
                    op_type = type(node.op)
                    if op_type in self._OPS:
                        op_func = self._OPS[op_type]
                        if op_type == ast.Div and abs(right) < 1e-9:
                            raise ValueError("Division by zero")
                        return op_func(left, right)
                    raise ValueError(f"Invalid operation: {op_type}")

                if isinstance(node, ast.UnaryOp):
                    if isinstance(node.op, ast.USub):
                        return -_eval(node.operand)
                    raise ValueError(f"Invalid unary op: {type(node.op)}")

                if isinstance(node, ast.Expression):
                    return _eval(node.body)

                raise ValueError(f"Invalid AST node: {type(node)}")

            return float(_eval(node))

        except Exception as e:
            logger.warning(f"Failed to evaluate expression '{expr}': {e}")
            return None


class ExecRepairValidator(BaseValidator):
    """Validator for code-augmented tasks using execution and repair loop."""

    def validate(self, answer: str, problem: str) -> bool:
        """Validate using execution and repair logic."""
        raise NotImplementedError("ExecRepairValidator not yet implemented")

    def get_validator_name(self) -> str:
        return "exec_repair"


class WeakLLMJudgeValidator(BaseValidator):
    """Weak validator using LLM judgment for domains without ground truth."""

    def validate(self, answer: str, problem: str) -> bool:
        """Validate using LLM as judge."""
        raise NotImplementedError("WeakLLMJudgeValidator not yet implemented")

    def get_validator_name(self) -> str:
        return "weak_llm_judge"


_VALIDATOR_REGISTRY: Dict[str, type[BaseValidator]] = {
    "oracle": Game24Validator,
    "exec_repair": ExecRepairValidator,
    "weak_llm_judge": WeakLLMJudgeValidator,
}


def get_validator(mode: str) -> BaseValidator:
    """Factory function to get validator instance.

    Args:
        mode: Validator mode (oracle, exec_repair, weak_llm_judge)

    Returns:
        Validator instance
    """
    validator_class = _VALIDATOR_REGISTRY.get(mode)
    if validator_class is None:
        raise ValueError(f"Unknown validator mode: {mode}")
    return validator_class()
