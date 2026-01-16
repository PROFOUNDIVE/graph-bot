from __future__ import annotations

import ast
import operator
import re
from abc import ABC, abstractmethod
from typing import Dict

from ..logsetting import logger


class BaseValidator(ABC):
    """Base class for answer validation."""

    @abstractmethod
    def validate(self, answer: str, problem: str) -> bool:
        """Validate answer for given problem."""
        pass

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

    def __init__(self):
        self._pattern = re.compile(r"(-?\d+\.?\d*)")

    def validate(self, answer: str, problem: str) -> bool:
        """Validate Game24 answer.

        Args:
            answer: Arithmetic expression (e.g., "(2 + 5) * 8 / 4")
            problem: Problem statement (e.g., "2 5 8 4 → 24")

        Returns:
            True if answer is valid and evaluates to 24
        """
        try:
            problem_numbers = self._extract_numbers(problem)
            answer_numbers = self._extract_numbers(answer)

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

    def _extract_numbers(self, text: str) -> list[float]:
        """Extract all numbers from text as floats."""
        matches = self._pattern.findall(text)
        return [float(m) for m in matches]

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
