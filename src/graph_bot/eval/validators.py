from __future__ import annotations

import ast
import datetime
import operator
import re
import time
import uuid
from abc import abstractmethod
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

from ..datatypes import ReasoningNode
from ..interfaces import AbstractValidator
from ..logsetting import logger
from ..tools.python_executor import run_python


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


class BaseValidator(AbstractValidator):
    """Base class for answer validation."""

    @abstractmethod
    def validate(
        self, node: str | ReasoningNode, problem: str | None = None
    ) -> bool | float:
        pass

    def failure_reason(self, answer: str, problem: str) -> str | None:
        del answer
        del problem
        return None

    def get_failure_reason(self, node: ReasoningNode) -> Optional[str]:
        problem = node.attributes.get("problem") if node.attributes else None
        if not problem:
            return "missing_problem_context"
        return self.failure_reason(node.text, problem)

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

    def validate(
        self, node: str | ReasoningNode, problem: str | None = None
    ) -> bool | float:
        if isinstance(node, ReasoningNode):
            problem_str = node.attributes.get("problem") if node.attributes else None
            if not problem_str:
                logger.debug(
                    "Validation failed: missing problem context in node attributes"
                )
                return 0.0
            return 1.0 if self._validate_answer(node.text, problem_str) else 0.0

        if problem is None:
            logger.warning(
                "Validation failed: problem string is required for legacy validate call"
            )
            return False
        return self._validate_answer(node, problem)

    def _validate_answer(self, answer: str, problem: str) -> bool:
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

    def __init__(self) -> None:
        self._last_failure_reason: str | None = None

    def failure_reason(self, answer: str, problem: str) -> str | None:
        del answer
        del problem
        return self._last_failure_reason

    @staticmethod
    def _extract_python_block(raw_output: str) -> str | None:
        match = re.search(r"```python\s*(.*?)```", raw_output, flags=re.DOTALL)
        if not match:
            return None
        code = match.group(1).strip()
        return code or None

    @staticmethod
    def _first_nonempty_line(text: str) -> str | None:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    @staticmethod
    def _normalize_numeric_text(raw: str) -> str | None:
        text = raw.strip().replace(",", "")
        if not text:
            return None
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
            return None
        try:
            value = Decimal(text)
        except InvalidOperation:
            return None
        normalized = format(value, "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        if normalized in {"", "-0", "+0"}:
            return "0"
        return normalized

    @staticmethod
    def _normalize_whitespace_text(raw: str) -> str:
        return " ".join(raw.strip().split())

    def validate(
        self, node: str | ReasoningNode, problem: str | None = None
    ) -> bool | float:
        """Validate using execution and repair logic."""
        del problem

        if not isinstance(node, ReasoningNode):
            self._last_failure_reason = "missing_raw_output"
            return 0.0

        raw_output = node.attributes.get("raw_output") if node.attributes else None
        if not raw_output:
            self._last_failure_reason = "missing_raw_output"
            return 0.0

        if not node.attributes or not node.attributes.get("problem"):
            self._last_failure_reason = "missing_raw_output"
            return 0.0

        code = self._extract_python_block(raw_output)
        if code is None:
            self._last_failure_reason = "exec_no_output"
            return 0.0

        exec_result = run_python(
            code,
            timeout_sec=3.0,
            max_stdout_chars=1000,
            max_stderr_chars=1000,
        )

        if exec_result.timed_out:
            self._last_failure_reason = "exec_timeout"
            return 0.0

        if (
            exec_result.exit_code != 0
            and "Execution blocked: banned token detected:" in exec_result.stderr
        ):
            self._last_failure_reason = "exec_banned_token"
            return 0.0

        if exec_result.exit_code != 0:
            self._last_failure_reason = "exec_runtime_error"
            return 0.0

        output_line = self._first_nonempty_line(exec_result.stdout)
        if output_line is None:
            self._last_failure_reason = "exec_no_output"
            return 0.0

        exec_answer = re.sub(
            r"^ANSWER\s*:\s*", "", output_line, flags=re.IGNORECASE
        ).strip()
        if not exec_answer:
            self._last_failure_reason = "exec_no_output"
            return 0.0

        candidate = node.text.strip()
        exec_numeric = self._normalize_numeric_text(exec_answer)
        candidate_numeric = self._normalize_numeric_text(candidate)

        if exec_numeric is not None and candidate_numeric is not None:
            is_match = exec_numeric == candidate_numeric
        else:
            is_match = self._normalize_whitespace_text(exec_answer) == (
                self._normalize_whitespace_text(candidate)
            )

        if is_match:
            self._last_failure_reason = None
            return 1.0

        self._last_failure_reason = "exec_mismatch"
        return 0.0

    def get_validator_name(self) -> str:
        return "exec_repair"


class WeakLLMJudgeValidator(BaseValidator):
    """Weak validator using LLM judgment for domains without ground truth."""

    def __init__(self, model: str | None = None) -> None:
        from ..settings import settings

        settings_model = (
            getattr(settings, "validator_model", None) or settings.llm_model
        )
        self._model = model or settings_model
        self._last_failure_reason: str | None = None
        self._pricing_table: dict[str, Any] | None = None

    def failure_reason(self, answer: str, problem: str) -> str | None:
        del answer
        del problem
        return self._last_failure_reason

    def _load_pricing_table(self) -> dict[str, Any] | None:
        if self._pricing_table is not None:
            return self._pricing_table

        try:
            from ..settings import settings
            from ..utils.pricing import load_pricing_table

            self._pricing_table = load_pricing_table(settings.pricing_path)
        except Exception:
            self._pricing_table = None

        return self._pricing_table

    @staticmethod
    def _parse_judge_output(text: str) -> tuple[bool | None, str | None]:
        raw = (text or "").strip()
        if not raw:
            return None, None

        first_nonempty_line = None
        for line in raw.splitlines():
            if line.strip():
                first_nonempty_line = line.strip()
                break

        if first_nonempty_line:
            m = re.match(r"^(YES|NO)\b", first_nonempty_line, flags=re.IGNORECASE)
            if m:
                decision = m.group(1).upper()
                remainder = first_nonempty_line[m.end() :].strip(" -:\t")
                reason = remainder or None

                # If reason isn't on the decision line, try to find a Reason: line.
                if reason is None:
                    m_reason = re.search(
                        r"^\s*Reason\s*:\s*(.+)$",
                        raw,
                        flags=re.IGNORECASE | re.MULTILINE,
                    )
                    if m_reason:
                        reason = m_reason.group(1).strip() or None

                return decision == "YES", reason

        # Fallback: find first YES/NO occurrence anywhere.
        m_any = re.search(r"\b(YES|NO)\b", raw, flags=re.IGNORECASE)
        if not m_any:
            return None, None
        decision = m_any.group(1).upper()

        m_reason = re.search(
            r"\bReason\s*:\s*(.+)$", raw, flags=re.IGNORECASE | re.MULTILINE
        )
        reason = m_reason.group(1).strip() if m_reason else None
        return decision == "YES", reason

    def _maybe_log_token_event(
        self,
        *,
        attributes: dict[str, Any] | None,
        status: str,
        latency_ms: float,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float,
        error_type: str | None,
    ) -> None:
        if not attributes:
            return

        metrics = attributes.get("metrics")
        if metrics is None:
            return

        if not hasattr(metrics, "log_token_event"):
            return

        stream_run_id = attributes.get("stream_run_id") or getattr(
            metrics, "run_id", None
        )
        problem_id = attributes.get("problem_id")
        t = attributes.get("t")
        if stream_run_id is None or problem_id is None or t is None:
            return

        span_id = None
        if hasattr(metrics, "new_call_id"):
            try:
                span_id = metrics.new_call_id()
            except Exception:
                span_id = None
        if span_id is None:
            span_id = str(uuid.uuid4())

        event: dict[str, Any] = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "stream_run_id": stream_run_id,
            "problem_id": problem_id,
            "t": t,
            "event_type": "llm_completion",
            "operation": "validate_llm_judge",
            "status": status,
            "model": model,
            "latency_ms": int(round(latency_ms)),
            "run_id": f"{stream_run_id}:{problem_id}",
            "span_id": span_id,
            "component": "evaluator",
            "metadata": {"pricing_version": "v0", "error_type": error_type},
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
            },
            "cost_usd": float(cost_usd),
        }

        try:
            metrics.log_token_event(event)
        except Exception:
            return

    def validate(
        self, node: str | ReasoningNode, problem: str | None = None
    ) -> bool | float:
        """Validate using LLM as judge."""
        attributes: dict[str, Any] | None = None
        if isinstance(node, ReasoningNode):
            attributes = node.attributes
            problem_str = node.attributes.get("problem") if node.attributes else None
            if not problem_str:
                logger.debug(
                    "Validation failed: missing problem context in node attributes"
                )
                self._last_failure_reason = "missing_problem_context"
                return 0.0
            answer = node.text
            problem = problem_str
        else:
            answer = node
            if problem is None:
                logger.warning(
                    "Validation failed: problem string is required for legacy validate call"
                )
                self._last_failure_reason = "missing_problem_context"
                return 0.0

        from ..settings import settings

        judge_system = (
            "You are a strict evaluator.\n"
            "Given a Problem and a Proposed Answer, decide whether the answer is correct and adequately addresses the problem.\n"
            "Reply with exactly one of the following formats:\n"
            "YES\nReason: <short reason>\n"
            "NO\nReason: <short reason>\n"
            "Do not output anything else."
        )
        judge_user = f"Problem:\n{problem}\n\nProposed Answer:\n{answer}\n"

        # Allow injection for tests / callers.
        injected_client = attributes.get("llm_client") if attributes else None

        if injected_client is not None:
            client = injected_client
        elif settings.llm_provider == "mock":
            from ..adapters.mock_client import MockLLMClient

            client = MockLLMClient(model=self._model)
        else:
            from ..adapters.vllm_openai_client import VLLMOpenAIClient

            client = VLLMOpenAIClient(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=self._model,
            )

        start = time.perf_counter()
        usage = None
        raw = ""
        error_type = None
        try:
            raw, usage = client.chat(
                system=judge_system, user=judge_user, temperature=0.0
            )
        except Exception as exc:
            error_type = type(exc).__name__
        latency_ms = (time.perf_counter() - start) * 1000.0

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        cost_usd = 0.0
        pricing_table = self._load_pricing_table()
        if pricing_table is not None:
            try:
                from ..utils.pricing import calculate_cost

                cost_usd = float(
                    calculate_cost(
                        pricing_table=pricing_table,
                        model_name=self._model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                )
            except Exception:
                cost_usd = 0.0

        if error_type is not None:
            self._last_failure_reason = f"judge_error:{error_type}"
            self._maybe_log_token_event(
                attributes=attributes,
                status="failed",
                latency_ms=latency_ms,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                error_type=error_type,
            )
            return 0.0

        decision, reason = self._parse_judge_output(raw)
        if decision is None:
            self._last_failure_reason = "judge_parse_error"
            self._maybe_log_token_event(
                attributes=attributes,
                status="failed",
                latency_ms=latency_ms,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                error_type="parse_error",
            )
            return 0.0

        if decision:
            self._last_failure_reason = None
            self._maybe_log_token_event(
                attributes=attributes,
                status="success",
                latency_ms=latency_ms,
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                error_type=None,
            )
            return 1.0

        self._last_failure_reason = (reason or "llm_judge_rejected").strip()
        self._maybe_log_token_event(
            attributes=attributes,
            status="success",
            latency_ms=latency_ms,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            error_type=None,
        )
        return 0.0

    def get_validator_name(self) -> str:
        return "weak_llm_judge"


_VALIDATOR_REGISTRY: Dict[str, type[BaseValidator]] = {
    "oracle": Game24Validator,
    "exec_repair": ExecRepairValidator,
    "weak_llm_judge": WeakLLMJudgeValidator,
}


def get_validator(mode: str, model: str | None = None) -> BaseValidator:
    """Factory function to get validator instance.

    Args:
        mode: Validator mode (oracle, exec_repair, weak_llm_judge)

    Returns:
        Validator instance
    """
    validator_class = _VALIDATOR_REGISTRY.get(mode)
    if validator_class is None:
        raise ValueError(f"Unknown validator mode: {mode}")
    # Special-case WeakLLMJudgeValidator to allow passing a model explicitly without
    # triggering type-checker issues for other validators.
    if mode == "weak_llm_judge":
        return WeakLLMJudgeValidator(model)
    # For other validators, rely on default no-arg constructors to preserve
    # backward compatibility.
    return validator_class()
