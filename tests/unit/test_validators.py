from __future__ import annotations
from types import SimpleNamespace

from graph_bot.datatypes import ReasoningNode
from graph_bot.eval.validators import (
    Game24Validator,
    WeakLLMJudgeValidator,
    extract_game24_problem_numbers,
    extract_game24_expression_number_literals,
)


def test_extract_game24_problem_numbers():
    assert extract_game24_problem_numbers("2 4 6 8 → 24") == [2, 4, 6, 8, 24]
    assert extract_game24_problem_numbers("1 2 3 4") == [1, 2, 3, 4]
    assert extract_game24_problem_numbers("2 5 8 11") == [2, 5, 8, 11]
    assert extract_game24_problem_numbers("no numbers here") == []


def test_extract_game24_expression_number_literals():
    assert extract_game24_expression_number_literals("(2 + 4) * (6 - 2)") == [
        2,
        4,
        6,
        2,
    ]
    assert extract_game24_expression_number_literals("24") == [24]
    assert extract_game24_expression_number_literals("invalid") is None
    assert extract_game24_expression_number_literals("os.system('ls')") is None
    assert extract_game24_expression_number_literals("1.5 + 2") is None


def test_game24_validator_safe_eval():
    validator = Game24Validator()
    assert validator._safe_eval("(2 + 4) * 4") == 24.0
    assert validator._safe_eval("10 / 2 + 19") == 24.0
    assert validator._safe_eval("3 * (1 + 7)") == 24.0

    assert validator._safe_eval("24 / 0") is None

    assert validator._safe_eval("__import__('os').system('ls')") is None
    assert validator._safe_eval("1; import os; os.system('ls')") is None


def test_game24_validator_validate():
    validator = Game24Validator()
    problem = "2 4 6 8 → 24"

    assert validator.validate("8 * (6 / (4 - 2))", problem) is True
    assert validator.validate("8 * 6 / 4 + 12", "8 6 4 12 → 24") is True

    assert validator.validate("2 + 4 + 6 + 8", problem) is False

    assert validator.validate("8 * (6 / (4 - 3))", problem) is False
    assert validator.validate("24", problem) is False

    assert validator.validate("8 / (1 - 2/3)", "8 1 2 3 → 24") is True


def test_game24_validator_failure_reason():
    validator = Game24Validator()
    problem = "2 4 6 8 → 24"

    assert validator.failure_reason("8 * (6 / (4 - 2))", problem) is None
    assert validator.failure_reason("invalid expr", problem) == "format_error"
    assert validator.failure_reason("2 + 4 + 6 + 8", problem) == "math_error"
    assert validator.failure_reason("8 * (6 / (4 - 3))", problem) == "wrong_numbers"


class _StubJudgeClient:
    def __init__(
        self,
        response_text: str = "YES",
        exc: Exception | None = None,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self._response_text = response_text
        self._exc = exc
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._total_tokens = total_tokens

    def chat(self, system: str, user: str, temperature: float):
        del system, user, temperature
        if self._exc is not None:
            raise self._exc
        usage = SimpleNamespace(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._total_tokens,
        )
        return self._response_text, usage


def _make_judge_node(answer: str, client: _StubJudgeClient) -> ReasoningNode:
    return ReasoningNode(
        node_id="n1",
        text=answer,
        type="answer",
        attributes={"problem": "2 4 6 8 -> 24", "llm_client": client},
    )


def test_llm_judge_validator_parse_error_sets_failure_reason():
    validator = WeakLLMJudgeValidator(model="mock-model")
    node = _make_judge_node(
        "(8 * 6) / (4 - 2)",
        _StubJudgeClient(response_text="MAYBE\nReason: uncertain"),
    )

    assert validator.validate(node) == 0.0
    assert validator.failure_reason(node.text, "2 4 6 8 -> 24") == "judge_parse_error"


def test_llm_judge_validator_fallback_yes_no_parse_from_body_text():
    validator = WeakLLMJudgeValidator(model="mock-model")
    node = _make_judge_node(
        "2 + 4 + 6 + 8",
        _StubJudgeClient(
            response_text="The correct decision is NO.\nReason: arithmetic is wrong"
        ),
    )

    assert validator.validate(node) == 0.0
    assert validator.failure_reason(node.text, "2 4 6 8 -> 24") == "arithmetic is wrong"


def test_llm_judge_validator_no_reason_defaults_to_llm_judge_rejected():
    validator = WeakLLMJudgeValidator(model="mock-model")
    node = _make_judge_node("2 + 4 + 6 + 8", _StubJudgeClient(response_text="NO"))

    assert validator.validate(node) == 0.0
    assert validator.failure_reason(node.text, "2 4 6 8 -> 24") == "llm_judge_rejected"


def test_llm_judge_validator_logs_token_event_when_metrics_present(monkeypatch):
    class _Metrics:
        def __init__(self) -> None:
            self.events: list[dict] = []
            self.run_id = "test_metrics_run"

        def log_token_event(self, event: dict) -> None:
            self.events.append(event)

        def new_call_id(self) -> str:
            return "call-1"

    metrics = _Metrics()
    validator = WeakLLMJudgeValidator(model="mock-model")
    monkeypatch.setattr(validator, "_load_pricing_table", lambda: None)

    client = _StubJudgeClient(
        response_text="YES\nReason: correct",
        prompt_tokens=12,
        completion_tokens=34,
        total_tokens=46,
    )
    node = ReasoningNode(
        node_id="n1",
        text="(8 * 6) / (4 - 2)",
        type="answer",
        attributes={
            "problem": "2 4 6 8 -> 24",
            "llm_client": client,
            "metrics": metrics,
            "stream_run_id": "test_stream_run",
            "problem_id": "p1",
            "t": 1,
        },
    )

    assert validator.validate(node) == 1.0
    assert len(metrics.events) == 1
    event = metrics.events[0]
    assert event["operation"] == "validate_llm_judge"
    assert event["status"] == "success"
    assert event["usage"]["prompt_tokens"] == 12
    assert event["usage"]["completion_tokens"] == 34
    assert event["usage"]["total_tokens"] == 46


def test_llm_judge_validator_correct_answer():
    validator = WeakLLMJudgeValidator(model="mock-model")
    node = _make_judge_node(
        "(8 * 6) / (4 - 2)", _StubJudgeClient(response_text="YES\nReason: correct")
    )

    assert validator.validate(node) == 1.0
    assert validator.failure_reason(node.text, "2 4 6 8 -> 24") is None


def test_llm_judge_validator_wrong_answer():
    validator = WeakLLMJudgeValidator(model="mock-model")
    node = _make_judge_node(
        "2 + 4 + 6 + 8",
        _StubJudgeClient(response_text="NO\nReason: arithmetic is wrong"),
    )

    assert validator.validate(node) == 0.0
    assert validator.failure_reason(node.text, "2 4 6 8 -> 24") == "arithmetic is wrong"


def test_llm_judge_validator_timeout_returns_false():
    validator = WeakLLMJudgeValidator(model="mock-model")
    node = _make_judge_node(
        "(8 * 6) / (4 - 2)", _StubJudgeClient(exc=TimeoutError("request timeout"))
    )

    assert validator.validate(node) == 0.0
    assert (
        validator.failure_reason(node.text, "2 4 6 8 -> 24")
        == "judge_error:TimeoutError"
    )


def test_llm_judge_validator_api_error_returns_false():
    validator = WeakLLMJudgeValidator(model="mock-model")
    node = _make_judge_node(
        "(8 * 6) / (4 - 2)", _StubJudgeClient(exc=RuntimeError("upstream 500"))
    )

    assert validator.validate(node) == 0.0
    assert (
        validator.failure_reason(node.text, "2 4 6 8 -> 24")
        == "judge_error:RuntimeError"
    )
