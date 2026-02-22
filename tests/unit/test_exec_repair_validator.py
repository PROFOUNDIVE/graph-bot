from __future__ import annotations

from graph_bot.datatypes import ReasoningNode
from graph_bot.eval.validators import ExecRepairValidator


def _make_node(
    candidate: str, raw_output: str, problem: str = "dummy problem"
) -> ReasoningNode:
    return ReasoningNode(
        node_id="n1",
        text=candidate,
        type="answer",
        attributes={"problem": problem, "raw_output": raw_output},
    )


def test_exec_repair_validator_happy_path() -> None:
    validator = ExecRepairValidator()
    node = _make_node(
        candidate="42",
        raw_output="""Some reasoning
```python
print("ANSWER: 42.0")
```
""",
    )

    assert validator.validate(node) == 1.0
    assert validator.failure_reason(node.text, "dummy problem") is None


def test_exec_repair_validator_timeout() -> None:
    validator = ExecRepairValidator()
    node = _make_node(
        candidate="anything",
        raw_output="""```python
while True:
    pass
```
""",
    )

    assert validator.validate(node) == 0.0
    assert validator.failure_reason(node.text, "dummy problem") == "exec_timeout"


def test_exec_repair_validator_runtime_error() -> None:
    validator = ExecRepairValidator()
    node = _make_node(
        candidate="1",
        raw_output="""```python
raise RuntimeError("boom")
```
""",
    )

    assert validator.validate(node) == 0.0
    assert validator.failure_reason(node.text, "dummy problem") == "exec_runtime_error"


def test_exec_repair_validator_mismatch() -> None:
    validator = ExecRepairValidator()
    node = _make_node(
        candidate="41",
        raw_output="""```python
print("42")
```
""",
    )

    assert validator.validate(node) == 0.0
    assert validator.failure_reason(node.text, "dummy problem") == "exec_mismatch"


def test_exec_repair_validator_str_input_returns_missing_raw_output() -> None:
    validator = ExecRepairValidator()

    assert validator.validate("42", "problem") == 0.0
    assert validator.failure_reason("42", "problem") == "missing_raw_output"


def test_exec_repair_validator_missing_problem_or_raw_output() -> None:
    validator = ExecRepairValidator()
    node_missing_problem = ReasoningNode(
        node_id="n2",
        text="42",
        type="answer",
        attributes={"raw_output": "```python\nprint('42')\n```"},
    )
    node_missing_raw_output = ReasoningNode(
        node_id="n3",
        text="42",
        type="answer",
        attributes={"problem": "dummy problem"},
    )

    assert validator.validate(node_missing_problem) == 0.0
    assert (
        validator.failure_reason(node_missing_problem.text, "dummy problem")
        == "missing_raw_output"
    )
    assert validator.validate(node_missing_raw_output) == 0.0
    assert (
        validator.failure_reason(node_missing_raw_output.text, "dummy problem")
        == "missing_raw_output"
    )
