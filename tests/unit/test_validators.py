from __future__ import annotations
from graph_bot.eval.validators import (
    Game24Validator,
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
