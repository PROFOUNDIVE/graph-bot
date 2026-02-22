from __future__ import annotations

from graph_bot.tools.python_executor import run_python  # pyright: ignore[reportMissingImports]


def test_run_python_blocks_banlist_before_execution() -> None:
    result = run_python(
        "import math\nprint('should not run')",
        timeout_sec=1.0,
        max_stdout_chars=200,
        max_stderr_chars=200,
    )

    assert result.timed_out is False
    assert result.exit_code != 0
    assert result.stdout == ""
    assert "banned token" in result.stderr.lower()
    assert "import" in result.stderr.lower()


def test_run_python_timeout_infinite_loop() -> None:
    result = run_python(
        "while True:\n    pass",
        timeout_sec=0.2,
        max_stdout_chars=200,
        max_stderr_chars=200,
    )

    assert result.timed_out is True
    assert result.exit_code != 0


def test_run_python_truncates_stdout_and_stderr() -> None:
    result = run_python(
        "print('x' * 50, end='')\nraise RuntimeError('y' * 50)",
        timeout_sec=1.0,
        max_stdout_chars=10,
        max_stderr_chars=15,
    )

    assert result.timed_out is False
    assert result.exit_code != 0
    assert result.stdout == "x" * 10
    assert len(result.stderr) == 15
