from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


_BANNED_TOKENS = (
    "import",
    "open(",
    "__import__",
    "os.",
    "sys.",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "requests",
    "http",
    "pip",
    "conda",
)


@dataclass(frozen=True)
class PythonExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return text[:max_chars]


def _contains_banned_token(code: str) -> str | None:
    lowered = code.lower()
    for token in _BANNED_TOKENS:
        if token in lowered:
            return token
    return None


def run_python(
    code: str,
    *,
    timeout_sec: float,
    max_stdout_chars: int,
    max_stderr_chars: int,
) -> PythonExecResult:
    banned = _contains_banned_token(code)
    if banned is not None:
        return PythonExecResult(
            stdout="",
            stderr=_truncate(
                f"Execution blocked: banned token detected: {banned}",
                max_stderr_chars,
            ),
            exit_code=1,
            timed_out=False,
        )

    env = {"PYTHONIOENCODING": "utf-8"}

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "snippet.py"
            script_path.write_text(code, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=temp_dir,
                env=env,
                check=False,
            )
            return PythonExecResult(
                stdout=_truncate(completed.stdout or "", max_stdout_chars),
                stderr=_truncate(completed.stderr or "", max_stderr_chars),
                exit_code=completed.returncode,
                timed_out=False,
            )
    except subprocess.TimeoutExpired as exc:
        partial_stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        partial_stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr_text = f"Execution timed out after {timeout_sec} seconds." + (
            "\n" + partial_stderr if partial_stderr else ""
        )
        return PythonExecResult(
            stdout=_truncate(partial_stdout, max_stdout_chars),
            stderr=_truncate(stderr_text, max_stderr_chars),
            exit_code=124,
            timed_out=True,
        )
    except Exception as exc:
        return PythonExecResult(
            stdout="",
            stderr=_truncate(
                f"Execution failed: {type(exc).__name__}: {exc}", max_stderr_chars
            ),
            exit_code=1,
            timed_out=False,
        )
