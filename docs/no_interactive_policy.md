# Non-Interactive Execution Policy

## 1. Objective
To guarantee that all experiments and benchmarks run autonomously without human intervention. "Hanging" experiments due to `input()` prompts or debuggers invalidate latency metrics and waste compute resources.

## 2. Blocking Policy: Global Timeout
We adopt a **Hard Timeout** strategy to cover all forms of hanging (interactive waits, infinite loops, logic stalls).

*   **Mechanism**: **Per-Problem Execution Timeout**
    *   Set a strict time limit (e.g., 60s or 120s) for each problem execution.
    *   If the limit is exceeded, the process is terminated or the function raises a `TimeoutError`.
*   **Rationale**:
    *   `input()` mocking misses other infinite loop scenarios.
    *   A hard timeout acts as a catch-all safety net for any non-terminating behavior.

## 3. Implementation Requirements
1.  **Timeout Utility**: Use standard libraries (e.g., `func_timeout` or `signal`) to wrap the main execution logic.
2.  **Configuration**:
    *   Default Timeout: **60 seconds** (adjustable via config).
    *   Environment Variable: `EXECUTION_TIMEOUT_SEC=60`.

```python
from func_timeout import func_timeout, FunctionTimedOut

try:
    result = func_timeout(60, solve_problem, args=(...))
except FunctionTimedOut:
    log_failure("ERR_TIMEOUT", ...)
```

## 4. Logging Rules (3-Line Definition)
When a failure or timeout occurs, log exactly these fields to the experiment registry:

1.  **Event Type**: `ERR_INTERACTIVE` (for input attempts) or `ERR_TIMEOUT` (for duration limit).
2.  **Context**: Function/Module name + Last known state/prompt.
3.  **Trace**: File path and Line number where the blocking call originated.

*Example*:
```text
[ERROR] Type: ERR_INTERACTIVE
[CONTEXT] Module: graph_bot.reasoning.step_3
[TRACE] File: /src/engine.py:42 (input call)
```
