# Fix List: Log Schema & Policy Alignment

## 1. Missing Fields in Log Schema
*   **Log Files Checked**: `outputs/test_logs_3/*.jsonl`
*   **Observations**:
    *   `api_cost_usd` is `0.0` or `null` in calls. We need to implement a token cost calculation logic for the Mock client or default provider to verify cost tracking.
    *   `latency_ms` is `null` for `retrieve` and `validate` operations (only present in `solve`). These operations should also have latency tracking.
    *   `parent_id` is correctly populated, preserving trace structure.

## 2. Policy Violations (B2)
*   **Interactive Blocking**: The current mock run did not trigger interactive waits, so this is technically passed, but we should verify the Timeout mechanism exists in `stream_loop.py` (currently it relies on `client.chat` timeout).
*   **Latency Metrics**: The logs provide `latency_total_ms`, but the report policy requires `p50/p95`. The aggregation logic (in `graph-bot amortize` command) needs to be verified to support this.

## 3. Recommended Actions
1.  **Implement Cost Calculation**: Add a simple cost calculator in `StreamMetricsLogger` or `MockLLMClient` to populate `api_cost_usd` based on token counts.
2.  **Add Latency Tracking for All Ops**: Wrap `retrieve` and `validate` calls with `time.perf_counter()` to capture their latency in `StreamCallMetrics`.
3.  **Explicit Timeout**: Ensure `func_timeout` or similar is applied at the problem-solving level, not just the LLM client level, to catch infinite loops in logic/validators.
