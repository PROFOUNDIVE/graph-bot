# TokenTracker Integration Checklist (Pre-Implementation)

**Status:** Draft (v0)

This checklist is the pre-run gate before implementing `TokenTracker` hooks and
before re-collecting baseline streams.

References:
- `docs/metrics/token_cost_boundaries.md`
- `docs/policies/latency_reporting_policy.md`
- `docs/specs/token_tracker_schema_v0.json`
- `docs/specs/fix_list_log_schema.md`
- `src/graph_bot/pipelines/stream_loop.py`
- `src/graph_bot/pipelines/metrics_logger.py`

## 0) Scope And Deliverables

Deliverables to validate after integration:
- TokenTracker event log (new): JSONL lines conforming to
  `docs/specs/token_tracker_schema_v0.json`.
- Stream logs (existing): `*.calls.jsonl`, `*.problems.jsonl`, `*.stream.jsonl`
  written by `src/graph_bot/pipelines/metrics_logger.py`.

Non-goal (today): changing how the analysis script works. This checklist is only
about contracts and integration gates.

## 1) Hook Points (LLM / Embedding / RAG)

For each hook point, define: (a) run identity, (b) span identity and parent
linkage, (c) component classification (Track A vs Track B), and (d)
failure/interrupt semantics.

- LLM (pipeline solve)
  - Location: `src/graph_bot/pipelines/stream_loop.py` via
    `src/graph_bot/adapters/vllm_openai_client.py` and
    `src/graph_bot/adapters/mock_client.py`.
  - Required: log `event_type=llm_completion`, `component=pipeline`.
- LLM (auxiliary / evaluator)
  - Examples: validators that call an LLM (e.g., weak judge), summarizers.
  - Required: log `event_type=llm_completion`, `component=auxiliary` or
    `component=evaluator`.
- Embedding
  - Location: embedding model calls used for retrieval / rerank.
  - Required: log `event_type=embedding`, `component=rag_infra`.
- RAG retrieval
  - Location: the high-level retrieval span (e.g., `GraphRAGAdapter.retrieve_paths`).
  - Required: log `event_type=rag_retrieval`, `component=rag_infra`.
  - Required: if RAG has internal sub-steps, they must share a `parent_id`
    pointing to the high-level retrieval span.

## 2) Identity And Aggregation Keys (Run/Problem/Span)

The repository currently uses:
- `run_id` as a file name prefix for stream logs (not embedded in JSON lines).
- `problem_id` and `t` embedded in `.calls.jsonl` / `.problems.jsonl`.

TokenTracker schema v0 requires a `run_id` on every event.

Contract to fix *before* implementation:
- TokenTracker `run_id` MUST represent a single problem-solving session.
  Recommended: `{stream_run_id}:{problem_id}`.
- TokenTracker MUST include these metadata keys (at minimum):
  - `stream_run_id`
  - `problem_id`
  - `t`
  - `operation` (e.g., `retrieve`, `solve`, `validate`)

## 3) Track A vs Track B Classification Rules

Use `docs/metrics/token_cost_boundaries.md` as the source of truth.

- Track A (pipeline-only)
  - `component=pipeline` for reasoning steps that produce the solution.
  - Algorithmic retries/backtracks count in Track A.
- Track B (total)
  - Includes Track A plus `component=rag_infra`, `component=auxiliary`,
    `component=evaluator`.
  - System failures/timeouts MUST be counted in Track B.

## 4) Failure / Interrupt Semantics (No Lost Accounting)

Observed gaps (from `docs/specs/fix_list_log_schema.md`):
- `api_cost_usd` missing/zero in mock runs.
- `latency_ms` missing for `retrieve` and `validate` operations.

Rules to enforce:
- Every tracked span MUST emit exactly one event line when it completes.
- If an exception occurs, an event MUST still be emitted with:
  - `cost_usd=0.0` only when known to be 0 (e.g., cache hit), otherwise a best
    estimate or omit cost and mark it in `metadata`.
  - `metadata.status` in {`ok`, `error`, `timeout`, `cancelled`}.
  - `metadata.error_type` for exceptions.
- On stream interruption (KeyboardInterrupt/SIGTERM):
  - flush all buffered JSONL writes (use `finally` semantics in implementation).
  - do not drop partial usage; partial usage MUST be marked as estimated.

## 5) Baseline Re-measurement Checklist

Before re-collecting baseline runs:
- Usage reset boundaries
  - Per-problem counters MUST reset at problem start.
  - Aggregation must be per problem session (`{stream_run_id}:{problem_id}`), not
    across problems.
- Multi-thread safety guardrails
  - Current `graph-bot stream` is sequential.
  - If concurrency is added later, the tracker must be context-local
    (thread-local or explicit context propagation) and log sinks must be
    concurrency-safe (file locks or per-worker files).
- Spot-check log completeness on a small run (N=1..3)
  - `outputs/test_logs_3/test_run_mock_3.calls.jsonl` shows `retrieve` and
    `validate` latency are currently null; this must be fixed after integration.
  - TokenTracker events must exist for: `retrieve`, `solve`, `validate`.

## 6) Open Decisions (Resolved As Q&A)

Q1: When do we calculate `cost_usd` (logging-time vs analysis-time)?
- A: Do both.
  - Logging-time: store `usage` + an estimated `cost_usd` with
    `metadata.pricing_version`.
  - Analysis-time: allow recomputation/backfills if prices change; treat the
    pricing table as versioned input.

Q2: What exactly is the usage reset boundary?
- A: Reset is per problem session.
  - TokenTracker `run_id` represents one problem-solving session.
  - Stream-level cumulative metrics are derived from per-problem metrics, not
    the other way around.

Q3: How do we aggregate in multi-thread runs?
- A: Not supported today; contract for future support:
  - Each event carries a problem-session `run_id` and unique `span_id`.
  - Aggregation groups by problem-session `run_id` regardless of wall-clock
    overlap.

## 7) G2 Smoke Run Definition

This run confirms the integration of pricing, timeout, and logging components.

- **Problems Fixture**: `tests/fixtures/game24_smoke.jsonl`
- **Environment Variables**:
  - `GRAPH_BOT_METAGRAPH_PATH=outputs/metagraph_smoke.json`
  - `GRAPH_BOT_PRICING_PATH=configs/pricing/pricing_v0.yaml`
  - `GRAPH_BOT_EXECUTION_TIMEOUT_SEC=60.0`
  - `GRAPH_BOT_LLM_PROVIDER=mock`
  - `GRAPH_BOT_LLM_MODEL=mock`
- **Expected Outputs**:
  - `outputs/stream_logs/g2_smoke_1.calls.jsonl`
  - `outputs/stream_logs/g2_smoke_1.problems.jsonl`
  - `outputs/stream_logs/g2_smoke_1.stream.jsonl`
  - `outputs/stream_logs/g2_smoke_1.token_events.jsonl` (Verify each problem has `solve`, `retrieve`, `validate` events)
- **Command (Copy-Pasteable)**:
  ```bash
  GRAPH_BOT_LLM_PROVIDER=mock GRAPH_BOT_LLM_MODEL=mock GRAPH_BOT_METAGRAPH_PATH=outputs/metagraph_smoke.json GRAPH_BOT_PRICING_PATH=configs/pricing/pricing_v0.yaml GRAPH_BOT_EXECUTION_TIMEOUT_SEC=60.0 graph-bot stream tests/fixtures/game24_smoke.jsonl --run-id g2_smoke_1 --metrics-out-dir outputs/stream_logs --mode graph_bot --use-edges --policy-id semantic_topK_stats_rerank --validator-mode oracle --max-problems 3
  ```
