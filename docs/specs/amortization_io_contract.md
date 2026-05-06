# Amortization I/O Contract (Stream JSONL -> Analysis -> Figure)

**Status:** Draft (v0)

This document pins the I/O contract between:
- stream execution logs (`*.calls.jsonl`, `*.problems.jsonl`, `*.stream.jsonl`),
- TokenTracker event logs (new),
- the amortization analysis script(s),
- and figure-ready outputs.

References:
- `src/graph_bot/pipelines/metrics_logger.py`
- `src/graph_bot/pipelines/stream_loop.py`
- `src/graph_bot/datatypes.py`
- `src/graph_bot/utils/amortization.py`
- `docs/policies/latency_reporting_policy.md`
- `docs/specs/token_tracker_schema_v0.json`

## Canonical Telemetry Semantics Contract

This contract freezes one machine-readable split for edge-related telemetry:

1. **retrieval-phase traversal evidence**: what retrieval consumed before solve.
2. **post-solve persisted edge growth**: what memory update stored or increased
   after solve.

All downstream docs, diagnostics, and claim gates MUST preserve this split.
`memory_n_edges` and `edges_added_count` MUST be treated as persisted-edge
growth signals only and MUST NOT be used by themselves as graph-structure-use
proof.

Canonical slots and mapping rules:

| Canonical slot | Phase | Meaning | Current/allowed mapping |
| --- | --- | --- | --- |
| `retrieval_path_node_cardinality` | retrieval | Node cardinality of retrieved path material consumed by solve | Use explicit `retrieval_path_node_cardinality` if present. Otherwise map legacy `reuse_count` to this slot strictly as retrieved-node cardinality. |
| `retrieval_used_stored_edges` | retrieval | Whether retrieval traversed persisted graph edges before solve | Use explicit `retrieval_used_stored_edges` if present. Otherwise leave this slot unproven unless retrieval-path artifacts directly demonstrate stored-edge traversal. |
| `persisted_edge_snapshot` | post-solve persistence | Persisted edge count after update/snapshot | Map `memory_n_edges` to this slot. |
| `persisted_edges_added_delta` | post-solve persistence | Persisted edges added during update | Map `edges_added_count` to this slot. |

Normative interpretation rules:

- Retrieval-phase graph-structure claims MUST be gated by retrieval-phase slots
  only.
- `reuse_count` can establish that retrieval reused nodes; it cannot by itself
  establish that persisted edges were traversed.
- `packed_context_tokens` can establish context size; it cannot by itself
  establish graph-edge traversal.
- `memory_n_edges` and `edges_added_count` describe persisted state/growth after
  solve and are insufficient as graph-structure-use proof on their own.
- Reporting packages MUST keep retrieval-phase traversal evidence, execution
  readiness, and performance outcomes on separate claim axes.

## 1) Inputs Produced By `graph-bot stream`

All files are JSONL (one JSON object per line).

### 1.1 `*.calls.jsonl` (per operation)

Model: `StreamCallMetrics` (`src/graph_bot/datatypes.py`).

Required keys (current code emits some as null):
- `call_id` (string)
- `parent_id` (string | null)
- `t` (int)
- `problem_id` (string)
- `operation` (string; expected: `retrieve`, `solve`, `validate`)

Optional keys (may be null):
- `attempt_index` (int | null)
- `temperature` (float | null)
- `validator_passed` (bool | null)
- `failure_reason` (string | null)
- `prompt_variant` (string | null)
- `prompt_tokens`, `completion_tokens`, `total_tokens` (int | null)
- `latency_ms` (float | null)
- `api_cost_usd` (float | null)
- `error_type` (string | null)
- `raw_output`, `candidate_line`, `normalization`, `precheck_failure_reason`

Example (from `outputs/test_logs_3/test_run_mock_3.calls.jsonl`):
```json
{"call_id":"...","parent_id":null,"t":1,"problem_id":"q1","operation":"retrieve","prompt_tokens":null,"latency_ms":null,"api_cost_usd":null,"error_type":null}
{"call_id":"...","parent_id":"...","t":1,"problem_id":"q1","operation":"solve","prompt_tokens":50,"completion_tokens":10,"total_tokens":60,"latency_ms":100.0,"api_cost_usd":null,"error_type":null}
```

### 1.2 `*.problems.jsonl` (per problem)

Model: `StreamProblemMetrics` (`src/graph_bot/datatypes.py`).

Keys:
- `t` (int)
- `problem_id` (string)
- `solved` (bool)
- `attempts` (int)
- `solved_attempt` (int | null)
- `attempt_success_rate` (float | null)
- `llm_calls` (int)
- `tokens_total` (int)
- `latency_total_ms` (float)
- `api_cost_usd` (float)
- `retrieval_hit` (bool)
- `reuse_count` (int; legacy mapping for
  `retrieval_path_node_cardinality`, not a stored-edge traversal proof)
- `memory_n_nodes` (int)
- `memory_n_edges` (int; mapping for `persisted_edge_snapshot` only)
- `contamination_rate` (float | null)

### 1.3 `*.stream.jsonl` (cumulative stream metrics)

Model: `StreamCumulativeMetrics` (`src/graph_bot/datatypes.py`).

Keys:
- `t` (int)
- `cumulative_solved` (int)
- `cumulative_api_cost_usd` (float)
- `cost_per_solved` (float)
- `contamination_rate` (float | null)

Example (from `outputs/test_logs_3/test_run_mock_3.stream.jsonl`):
```json
{"t":1,"cumulative_solved":1,"cumulative_api_cost_usd":0.0,"cost_per_solved":0.0,"contamination_rate":null}
```

## 2) TokenTracker Event Log (New; Event-Level Accounting)

Purpose: provide Track A vs Track B accounting across LLM/embedding/RAG (and
failures), independent from the stream logger.

Format: JSONL; each line MUST conform to `docs/specs/token_tracker_schema_v0.json`.

Required keys (schema v0):
- `timestamp` (string, ISO 8601)
- `run_id` (string; problem-session id, recommended `{stream_run_id}:{problem_id}`)
- `event_type` (enum)
- `model` (string)
- `usage` (object; `prompt_tokens`, `completion_tokens`, `total_tokens`)
- `cost_usd` (number)

Optional keys:
- `parent_id`, `span_id`, `component`, `latency_ms`, `metadata`
- `metadata` recommended keys in v0.3:
    - `stream_run_id`, `problem_id`, `t`, `operation`, `status`
    - `packed_context_tokens` (for `rag_retrieval` events): total tokens of
      context packed into the prompt. Supportive retrieval-size signal only; not
      stored-edge traversal proof by itself.
    - `edges_added_count` (for `insertion` events): number of new edges created.
      Map only to `persisted_edges_added_delta`.

Preferred future retrieval metadata fields for direct claim-gating:
- `retrieval_path_node_cardinality`
- `retrieval_used_stored_edges`

Minimal example:
```json
{"timestamp":"2026-01-22T08:10:00Z","run_id":"test_run_mock_3:q1","span_id":"...","parent_id":null,"event_type":"llm_completion","component":"pipeline","model":"llama3-8b-instruct","usage":{"prompt_tokens":50,"completion_tokens":10,"total_tokens":60},"cost_usd":0.0,"latency_ms":100,"metadata":{"stream_run_id":"test_run_mock_3","problem_id":"q1","t":1,"operation":"solve","status":"ok","pricing_version":"v0"}}
```

## 3) Analysis Scripts: Required Inputs And Guarantees

### 3.1 Current amortization script (v0)

`src/graph_bot/utils/amortization.py` consumes ONLY `*.stream.jsonl` and outputs
a CSV with the following columns:
- `t`
- `cumulative_api_cost_usd`
- `cumulative_solved`
- `cost_per_solved`

Contract for `*.stream.jsonl` (v0): lines must include those four keys and `t`
must be parseable as int.

### 3.2 Extended amortization analysis (planned)

To generate figure-ready curves aligned with `docs/policies/latency_reporting_policy.md`:
- accuracy curve and attempt success rate should be derived from `*.problems.jsonl`.
- p50/p95 latency should be computed from per-problem `latency_total_ms`.
- Track A vs Track B cost should be derived from TokenTracker events.

## 4) Outputs (Figure-Ready Artifacts)

### 4.1 CSV (canonical)

At minimum (current v0):
```text
t,cumulative_api_cost_usd,cumulative_solved,cost_per_solved
```

Recommended (planned extension; stable names):
```text
t,cum_accuracy,attempt_success_rate_p50,tokens_total_p50,latency_ms_p50,latency_ms_p95,cost_usd_pipeline_avg,cost_usd_total_avg
```

## 5) Failure And Partial Data Rules

- JSONL parsing
  - analysis scripts MUST ignore empty lines.
  - analysis scripts MUST fail fast on invalid JSON (recommended for early
    contract enforcement) unless running in a best-effort mode.
- Stream interruption
  - incomplete trailing lines are allowed; readers should ignore the last line
    if it is not valid JSON.
- Missing cost/latency
  - if TokenTracker cost is estimated (e.g., stream interrupted before final
    usage is known), mark `metadata.is_estimated=true` and include
    `metadata.estimate_method`.
