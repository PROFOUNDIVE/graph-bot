# EXP5 Issue Resolution (v0.7.1)

## Scope

This document validates the three EXP5 findings from `outputs/exp5_report_v0.7.1.md`, identifies gaps in the original analysis, and records production fixes applied in code.

## Validation Verdicts

| Issue | Verdict | Gap in original analysis |
| --- | --- | --- |
| #1 Graph edges not traversable | Correct but incomplete | The reported `dst` mismatch is real, but `src` was also unresolved because `insert_trees()` only mapped tree-local node IDs. |
| #2 graph_bot_exec vs Game24 mismatch | Correct | No major gap; the prompt/validator contract mismatch was accurately identified. |
| #3 Mode parsing error (`graph_bot_no`) | Correct | Root cause is outside tracked source scripts; failure evidence is in run artifacts/logs. |

---

## Issue #1: Graph Edges Not Creating Traversable Graph

### Root Cause

Two independent endpoint-resolution problems caused `used_for` edges to be dropped:

1. `dst` endpoint in `_insert_solution_template()` referenced `episode-<id>-solution`, but the persisted node after distillation is `distilled_nodes[0]`.
2. `src` endpoint referenced a retrieved node already in the MetaGraph, but `GraphRAGAdapter.insert_trees()` only resolved edge endpoints from `node_id_map` (new tree-local nodes), not existing graph nodes.

### Fix Implementation

#### 1) Use actual distilled node ID for edge destination

File: `src/graph_bot/pipelines/stream_loop.py`

- Updated `_insert_solution_template()` edge creation:
  - Before: `dst=f"episode-{problem_id}-solution"`
  - After: `dst=distilled_nodes[0].node_id`

#### 2) Resolve edge endpoints against existing graph nodes

File: `src/graph_bot/adapters/graphrag.py`

- Updated `insert_trees()` edge resolution logic:
  - Keep tree-local mapping via `node_id_map`
  - If unresolved, fall back to existing node IDs in `node_index`
  - Drop edge only when neither mapping resolves the endpoint

This enables edges from retrieved historical nodes to newly inserted distilled nodes.

### Regression Test

File: `tests/unit/test_graphrag_edge_resolution.py`

- Added test proving a tree edge from an existing MetaGraph node (`src`) to a new node (`dst`) is persisted with relation `used_for`.

---

## Issue #2: Code-Augmentation Mode Incompatible with Game24

### Root Cause

`graph_bot_exec` runs Python execution validation, but `Game24Task.build_solver_prompt()` did not request executable Python output in the expected format. As a result, exec checks failed with `exec_no_output` (no fenced python block).

### Fix Implementation

File: `src/graph_bot/tasks/game24.py`

#### 1) Added graph_bot_exec-specific prompt contract

For `mode == "graph_bot_exec"`, prompt now requires exactly:

1. A fenced Python block printing one line with the final expression candidate
2. A final `<answer>...</answer>` block with the same candidate

Constraints now explicitly include no imports, no file/network access, and print-only final expression.

#### 2) Hardened candidate extraction for code-augmented outputs

- Added explicit `<answer>` block preference (last answer block)
- Strips fenced code blocks before fallback normalization
- Preserves existing Game24 normalization behavior for non-exec outputs

### Regression Tests

File: `tests/unit/test_game24_task.py`

- Added prompt contract test for `graph_bot_exec`
- Added extraction test ensuring `<answer>` is preferred over code block content

---

## Issue #3: Mode Parsing Error (`graph_bot_no_edges` -> `graph_bot_no`)

### Root Cause

The failing EXP5 run was started with `mode='graph_bot_no'` (invalid) and `use_edges=True` (incorrect for no-edge arm), indicating external arm-to-CLI parsing logic was buggy.

Evidence:

- `outputs/run_manifest.jsonl` contains:
  - `run_id=exp5_graph_bot_no_edges_rep01`
  - `config.mode="graph_bot_no"`
- `outputs/stream_logs/exp5_graph_bot_no_edges_rep01.log` shows unsupported mode failure.

### Fix Implementation

File: `src/graph_bot/pipelines/stream_loop.py`

- Added `MODE_ALIASES` and `_resolve_mode_alias()`.
- Normalizes legacy/incorrect aliases:
  - `graph_bot_no` -> `graph_bot` + force `use_edges=False`
  - `graph_bot_no_edges` -> `graph_bot` + force `use_edges=False`
- Applied normalization before adapter initialization, manifest config logging, and per-problem mode checks.

This hardens runtime behavior even when external orchestration scripts pass malformed mode strings.

### Regression Test

File: `tests/unit/test_stream_mode_alias.py`

- Added test verifying both aliases (`graph_bot_no_edges`, `graph_bot_no`) execute successfully with canonicalized mode behavior.

---

## Verification Runbook

Commands executed after implementation:

```bash
pytest tests/unit/test_graphrag_edge_resolution.py tests/unit/test_game24_task.py tests/unit/test_stream_mode_alias.py -q
pytest tests/unit/test_graphrag.py tests/unit/test_edge_bootstrap.py tests/unit/test_wordsorting_task.py tests/unit/test_mgsm_task.py tests/unit/test_bot_prompt_parity.py tests/test_contract.py::test_solve_with_retries_graph_bot_exec_fallback_and_exec_logging -q
ruff check src/graph_bot/adapters/graphrag.py src/graph_bot/pipelines/stream_loop.py src/graph_bot/tasks/game24.py tests/unit/test_game24_task.py tests/unit/test_graphrag_edge_resolution.py tests/unit/test_stream_mode_alias.py
black --check src/graph_bot/adapters/graphrag.py src/graph_bot/pipelines/stream_loop.py src/graph_bot/tasks/game24.py tests/unit/test_game24_task.py tests/unit/test_graphrag_edge_resolution.py tests/unit/test_stream_mode_alias.py
```

Observed results:

- New regression tests: pass
- Related regression suite: pass
- Ruff: pass
- Black check: pass
- LSP diagnostics: clean on all changed source and test files

---

## Recommended Follow-up (EXP5 rerun)

1. Rerun EXP5 arms with corrected external arm parsing (explicit `--mode graph_bot` and explicit `--use-edges` only for edge-enabled arm).
2. Inspect `token_events` for `operation="exec"` and `code_block_present=true` rate in `graph_bot_exec` runs.
3. Compare post-fix graph metrics (edge count and multi-hop retrieval path length) between no-edge and edge-enabled arms to validate graph contribution claims.
