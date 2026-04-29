# EXP5 Issues Verification

Checked against:
- `outputs/exp5_report_v0.7.1.md`
- `src/graph_bot/pipelines/stream_loop.py`
- `src/graph_bot/adapters/graphrag.py`
- `src/graph_bot/tasks/game24.py`

## Verdict Summary

| Issue | Verdict | Notes |
| --- | --- | --- |
| #1 Edge destination node missing in `_insert_solution_template()` | **Not confirmed (current code)** | Current implementation points edges to `distilled_nodes[0].node_id` and `insert_trees()` resolves both tree-local and existing-node endpoints. |
| #2 `graph_bot_exec` incompatible with Game24 prompt format | **Partially confirmed historically, not confirmed in current code** | Current `Game24Task.build_solver_prompt()` explicitly requires Python code block + `<answer>` block for `graph_bot_exec`. Exec mismatch can still happen if model output violates that contract. |
| #3 `graph_bot_no_edges` unsupported because missing from `MODE_CAPABILITIES` | **Not confirmed (current code)** | Alias normalization maps `graph_bot_no_edges` and `graph_bot_no` to `graph_bot` before capabilities check. |

---

## Issue #1 Verification

### Claim
`_insert_solution_template()` creates edges to `f"episode-{problem_id}-solution"` (missing node), so edges are dropped in `insert_trees()`.

### What the code currently does

1. Edge destination uses the distilled node ID (not hardcoded solution ID):

`src/graph_bot/pipelines/stream_loop.py:879-883`

```python
edges.append(
    ReasoningEdge(
        src=source_id,
        dst=distilled_nodes[0].node_id,
        relation="used_for",
```

2. The inserted tree contains that same `distilled_nodes[0]` node:

`src/graph_bot/pipelines/stream_loop.py:891-895`

```python
tree = ReasoningTree(
    tree_id=f"episode-{problem_id}",
    root_id=distilled_nodes[0].node_id,
    nodes=distilled_nodes,
    edges=edges,
```

3. `insert_trees()` resolves endpoints via both tree-local map and existing graph IDs:

`src/graph_bot/adapters/graphrag.py:481-487`

```python
src = node_id_map.get(edge.src)
dst = node_id_map.get(edge.dst)
if src is None and edge.src in node_index:
    src = edge.src
if dst is None and edge.dst in node_index:
    dst = edge.dst
if not src or not dst:
    continue
```

### Conclusion
The reported `dst`-missing path is not present in current code. This issue appears already fixed.

### Recommended fix (for older branches that still have the bug)

```python
# stream_loop.py
ReasoningEdge(
    src=source_id,
    dst=distilled_nodes[0].node_id,  # do not hardcode episode-...-solution
    relation="used_for",
)
```

```python
# graphrag.py (endpoint resolution fallback)
src = node_id_map.get(edge.src)
dst = node_id_map.get(edge.dst)
if src is None and edge.src in node_index:
    src = edge.src
if dst is None and edge.dst in node_index:
    dst = edge.dst
if not src or not dst:
    continue
```

---

## Issue #2 Verification

### Claim
`graph_bot_exec` expects Python code blocks, but Game24 prompts ask for expression-only outputs, causing exec validator failure.

### What the code currently does

1. `graph_bot_exec` prompt explicitly requires a Python code block and matching `<answer>` block:

`src/graph_bot/tasks/game24.py:122-128`

```python
if mode == "graph_bot_exec":
    exec_system = (
        f"{META_REASONER_SYSTEM}\n\n{got_cot_system}\n"
        "For graph_bot_exec, output exactly two parts in order:\n"
        "1) A fenced python code block (```python ... ```) that prints exactly one line containing only the final Python expression candidate.\n"
        "2) A final <answer>...</answer> block containing the same final expression candidate.\n"
        "Constraints: no imports; no file or network access; print only the final expression."
    )
```

2. Exec path enforces code-block presence and output/candidate match:

`src/graph_bot/pipelines/stream_loop.py:1470-1472`, `src/graph_bot/pipelines/stream_loop.py:1508-1512`

```python
code = _extract_python_fenced_block(raw_output)
if code is None:
    exec_failure_reason = "exec_no_output"

exec_candidate_match = _answers_match(exec_answer, candidate_line)
if not exec_candidate_match:
    exec_failure_reason = "exec_mismatch"
```

3. Candidate extraction prefers `<answer>` and strips code blocks for fallback parsing:

`src/graph_bot/tasks/game24.py:145-155`

```python
answer_matches = list(_ANSWER_BLOCK_PATTERN.finditer(raw_output))
...
text_without_code = _PYTHON_CODE_BLOCK_PATTERN.sub("\n", raw_output)
candidate, _ = _normalize_candidate_line(text_without_code, allowed_numbers=numbers)
```

### Conclusion
The prompt/validator incompatibility described in the report is not present in current code. However, `exec_no_output`/`exec_mismatch` can still occur when model output violates the required format.

### Recommended fix (if still failing often in runs)

```python
# Keep the strict prompt contract for graph_bot_exec
if mode == "graph_bot_exec":
    # require python block + matching <answer>
    ...
```

```python
# Optional hardening: ensure fallback mode remains explicit
if mode == "graph_bot_exec" and attempt_index == settings.retry_max_attempts:
    attempt_mode = "graph_bot"
```

---

## Issue #3 Verification

### Claim
`graph_bot_no_edges` is unsupported because it is not in `MODE_CAPABILITIES`.

### What the code currently does

1. Capabilities are canonical modes only:

`src/graph_bot/pipelines/stream_loop.py:33-41`

```python
MODE_CAPABILITIES: dict[str, dict[str, bool]] = {
    "graph_bot": {"uses_retrieval": True, "updates_memory": True},
    "graph_bot_exec": {"uses_retrieval": True, "updates_memory": True},
    ...
}
```

2. Legacy aliases are normalized before capability lookup:

`src/graph_bot/pipelines/stream_loop.py:44-47`, `src/graph_bot/pipelines/stream_loop.py:125-126`

```python
MODE_ALIASES = {
    "graph_bot_no": ("graph_bot", False),
    "graph_bot_no_edges": ("graph_bot", False),
}

requested_mode = mode or settings.mode
resolved_mode, resolved_use_edges = _resolve_mode_alias(requested_mode, use_edges)
```

3. Unsupported-mode error is raised only after alias normalization:

`src/graph_bot/pipelines/stream_loop.py:185-189`

```python
mode_capability = MODE_CAPABILITIES.get(active_mode)
if mode_capability is None:
    raise ValueError(
        f"Unsupported mode '{active_mode}'. Supported modes: {supported_modes}"
    )
```

### Conclusion
This specific failure mode is not reproducible in current code for `graph_bot_no_edges` or `graph_bot_no`. It appears already mitigated by alias resolution.

### Recommended fix (for branches without alias support)

```python
MODE_ALIASES = {
    "graph_bot_no": ("graph_bot", False),
    "graph_bot_no_edges": ("graph_bot", False),
}

requested_mode = mode or settings.mode
resolved_mode, resolved_use_edges = _resolve_mode_alias(requested_mode, use_edges)
```

---

## Bottom Line

Against the current source files, Issue #1 and Issue #3 are not confirmed, and Issue #2 is only partially true (exec strictness remains, but prompt incompatibility has been addressed).
