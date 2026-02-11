## Task Module Authoring Guide (`src/graph_bot/tasks/`)

This guide explains the minimum contract for adding a new stream task (domain) to Graph-Bot.

### 1) Implement the `TaskSpec` contract

Create `src/graph_bot/tasks/<task_name>.py` and implement all methods from `TaskSpec` in `src/graph_bot/tasks/base.py`:

- `name` (lowercase identifier)
- `load_problems(jsonl_path)`
- `to_user_query(problem)`
- `build_solver_prompt(mode, query, retrieval)`
- `extract_candidate(raw_output, query)`
- `oracle_validate(candidate, query, problem)`
- `summarize_steps(raw_output, candidate, query)`
- `distill_template_input(query, steps_summary)`

Recommended pattern: define a `@dataclass(frozen=True)` task class and (if needed) a Pydantic `Problem` model for JSONL rows.

### 2) Enforce metadata conventions

In `to_user_query`, always set:

- `metadata["task"] = <task_name>`

If oracle validation needs labels, store deterministic fields in metadata (examples in current code):

- WordSorting: `metadata["target"]`
- MGSM: `metadata["gold_answer"]`

### 3) Keep candidate extraction deterministic

`extract_candidate` should produce a normalized value that is stable across formatting variance.

Existing patterns:

- WordSorting: first non-empty line + whitespace normalization
- MGSM: final numeric token extraction + numeric normalization
- Game24: expression normalization constrained by allowed numbers

### 4) Keep oracle validation strict and explainable

`oracle_validate` must return `(bool, reason_or_none)`:

- return `(True, None)` when valid
- return `(False, "<reason>")` when invalid

Use deterministic comparisons (exact normalized string/number match). Avoid probabilistic checks in oracle mode.

### 5) Distillation contract (BoT-aligned)

Do not persist raw CoT. `summarize_steps` should return compact structured summaries.

`distill_template_input` must produce a task-prefixed prompt body:

```text
Task: <task>
Problem: <query.question>
Solution Steps Summary: <summary>
Final Candidate: <candidate>
```

### 6) Register the task

Update `src/graph_bot/tasks/registry.py` so `TaskRegistry` loads the new task and exposes it by `name`.

Also export from `src/graph_bot/tasks/__init__.py` if needed by callers/tests.

### 7) Add task-focused tests

Add `tests/unit/test_<task_name>_task.py` covering at minimum:

- loader parsing from JSONL
- `to_user_query` metadata (`task` + required label fields)
- candidate extraction normalization
- oracle validation success/failure cases
- `distill_template_input` shape (`Task:` prefix)

If stream behavior is affected, extend integration tests in `tests/integration/test_pipelines.py`.
