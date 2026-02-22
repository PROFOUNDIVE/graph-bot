# Graph-Bot v0.6 Architecture Summary

**Status:** Released (v0.6.0)

Graph-Bot v0.6 introduces a **code-augmented solve path** (`graph_bot_exec`), a **safe execution validator** (`exec_repair`), and a **dense template retrieval backend** (`dense_template`) with embedding-provider abstraction and truthful retrieval accounting. It also adds EXP5-oriented smoke/runbook assets and closes the cold-start Edge=0 gate with minimal bootstrap edges.

## 1) What's New in v0.6

### 1.1) Code-Augmented Solve Path (`graph_bot_exec`)

`_solve_with_retries` now supports a code-augmented attempt path:

- Attempts `1..N-1`: `graph_bot_exec`
- Last attempt: fallback to `graph_bot`
- Retry prompt injection is bounded (`Previous output` truncated to 1000 chars)

Execution calls are logged as structured spans/events (`operation=exec`) to preserve end-to-end traceability.

### 1.2) Safe Python Sandbox Runner + ExecRepair Validator

Two core components were added:

- `src/graph_bot/tools/python_executor.py`
  - Isolated subprocess execution (`-I -S`), timeout, deterministic stdout/stderr truncation
  - Fail-closed token banlist for import/file/network/system package paths
- `src/graph_bot/eval/validators.py` (`ExecRepairValidator`)
  - Parses fenced Python code from model output and validates candidate by actual execution result
  - Standardized failure codes:
    - `exec_timeout`
    - `exec_banned_token`
    - `exec_runtime_error`
    - `exec_no_output`
    - `exec_mismatch`
    - `missing_raw_output`

This closes v0.5's "Code-Augmented Tasks" roadmap item with a bounded, test-covered MVP.

### 1.3) Dense Template Retrieval Backend

GraphRAG retrieval now supports two seed backends:

- `sparse_jaccard` (default)
- `dense_template` (embedding cosine similarity)

`GraphRAGAdapter` now includes embedding-aware internals:

- Embedding provider construction (`deterministic`, `local`, `openai`, `hybrid`)
- Query/node embedding caches
- Dense seed scoring via cosine similarity
- Cache refresh behavior on graph updates (`insert_trees` / `import_graph`)

### 1.4) Embedding Provider Abstraction + Pricing Hook

New module: `src/graph_bot/adapters/embeddings.py`

- `DeterministicHashEmbeddingProvider` (offline deterministic test provider)
- `SentenceTransformerProvider` (lazy import guarded)
- `OpenAIEmbeddingProvider` (optional import guarded, pricing-aware)
- `HybridEmbeddingProvider` (primary/fallback with warnings)

Pricing table now includes `text-embedding-3-large` in:

- `configs/pricing/pricing_v0.yaml`

### 1.5) Retrieval Logging Truthfulness + Embedding Events

`rag_retrieval` token events now report backend/model truthfully:

- Sparse: `model="sparse_jaccard"`
- Dense: `model=<actual embedding model>` with metadata:
  - `retrieval_backend`
  - `embedding_provider`
  - `embedding_model_actual`

Dense runs additionally emit `event_type="embedding"` / `operation="embed"` events with correlation fields (`t`, `problem_id`, `stream_run_id`, `run_id`) for Track B accounting.

### 1.6) Prompt Contract Updates for MGSM / WordSorting

For `mode == "graph_bot_exec"`, task prompts now require:

1. A fenced Python block that prints exactly one final answer line
2. A final `<answer>...</answer>` block

Candidate extraction prioritizes `<answer>` and ignores Python block text to avoid leakage from executable scaffolding.

### 1.7) Edge=0 Cold-Start Remediation

`GraphRAGAdapter.insert_trees` adds a minimal bootstrap edge for edge-less cold-start inserts so early stream metrics can satisfy non-zero edge checks without changing retrieval policy semantics.

### 1.8) EXP5 Smoke Runbook + Fixtures

Added:

- `docs/checklists/exp5_smoke_runbook.md`
- `tests/fixtures/mgsm_smoke.jsonl`
- `tests/fixtures/wordsorting_smoke.jsonl`

The runbook pins 4-way smoke matrix commands (`graph_bot`/`graph_bot_exec` x `sparse`/`dense`) and EXP5 ablation arms.

## 2) CLI Updates

`graph-bot stream` gained/extended options relevant to v0.6:

| Option | Default | Description |
|--------|---------|-------------|
| `--retrieval-backend` | `settings.retrieval_backend` (`sparse_jaccard`) | Select seed retrieval backend: `sparse_jaccard` or `dense_template`. |
| `--validator-mode` | `settings.validator_mode` (`oracle`) | Includes `exec_repair` in addition to `oracle`, `weak_llm_judge`. |

Runbook examples include `--mode graph_bot_exec` for code-augmented solving and validation.

## 3) New / Updated Settings

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `retrieval_backend` | `GRAPH_BOT_RETRIEVAL_BACKEND` | `sparse_jaccard` | Seed retrieval backend (`sparse_jaccard`, `dense_template`). |
| `embedding_provider` | `GRAPH_BOT_EMBEDDING_PROVIDER` | `hybrid` | Embedding provider selection (`hybrid`, `openai`, `local`, `deterministic`). |
| `openai_embedding_model` | `GRAPH_BOT_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model for dense retrieval. |

Existing `embedding_model` (`all-MiniLM-L6-v2`) remains local fallback model.

## 4) Token Event / Accounting Changes

v0.6 expands accounting coverage for retrieval infrastructure:

- `rag_retrieval` events now carry backend-aware model attribution
- Dense retrieval emits `embedding` token events
- `exec` tool calls are logged with operation/component metadata

Schema compatibility is maintained (no schema file replacement required for existing readers).

## 5) Test Coverage Added in v0.6

- `tests/unit/test_python_executor.py`
  - timeout / banlist / truncation behavior
- `tests/unit/test_exec_repair_validator.py`
  - happy path, timeout, runtime error, mismatch, missing raw output
- `tests/unit/test_embeddings_provider.py`
  - deterministic stability, batching, hybrid fallback behavior
- `tests/unit/test_graphrag_dense.py`
  - dense-vs-sparse ordering and lazy cache fill after insert
- `tests/unit/test_edge_bootstrap.py`
  - cold-start bootstrap edge creation
- `tests/test_contract.py`
  - graph_bot_exec fallback behavior
  - non-oracle validator receives rich `ReasoningNode`
  - sparse/dense retrieval logging truthfulness
  - dense embedding events presence and sparse absence

## 6) Migration from v0.5

### Potentially Impactful Changes

- If you rely on retrieval behavior assumptions, note the new backend switch:
  - explicit sparse baseline: `--retrieval-backend sparse_jaccard`
  - dense backend requires embedding provider availability (or deterministic provider for smoke)
- `exec_repair` is now available as a validator mode for code-augmented workflows.

### Backward Compatibility

- Default stream behavior remains sparse retrieval + oracle validation unless explicitly changed.
- Existing Game24 / WordSorting / MGSM task registry and interfaces remain intact.
- Existing JSONL output families remain unchanged (`calls`, `problems`, `stream`, `token_events`).

### Recommended Upgrade Commands

```bash
# Explicitly preserve v0.5-like retrieval behavior
graph-bot stream data/game24.jsonl --mode graph_bot --retrieval-backend sparse_jaccard --validator-mode oracle

# Enable v0.6 dense+exec path (smoke-friendly)
PYTHONPATH=src GRAPH_BOT_LLM_PROVIDER=mock GRAPH_BOT_EMBEDDING_PROVIDER=deterministic \
graph-bot stream tests/fixtures/game24_smoke.jsonl --mode graph_bot_exec --retrieval-backend dense_template --validator-mode oracle --max-problems 1
```

## 7) Relationship to v0.5 Roadmap

From `docs/specs/v05_summary.md` future roadmap, v0.6 addresses:

- Dense Retrieval: implemented (`dense_template`)
- Code-Augmented Tasks: implemented (`graph_bot_exec` + `ExecRepairValidator` + sandbox)
- Edge Bootstrap Strategy: implemented (cold-start bootstrap edge gate)

Still open for future iterations:

- More advanced template similarity gates and broader GraphRAG backend variants
- Deeper benchmark reporting and long-horizon contamination controls
