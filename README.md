# graph-bot

Graph-augmented Buffer of Thoughts (Graph-BoT).

This repository implements a **Graph-augmented Buffer of Thoughts**. The goal
is to compare episodic reasoning (ToT/GoT) with a persistent MetaGraph-based
memory for reuse and better long-horizon performance.

Many parts are still **placeholders (stubs)**, but v0.1 includes a minimal,
persistent MetaGraph implementation (on-disk JSON) so the end-to-end loop can be
run and iterated on.

## Quickstart

```bash
pip install -e ".[dev,hf,vllm]"
```

### Tooling Versions (Compatibility)

For consistent `black`/`ruff`/`pre-commit` results across machines, install the
versions pinned by `.pre-commit-config.yaml`:

```bash
python -m pip install "pre-commit==3.7.1" "black==24.3.0" "ruff==0.5.5"
pre-commit run --all-files
```

## CLI Pipeline (v0.1)

1. **Tree generation**: `graph-bot seeds-build`
2. **Insert into MetaGraph**: `graph-bot trees-insert`
3. **Postprocess (prune/decay)**: `graph-bot postprocess`
4. **Retrieve & answer**: `graph-bot retrieve`
5. **Loop once**: `graph-bot loop-once`

Example:

```bash
graph-bot seeds-build data/seeds.jsonl --out outputs/trees.json
graph-bot trees-insert outputs/trees.json
graph-bot postprocess --t 10
graph-bot retrieve "2 5 8 11 → 24" --k 3 --show-paths --task game24
```

MetaGraph state is persisted to `outputs/metagraph.json` by default and can be
overridden with `GRAPH_BOT_METAGRAPH_PATH`.

## Continual Stream (v0.2 / B-main)

Run a continual Game-of-24 stream that accumulates templates into the MetaGraph
and emits JSONL metrics (per-call, per-problem, cumulative stream metrics).

Supports multiple modes:
- `--mode graph_bot`: Full Graph-BoT pipeline (default).
- `--mode io`: Baseline IO prompting (single-query).
- `--mode cot`: Baseline Chain-of-Thought (single-query).

For production-style runs using OpenAI models, see `scripts/run_g2_openai.sh`.

1) Prepare a JSONL problems file (one per line):

```json
{"id":"q-001","numbers":[2,5,8,11],"target":24}
```

2) Run the vLLM server (OpenAI-compatible API):

```bash
graph-bot llm-server start --port 2427 --served-model-name llama3-8b-instruct --model /path/to/local/hf/model
```

3) Run the stream:

```bash
GRAPH_BOT_LLM_BASE_URL=http://127.0.0.1:2427/v1 \
GRAPH_BOT_LLM_MODEL=llama3-8b-instruct \
graph-bot stream data/game24.jsonl --run-id run --metrics-out-dir outputs/stream_logs --mode graph_bot --use-edges --policy-id semantic_topK_stats_rerank --validator-mode oracle
```

This produces JSONL logs under `outputs/stream_logs/`:

- `outputs/stream_logs/run.calls.jsonl`
- `outputs/stream_logs/run.problems.jsonl`
- `outputs/stream_logs/run.stream.jsonl`

4) Generate EXP1 amortization curve (CSV):

```bash
graph-bot amortize outputs/stream_logs/run.stream.jsonl --out outputs/amortization_curve.csv
```

## Repository Structure

- `src/`: main source code
- `configs/`: runtime configs (settings and templates)
- `libs/`: external libraries / submodules
- `docs/`: project documentation
  - `docs/specs/`: schemas, I/O contracts, and design specs (source of truth)
  - `docs/checklists/`: runbooks and pre-run checklists (human procedure)
  - `docs/policies/`: policies and reporting rules
  - `docs/metrics/`: metrics definitions and cost boundaries
- `tests/`: (currently empty) test skeletons
- `pyproject.toml`: dependencies, scripts
- `.env.example`: env var template

## Docs

Start here:

- `docs/checklists/integration_checklist.md`: pre-run checklist for end-to-end stream + amortization reporting
- `docs/specs/amortization_io_contract.md`: I/O contract for `graph-bot amortize` inputs/outputs
- `docs/specs/token_tracker_schema_v0.json`: event-level token/cost schema for usage accounting
- `docs/policies/no_interactive_policy.md`: non-interactive execution requirements and timeout policy
- `docs/policies/latency_reporting_policy.md`: latency reporting format (p50/p95) and outlier handling

## Artifacts & Logs

This repo writes two different kinds of files during runs:

- `logs/`: application/runtime logs from the Python logger (`src/graph_bot/logsetting.py`).
- `outputs/`: experiment artifacts and persisted state.
  - `outputs/metagraph.json`: persisted MetaGraph state (default; override with `GRAPH_BOT_METAGRAPH_PATH`).
  - `outputs/stream_logs/`: JSONL metrics from `graph-bot stream`.
    - `<run_id>.calls.jsonl`: Individual LLM call metrics.
    - `<run_id>.problems.jsonl`: Per-problem aggregated metrics.
    - `<run_id>.stream.jsonl`: Cumulative stream metrics.
    - `<run_id>.token_events.jsonl`: Event-level usage/cost logs (G2 instrumentation).
  - `outputs/test_logs*/`: ad-hoc / test run outputs used for schema/policy verification.
  - `outputs/runs/`, `outputs/artifacts/`: optional structured output roots (see `configs/paths.yaml`).

Note: `graph-bot llm-server` writes server logs to `vllm_server.log` by default.

## Technical Architecture (`src/graph_bot`)

### Data Models (`datatypes.py`)

Pydantic models define the system I/O and internal representations.

- `SeedData`: seed input to tree generation
- `ReasoningNode` / `ReasoningEdge`: graph primitives
- `ReasoningTree`: tree artifact from a single episode
- `MetaGraph`: persistent graph buffer spanning multiple trees
- `UserQuery`: incoming query
- `RetrievalResult` / `RetrievalPath`: retrieval output

### Configuration (`settings.py`)

Settings are managed via environment variables prefixed with `GRAPH_BOT_`.
Notable settings:

- `GRAPH_BOT_METAGRAPH_PATH`: JSON persistence location
- `GRAPH_BOT_PRICING_PATH`: path to pricing YAML (default: `configs/pricing/pricing_v0.yaml`)
- `GRAPH_BOT_EXECUTION_TIMEOUT_SEC`: per-problem hard timeout (default: `60.0`)
- `GRAPH_BOT_TOP_K_PATHS`: retrieval top-k
- `GRAPH_BOT_EMA_ALPHA`, `GRAPH_BOT_EMA_TAU_DAYS`: update/pruning parameters

### Adapters (`adapters/`)

- `hiaricl_adapter.py`: stub generator producing `ReasoningTree` from `SeedData`
- `graphrag.py`: v0.1 persistent MetaGraph store + retrieval (semantic + stats)
- `vllm_openai_client.py`: OpenAI-compatible client for vLLM
- `mock_client.py`: Mock client for testing

### Pipelines (`pipelines/`)

- `build_trees.py`: seeds → reasoning trees
- `retrieve.py`: query → retrieval result
- `main_loop.py`: glue layer (insert, retrieve+usage, postprocess pruning)
- `stream_loop.py`: continual learning stream (Game of 24)
- `metrics_logger.py`: structured JSONL logging for experiments
- `postprocess.py`: offline graph maintenance (pruning/verbalization)

### Evaluation (`eval/`)

- `validators.py`: problem-specific validators (e.g., `Game24Validator`)

## Design Spec v0.1 (Summary)

### Pipeline

1. Tree generation (`HiAR-ICL` or baseline solver)
2. Insert into MetaGraph
3. Postprocess every T inputs (rerank/verbalize/prune/augment)
4. Retrieve `k` optimal paths and instantiate prompt
5. Update node/edge stats with validator/scorer feedback

### Memory Schema

- `ReasoningNode.type`: `thought | action | evidence | answer`
- **Template nodes**: stored as `type="thought"` and `attributes.subtype="template"`
- Required attributes (v0.1)
  - `task`, `created_at`, `last_used_at`
  - `stats`: `n_seen`, `n_used`, `n_success`, `n_fail`, `ema_success`,
    `avg_tokens`, `avg_latency_ms`, `avg_cost_usd`
  - `quality`: `validator_passed`, `parse_ok`

### Merge & Update Rules

- Dedup key: `hash(normalize(text))`
- EMA update: `ema_success = (1-α)*ema_success + α*success`
- Decay: `ema_success *= exp(-Δt/τ)` when unused
- Pruning heuristic: if `n_seen >= 5` and `ema_success < 0.2`
- Defaults: `α=0.1`, `τ=7 days`, `N_min=5`, `p_min=0.2`

### Selection Policy

- Stage A: semantic top-K (embedding or lexical similarity)
- Stage B: stats rerank (`ema_success`, `avg_cost`, path length)

## Week 4 Research Notes (Summary)

Baseline results (Game of 24, 98 problems, `gpt-4o-mini`):

| Method | Accuracy | Tokens/problem (approx) | Latency/problem | Notes |
| --- | --- | --- | --- | --- |
| io prompting | 8.16% | 157.95 | 1.07s | single-query |
| CoT | 11.67% | 500.61 | 3.05s | single-query |
| ToT (oracle) | 40.80% | 6,160.32 | 11.02s | multi-query |
| GoT (oracle) | 69.40% | 10,256.99 | 25.04s | multi-query |
| BoT (text-only) | 7.14% | 2,475.77 | 16.52s | MetaBuffer noise |
| BoT (code-aug) | 41.84% | 3,385.20 | 151.13s | exec/repair loop |

Key observations:

- Text-only BoT underperforms due to missing validator and buffer pollution.
- Code-augmented BoT improves accuracy but adds large latency from exec/repair.
- Retrieval quality is critical: noisy templates dominate without validation.
- Token/cost accounting must include both pipeline calls and RAG components.
