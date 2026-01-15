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

## Repository Structure

- `src/`: main source code
- `configs/`: runtime configs (settings and templates)
- `libs/`: external libraries / submodules
- `tests/`: (currently empty) test skeletons
- `pyproject.toml`: dependencies, scripts
- `.env.example`: env var template

## Technical Architecture (`src/graph_bot`)

### Data Models (`types.py`)

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
- `GRAPH_BOT_TOP_K_PATHS`: retrieval top-k
- `GRAPH_BOT_EMA_ALPHA`, `GRAPH_BOT_EMA_TAU_DAYS`: update/pruning parameters

### Adapters (`adapters/`)

- `hiaricl_adapter.py`: stub generator producing `ReasoningTree` from `SeedData`
- `graphrag.py`: v0.1 persistent MetaGraph store + retrieval (semantic + stats)

### Pipelines (`pipelines/`)

- `build_trees.py`: seeds → reasoning trees
- `retrieve.py`: query → retrieval result
- `main_loop.py`: glue layer (insert, retrieve+usage, postprocess pruning)

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
