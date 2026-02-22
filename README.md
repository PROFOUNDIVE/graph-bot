# Graph-BoT: Graph-augmented Buffer of Thoughts

Research prototype for **Non-parametric Continual Learning** using persistent reasoning graphs.

> **Status:** v0.6.0 (Active Research)
> **Paper:** Graph-BoT: Amortizing Reasoning Costs via Persistent Memory (Work in Progress)

## Overview

Graph-Bot extends **Buffer of Thoughts (BoT)** by structuring reasoning templates into a persistent **MetaGraph**. It aims to solve the "Amortized System 2" problem:

> *Can we amortize the cost of expensive reasoning (CoT/ToT) by accumulating and reusing successful thought structures over time?*

### Key Features
- **Persistent MetaGraph**: Stores distilled reasoning templates (nodes) and their causal relationships (edges).
- **Continual Stream Pipeline**: Online learning loop where every solved problem updates the memory.
- **Amortized Efficiency**: Costs decrease over time as retrieval replaces generation.
- **Stability Mechanisms**: Validator-gated updates prevent memory contamination.
- **Multi-Task Support (v0.5)**: Extensible task architecture supporting `game24`, `wordsorting`, `mgsm`.
- **BoT Prompt Alignment (v0.5)**: Problem Distiller + Meta Reasoner + Template Distillation prompts integrated.

---

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

## Continual Stream Pipeline

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
graph-bot stream data/game24.jsonl --run-id run --metrics-out-dir outputs/stream_logs --mode graph_bot --task game24 --use-edges --policy-id semantic_topK_stats_rerank --validator-mode oracle
```

This produces JSONL logs under `outputs/stream_logs/`:

- `outputs/stream_logs/run.calls.jsonl`
- `outputs/stream_logs/run.problems.jsonl`
- `outputs/stream_logs/run.stream.jsonl`
- `outputs/stream_logs/run.token_events.jsonl`

4) Generate EXP1 amortization curve (CSV):

```bash
graph-bot amortize outputs/stream_logs/run.stream.jsonl --out outputs/amortization_curve.csv
```

## v0.5 New Features

### Domain Extensibility
Support for multiple reasoning domains beyond Game24.

- **Task Registry**: Plug-in system for loading, prompting, and validating different tasks.
- **Supported Tasks**:
  - `game24`: Arithmetic reasoning (default).
  - `wordsorting`: Text processing constraint satisfaction.
  - `mgsm`: Multilingual grade school math.

```bash
graph-bot stream data/math.jsonl --task mgsm --cross-task-retrieval
```

### Task-Scoped Retrieval
Prevents memory contamination between different tasks.

- **Default**: Retrieval is strictly isolated to the current task's templates.
- **Override**: `--cross-task-retrieval` enables sharing reasoning patterns across domains.

### BoT-Aligned Distillation
Ensures distilled templates follow the Buffer of Thoughts structure.

- **Input**: Summarized solution steps (not raw CoT).
- **Output**: Templates distilled via the BoT-style "Template Distillation" prompt, prefixed with `Task: <name>`.

### LLM Distillation Prompt Alignment
LLM-based distillation prompts are aligned to BoT-style roles.

- **Problem Distiller (Query)**: Emits `Key information` / `Restrictions` / `Distilled task`.
- **Template Distillation (Trace)**: Produces concise, structured templates for MetaGraph insertion.

### Solve&Instantiate (Meta Reasoner)
In `--mode graph_bot`, solver prompts are prefixed with the BoT "Meta Reasoner" prompt (verbatim) so retrieval context is instantiated within the single solve call.

See `docs/specs/bot_alignment_solve_instantiate_meta_reasoner.md`.

## v0.4 New Features

### WeakLLMJudgeValidator
Validation for domains without ground truth (cheap oracle) answers:
- **LLM-as-Judge**: Uses LLM to evaluate answer correctness with YES/NO output
- **Configurable Model**: `--validator-model` option to specify judge model
- **Fail-safe**: Returns 0.0 on timeout/error to prevent false positives

```bash
graph-bot stream data/problems.jsonl --validator-mode weak_llm_judge --validator-model gpt-4o-mini
```

### LLMDistiller
LLM-based query normalization and trace abstraction:
- **Query Normalization**: Extracts core intent from user queries via LLM
- **Trace Distillation**: Creates reusable template nodes from successful traces
- **Cold-start Guard**: Falls back to rule-based for simple numeric queries

```bash
graph-bot stream data/problems.jsonl --distiller-mode llm
```

### Distiller Registry
Factory pattern with three modes:
- `rulebased` (default): Regex-based extraction for Game24
- `llm`: LLM-based normalization for arbitrary domains
- `none`: No-op passthrough

### Naming Clarification
- `GraphRAGDistiller` → `RuleBasedDistiller` (clarifies it's regex-based, not GraphRAG library)

## v0.3 New Features

### Distillation Loop
Version 0.3 introduces a dual-path distillation process to compress MetaGraph knowledge:
- **`distill(query)`**: Pre-emptive distillation that generates optimized reasoning templates for specific query clusters.
- **`distill(trace)`**: Post-hoc distillation that refines successful reasoning traces into reusable graph structures.

### Online Edge Creation
The system now supports **connect-to-retrieved** logic during tree generation. When new thoughts are generated, the engine proactively searches the MetaGraph for existing nodes to link to, enabling cross-episode knowledge transfer in real-time.

### Budget Enforcement
Strict resource limits are now enforced at the pipeline level:
- **Max Templates**: Limits the number of unique reasoning templates stored per task category.
- **Max Paths**: Caps the number of retrieval paths explored during prompt instantiation to maintain low latency.

### Cost Audit & Reliability
To ensure experimental integrity and billing transparency:
- **Run Manifest**: A global ledger (`outputs/run_manifest.jsonl`) tracks every run attempt, ensuring failed/crashed runs are accounted for.
- **Token Audit**: Client-side token counting (via `tiktoken`) validates API-reported usage. Discrepancies > 5% trigger `token_audit_gap` warnings.
- **Site-Specific Pricing**: Local pricing overrides are supported via `configs/pricing/site_specific.yaml` to handle region-specific rates.

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
  - `outputs/run_manifest.jsonl`: Persistent ledger of all run attempts (status: STARTED/COMPLETED/FAILED).
  - `outputs/test_logs*/`: ad-hoc / test run outputs used for schema/policy verification.
  - `outputs/runs/`, `outputs/artifacts/`: optional structured output roots (see `configs/paths.yaml`).

### Testing Artifact Conventions

When writing tests under `tests/`, ensure any files created are unmistakably test-only:

- Prefer `tmp_path` for all generated artifacts.
- If a test needs a `run_id`, it MUST start with `test_` (e.g., `test_e2e_run`).
- Any persisted test artifacts under `outputs/` MUST use a `test_` prefix (file/dir).
- Logs produced during tests MUST be redirected to `logs/test_execution.log` (fixtures handle this; do not write timestamped log files during tests).

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
- `GRAPH_BOT_MAX_TEMPLATES`: max unique templates per category
- `GRAPH_BOT_MAX_PATHS`: max paths to explore during retrieval
- `GRAPH_BOT_EMA_ALPHA`, `GRAPH_BOT_EMA_TAU_DAYS`: update/pruning parameters
- `GRAPH_BOT_SLACK_WEBHOOK_URL`: Slack incoming webhook for automated experiment reporting
- `GRAPH_BOT_VALIDATOR_MODEL`: model for weak_llm_judge validator
- `GRAPH_BOT_DISTILLER_MODE`: distiller selection (rulebased, llm, none)

### Adapters (`adapters/`)

- `graphrag.py`: Persistent MetaGraph store + retrieval (semantic + stats)
- `distiller.py`: Query/trace distillation (RuleBasedDistiller, LLMDistiller, NullDistiller)
- `vllm_openai_client.py`: OpenAI-compatible client for vLLM
- `mock_client.py`: Mock client for testing

### Pipelines (`pipelines/`)

- `stream_loop.py`: continual learning stream (Game of 24)
- `metrics_logger.py`: structured JSONL logging for experiments

### Evaluation (`eval/`)

- `validators.py`: Problem validators (`Game24Validator`, `WeakLLMJudgeValidator`, etc.)

## Design Spec (v0.5.0)

> See [docs/specs/v05_summary.md](docs/specs/v05_summary.md) for architecture details.

### Pipeline

1. **Distill (Query)**: Input query is normalized (RuleBased or LLM).
2. **Retrieve**: Semantic Top-K retrieval from MetaGraph (using Jaccard).
3. **Solve**: LLM solves the problem conditioned on retrieved context.
4. **Validate**: Validator (Oracle or LLM-Judge) checks the answer.
5. **Distill (Trace)**: Successful trace is distilled into a template.
6. **Update**: New template is merged into MetaGraph (validator-gated).

### Memory Schema

- **Node Types**: `thought` (template), `action`, `evidence`, `answer`
- **Template Nodes**: `type="thought"`, `subtype="template"`
- **Attributes**: `task`, `stats` (n_used, n_success, ema_success), `quality` (validator_passed)

### Update Rules (EMA)

- **Dedup Key**: `hash(normalize(text))`
- **Update**: `ema_success = (1-α)*ema_success + α*success`
- **Decay**: `ema_success *= exp(-Δt/τ)` when unused
- **Pruning**: Candidate for removal if `n_seen >= 5` and `ema_success < 0.2`

### Retrieval Policy

- **Stage A (Candidate)**: Jaccard similarity (token overlap)
- **Stage B (Rerank)**: Combined score of semantic + EMA success + edge weights
- **Edge Traversal**: Prioritizes paths connected by high-probability edges

## Week 8 Research Notes (Summary)

**Status:** v0.4 Released. Focus on "Domain Extension" via LLM-based components.

### Key Findings (v0.3 → v0.4)

| Experiment | Status | Key Observation |
| --- | --- | --- |
| **EXP1 (Amortization)** | Complete | Demonstrated efficiency gains; `cost_per_solved` decreases as MetaGraph matures. |
| **EXP2 (Warm-start)** | Complete | Seeding with 10 successful traces improves initial solve rate (19 -> 24 solved). |
| **EXP3 (Contamination)** | Complete | Validator is critical. Without it, contamination hits ~91% and performance collapses. |
| **EXP4 (Memory Growth)** | Analysis | Long-run stability (N=300+) verified; OOM analysis pending. |
| **EXP6 (Domain Extension)** | Planned | LLM judge + LLM distiller now available for non-Game24 domains. |

### v0.4 Addresses

| v0.3 Limitation | v0.4 Status |
| --- | --- |
| LLM-based Distillation | ✅ Implemented (`LLMDistiller`) |
| Weak Validator / LLM-Judge | ✅ Implemented (`WeakLLMJudgeValidator`) |

### Remaining Limitations & Future Work

1.  **Retrieval Quality**: Currently uses Jaccard (token overlap). Dense embedding integration is planned.
2.  **Template Similarity Gate**: Add a dense similarity threshold check before inserting new templates into the MetaGraph.
3.  **Edge Creation**: Edges only form on retrieval hits. Cold-start creates isolated nodes.
4.  **Code-augmented Execution**: To match BoT (Code-aug) performance.
