# EXP5 Smoke Runbook (graph_bot vs graph_bot_exec, sparse vs dense)

**Status:** Ready

This runbook pins smoke and ablation commands for EXP5 preparation.

## 1) Preconditions

- Run from repository root: `/home/hyunwoo/git/graph-bot`
- Use mock provider for smoke:
  - `GRAPH_BOT_LLM_PROVIDER=mock`
  - `GRAPH_BOT_LLM_MODEL=mock`
- Pin distiller for EXP5-style runs:
  - `--distiller-mode rulebased`
- For dense_template smoke, embedding provider must be available. If sentence_transformers/openai are not installed, either install extras `.[dev,hf,openai]` or set `GRAPH_BOT_EMBEDDING_PROVIDER=deterministic`.

## 2) Core Smoke Matrix (4 Configurations)

All commands below must produce 4 files in `outputs/stream_logs/`:

- `<run_id>.calls.jsonl`
- `<run_id>.problems.jsonl`
- `<run_id>.stream.jsonl`
- `<run_id>.token_events.jsonl`

### 2.1 graph_bot + oracle + sparse

```bash
PYTHONPATH=src GRAPH_BOT_LLM_PROVIDER=mock GRAPH_BOT_LLM_MODEL=mock graph-bot stream tests/fixtures/game24_smoke.jsonl --run-id test_smoke_graph_sparse --metrics-out-dir outputs/stream_logs --task game24 --mode graph_bot --validator-mode oracle --retrieval-backend sparse_jaccard --use-edges --policy-id semantic_topK_stats_rerank --distiller-mode rulebased --max-problems 3
```

### 2.2 graph_bot + oracle + dense

```bash
PYTHONPATH=src GRAPH_BOT_LLM_PROVIDER=mock GRAPH_BOT_LLM_MODEL=mock GRAPH_BOT_EMBEDDING_PROVIDER=deterministic graph-bot stream tests/fixtures/game24_smoke.jsonl --run-id test_smoke_graph_dense --metrics-out-dir outputs/stream_logs --task game24 --mode graph_bot --validator-mode oracle --retrieval-backend dense_template --use-edges --policy-id semantic_topK_stats_rerank --distiller-mode rulebased --max-problems 3
```

### 2.3 graph_bot_exec + oracle + sparse

```bash
PYTHONPATH=src GRAPH_BOT_LLM_PROVIDER=mock GRAPH_BOT_LLM_MODEL=mock graph-bot stream tests/fixtures/game24_smoke.jsonl --run-id test_smoke_exec_sparse --metrics-out-dir outputs/stream_logs --task game24 --mode graph_bot_exec --validator-mode oracle --retrieval-backend sparse_jaccard --use-edges --policy-id semantic_topK_stats_rerank --distiller-mode rulebased --max-problems 3
```

### 2.4 graph_bot_exec + oracle + dense

```bash
PYTHONPATH=src GRAPH_BOT_LLM_PROVIDER=mock GRAPH_BOT_LLM_MODEL=mock GRAPH_BOT_EMBEDDING_PROVIDER=deterministic graph-bot stream tests/fixtures/game24_smoke.jsonl --run-id test_smoke_exec_dense --metrics-out-dir outputs/stream_logs --task game24 --mode graph_bot_exec --validator-mode oracle --retrieval-backend dense_template --use-edges --policy-id semantic_topK_stats_rerank --distiller-mode rulebased --max-problems 3
```

## 3) EXP5 Required Ablations

Use the same retrieval backend and k/top-k setting when comparing graph necessity.

Note: the plan describes `--use-edges True/False`. In this CLI, `True` is `--use-edges` and `False` is omitting the flag.

### 3.1 Flat-Template-RAG baseline

```bash
PYTHONPATH=src GRAPH_BOT_LLM_PROVIDER=mock GRAPH_BOT_LLM_MODEL=mock graph-bot stream tests/fixtures/game24_smoke.jsonl --run-id test_exp5_flat_dense_no_edges --metrics-out-dir outputs/stream_logs --task game24 --mode flat_template_rag --validator-mode oracle --policy-id semantic_only --retrieval-backend dense_template --distiller-mode rulebased --max-problems 3
```

### 3.2 Graph necessity (dense backend, same retrieval budget)

`use-edges=True` arm:

```bash
PYTHONPATH=src GRAPH_BOT_LLM_PROVIDER=mock GRAPH_BOT_LLM_MODEL=mock graph-bot stream tests/fixtures/game24_smoke.jsonl --run-id test_exp5_graph_dense_edges_true --metrics-out-dir outputs/stream_logs --task game24 --mode graph_bot --validator-mode oracle --policy-id semantic_topK_stats_rerank --use-edges --retrieval-backend dense_template --distiller-mode rulebased --max-problems 3
```

`use-edges=False` arm:

```bash
PYTHONPATH=src GRAPH_BOT_LLM_PROVIDER=mock GRAPH_BOT_LLM_MODEL=mock graph-bot stream tests/fixtures/game24_smoke.jsonl --run-id test_exp5_graph_dense_edges_false --metrics-out-dir outputs/stream_logs --task game24 --mode graph_bot --validator-mode oracle --policy-id semantic_topK_stats_rerank --retrieval-backend dense_template --distiller-mode rulebased --max-problems 3
```

## 4) Repeat Protocol (EXP5 Figures)

- Preferred: 5 repeats per arm (`rep01`..`rep05`) with paired bootstrap CI.
- Minimum fallback: 3 shuffles per arm (if run budget is constrained).
- Keep each pair matched by fixture/shuffle index and runtime settings.
- Keep distiller pinned: `--distiller-mode rulebased`.

## 5) Fixture Pointers For Cross-Task Smoke

- MGSM: `tests/fixtures/mgsm_smoke.jsonl`
- WordSorting: `tests/fixtures/wordsorting_smoke.jsonl`

These fixtures are intentionally small (3 problems each) for fast smoke checks.
