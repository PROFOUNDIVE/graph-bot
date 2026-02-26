# Graph-Bot v0.7 Architecture Summary

**Status:** Released (v0.7.0)

Graph-Bot v0.7 integrates **BoT (Buffer-of-Thoughts), GoT (Graph-of-Thoughts), and ToT (Tree-of-Thoughts)** baseline algorithms into the existing `graph-bot stream` harness. All baselines now execute from a single CLI with the same validator and log schema, enabling fair comparison and reproducible benchmarking.

## 1. Core Features (What's New)

### 1.1 Multi-Baseline Support

Unified execution environment for comparing different reasoning approaches.

**Supported Baselines:**

| Mode | Description | Retrieval | Memory Update |
|------|-------------|-----------|---------------|
| `bot` | BoT baseline - flat template retrieval (k=1) | ✅ Flat | ✅ No edges |
| `got` | GoT baseline - selective refined beam search | ❌ | ❌ |
| `tot` | ToT baseline - beam search | ❌ | ❌ |
| `io` | (existing) IO prompting | ❌ | ❌ |
| `cot` | (existing) Chain-of-Thought | ❌ | ❌ |
| `graph_bot` | (existing) Full Graph-BoT | ✅ Graph | ✅ With edges |
| `graph_bot_exec` | (existing) Graph-BoT with code execution | ✅ Graph | ✅ With edges |

**Key Benefit:** Single harness, single validator, single log schema across all baselines.

### 1.2 BoT Baseline Mode (`--mode bot`)

Faithful implementation of the Buffer-of-Thoughts paper baseline.

**Characteristics:**
- **Retrieval**: Flat template retrieval (`adapter.mode = "flat_template_rag"`)
- **Edges**: Disabled (`adapter.use_edges = False`)
- **Top-k**: Fixed at k=1 (single template like BoT paper)
- **Prompts**: BoT-style instantiation for Game24
  - System prompt enforces single-line output: `<expr> = 24`
  - User prompt includes "Distilled information" and "Thought template"
- **Memory**: Templates inserted without provenance edges

**CLI Usage:**
```bash
graph-bot stream data/game24.jsonl --mode bot --task game24 --run-id bot_run
```

### 1.3 GoT/ToT Baseline Modes

Search-based reasoning implementations ported from the `graph-of-thoughts` library.

**GoT Variant** (selective refined beam search):
- Depth: 3
- Branches (B): 30
- Beam width (K): 3
- Refine width: 10
- Num tries: 1

**ToT Variant** (beam search):
- Depth: 3
- Branches (B): 30
- Beam width (K): 3

**State Schema:**
```python
{
    "items": [{"id": int, "value": float, "expr": str}],
    "next_id": int,
    "depth": int,
    "items_json": str,
    "prev_items_json": str,
    "last_move": str
}
```

**Files:**
- `src/graph_bot/baselines/got/game24_search.py` - Search engine
- `src/graph_bot/baselines/got/game24_prompts.py` - JSONL move prompts
- `src/graph_bot/baselines/got/game24_utils.py` - Scoring utilities
- `src/graph_bot/baselines/got/llm_moves.py` - LLM client interface

**CLI Usage:**
```bash
# GoT baseline
graph-bot stream data/game24.jsonl --mode got --task game24 --run-id got_run

# ToT baseline
graph-bot stream data/game24.jsonl --mode tot --task game24 --run-id tot_run
```

### 1.4 Mode Capabilities Dispatch

Formalized mode semantics via `MODE_CAPABILITIES` in `stream_loop.py`:

```python
MODE_CAPABILITIES = {
    "graph_bot":      {"uses_retrieval": True, "updates_memory": True},
    "graph_bot_exec": {"uses_retrieval": True, "updates_memory": True},
    "flat_template_rag": {"uses_retrieval": True, "updates_memory": True},
    "bot":            {"uses_retrieval": True, "updates_memory": True},
    "io":             {"uses_retrieval": False, "updates_memory": False},
    "cot":            {"uses_retrieval": False, "updates_memory": False},
    "got":            {"uses_retrieval": False, "updates_memory": False},
    "tot":            {"uses_retrieval": False, "updates_memory": False},
}
```

## 2. CLI Updates

New baseline modes in `graph-bot stream`:

```bash
# Run all baselines for comparison
bash scripts/run_baselines_integrated.sh

# Individual baseline runs
graph-bot stream data/game24.jsonl --mode bot --task game24 --run-id bot_run
graph-bot stream data/game24.jsonl --mode got --task game24 --run-id got_run
graph-bot stream data/game24.jsonl --mode tot --task game24 --run-id tot_run
```

## 3. Testing

### 3.1 Unit Tests
- `tests/unit/test_got_search.py` - Deterministic search algorithm tests

### 3.2 Integration Tests
- `tests/integration/test_modes.py` - End-to-end mode verification

### 3.3 Smoke Test Script
```bash
bash scripts/run_baselines_integrated.sh
```

Runs all 5 modes on `tests/fixtures/game24_smoke.jsonl` with mock provider.

## 4. Migration from v0.6

- **No Breaking Changes** for existing modes (`io`, `cot`, `graph_bot`, `graph_bot_exec`).
- **New Options:** Three new baseline modes (`bot`, `got`, `tot`) available.
- **Backward Compatibility:** All existing workflows continue to work.

## 5. Provenance

Baseline sources pinned as git submodules:

- `libs/buffer-of-thought-llm/` - BoT reference implementation
- `libs/graph-of-thoughts/` - GoT/ToT reference implementation

## 6. Artifacts

All modes produce identical artifact schemas:

```
outputs/stream_logs/
├── {run_id}.calls.jsonl       # Per-call metrics
├── {run_id}.problems.jsonl    # Per-problem aggregates
├── {run_id}.stream.jsonl      # Cumulative stream metrics
└── {run_id}.token_events.jsonl # Event-level usage
```

## Commit History

```
chore(baselines): add graph-of-thoughts submodule and paths config
feat(datatypes): add baseline mode support types
feat(baselines): add GoT and ToT search implementation
test(mock): extend MockLLMClient for GoT/ToT prompts
feat(pipeline): add bot/got/tot baseline modes
feat(tasks): add bot mode prompts for Game24
docs(cli): document bot/got/tot modes
test(baselines): add tests for bot/got/tot modes
chore(scripts): add integrated baseline runner
docs(specs): add v0.7.0 architecture summary
```

## Success Criteria Verification

✅ **Single harness**: `graph-bot stream` runs all baselines  
✅ **Single validator**: `Game24Validator` validates all modes  
✅ **Single log schema**: 4 JSONL files per run  
✅ **In-process**: No conda environment switching  
✅ **No secrets**: Clean scan of submodules  
✅ **No regression**: Existing modes unchanged

## v0.7.1 Patch Notes

- Bot prompt parity fix (Game24 bot mode now uses Meta-Reasoner structure)
- Latency p50/p95 test assertions added
