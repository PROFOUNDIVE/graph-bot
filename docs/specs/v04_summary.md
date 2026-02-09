# Graph-Bot v0.4 Architecture Summary

**Status:** Released (v0.4.0)

This document summarizes the architectural changes and features introduced in Graph-Bot v0.4, focusing on LLM-based validation, LLM-based distillation, and improved naming conventions.

## 1) What's New in v0.4

### 1.1) WeakLLMJudgeValidator

A new validator for domains without ground truth (cheap oracle) answers.

**Use Case:** Extend Graph-BoT beyond Game24 to open-ended domains where programmatic validation is impossible.

**Implementation:**
- Binary YES/NO output parsing with regex fallback
- Configurable model via `--validator-model` option
- Fail-safe: Returns 0.0 on timeout/error
- Token event logging with `operation=validate_llm_judge`

**CLI Usage:**
```bash
graph-bot stream data/problems.jsonl \
  --validator-mode weak_llm_judge \
  --validator-model gpt-4o-mini
```

### 1.2) LLMDistiller

LLM-based query normalization and trace abstraction as alternative to rule-based (regex) approach.

**Use Case:** Generalize distillation beyond Game24's numeric patterns to arbitrary domains.

**Components:**
- **Query Normalization:** Extracts core intent from user queries via LLM
- **Trace Distillation:** Creates template nodes (`type=thought`, `subtype=template`)
- **Cold-start Guard:** Returns empty list when no reasoning nodes exist
- **Fallback:** Falls back to RuleBasedDistiller for simple numeric queries

**CLI Usage:**
```bash
graph-bot stream data/problems.jsonl \
  --distiller-mode llm
```

### 1.3) Distiller Registry

Factory pattern for distiller selection via `get_distiller(mode)`.

| Mode | Class | Description |
|------|-------|-------------|
| `rulebased` | `RuleBasedDistiller` | Regex-based number extraction (default) |
| `llm` | `LLMDistiller` | LLM-based normalization and abstraction |
| `none` | `NullDistiller` | No-op passthrough |

### 1.4) Naming Clarification

**`GraphRAGDistiller` → `RuleBasedDistiller`**

The previous name was misleading:
- `GraphRAGDistiller` implied integration with GraphRAG library
- Actual implementation: regex-based pattern matching for Game24
- `RuleBasedDistiller` accurately describes the implementation

**Note:** GraphRAG adapter (`graphrag.py`) still handles graph storage/retrieval.

## 2) New CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--validator-model` | Model for weak_llm_judge validator | `settings.llm_model` |
| `--distiller-mode` | Distiller selection: rulebased, llm, none | `rulebased` |

## 3) New Settings

| Setting | Env Variable | Description |
|---------|--------------|-------------|
| `validator_model` | `GRAPH_BOT_VALIDATOR_MODEL` | Override model for LLM judge |
| `distiller_mode` | `GRAPH_BOT_DISTILLER_MODE` | Default distiller mode |

## 4) Token Event Logging

WeakLLMJudgeValidator logs token events with:
- `operation: validate_llm_judge` (distinct from `validate`)
- `component: evaluator`
- Full usage tracking (prompt_tokens, completion_tokens, cost_usd)

## 5) Test Coverage

New unit tests in v0.4:
- `test_validators.py`: 4 tests for LLM judge (YES/NO parsing, error handling)
- `test_distiller.py`: 3 tests for LLM distiller (query normalization, trace distillation, empty tree guard)

All tests use mocked LLM responses to avoid real API calls.

## 6) Migration from v0.3

### Breaking Changes
- Registry key changed: `graphrag` → `rulebased` for distiller mode
- If you used `--distiller-mode graphrag`, update to `--distiller-mode rulebased`

### Backward Compatible
- `get_validator(mode)` signature extended to `get_validator(mode, model=None)`
- Existing code calling `get_validator("oracle")` works unchanged

## 7) Relationship to v0.3 Limitations

v0.4 addresses two items from v0.3 Future Work:

| v0.3 Future Work | v0.4 Status |
|------------------|-------------|
| 8.5) LLM-based Distillation | ✅ Implemented (`LLMDistiller`) |
| 8.6) Weak Validator / LLM-Judge | ✅ Implemented (`WeakLLMJudgeValidator`) |

Remaining v0.3 Future Work (not addressed in v0.4):
- 8.1) Dense Embedding Integration
- 8.2) Hybrid Scoring
- 8.3) Edge Bootstrap Strategies
- 8.4) Memory Pruning
- 8.7) Code-augmented Execution Loop

## 8) Architecture Diagram (Updated)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Continual Stream Loop                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │  Query   │───▶│  Distiller   │───▶│   Retrieve   │           │
│  │   q_t    │    │ (rulebased/  │    │  (GraphRAG)  │           │
│  └──────────┘    │  llm/none)   │    └──────────────┘           │
│                  └──────────────┘            │                   │
│                         │                    ▼                   │
│                         │           ┌──────────────┐            │
│                         │           │    Solve     │            │
│                         │           │    (LLM)     │            │
│                         │           └──────────────┘            │
│                         │                    │                   │
│                         │                    ▼                   │
│                         │           ┌──────────────┐            │
│                         │           │   Validate   │            │
│                         │           │ (oracle/     │            │
│                         │           │ weak_llm_    │            │
│                         │           │ judge)       │            │
│                         │           └──────────────┘            │
│                         │                    │                   │
│                         ▼                    ▼                   │
│                  ┌──────────────┐    ┌──────────────┐           │
│                  │   Distill    │◀───│   Update     │           │
│                  │   (Trace)    │    │  MetaGraph   │           │
│                  └──────────────┘    └──────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 9) Future Work (v0.5+)

- Dense embedding retrieval (replace Jaccard with cosine similarity)
- Edge bootstrap for cold-start scenarios
- Memory pruning for long-running streams
- Code-augmented execution loop (PoT-style)
- EXP6: Domain extension experiments with weak validator
