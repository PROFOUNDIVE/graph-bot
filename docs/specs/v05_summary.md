# Graph-Bot v0.5 Architecture Summary

**Status:** Released (v0.5.0)

Graph-Bot v0.5 introduces **Domain Extensibility**, allowing the system to execute, validate, and distill reasoning for diverse tasks beyond "Game of 24". It establishes the `TaskSpec` protocol, `TaskRegistry`, and scoped memory isolation to ensure multi-domain stability.

## 1. Core Features (What's New)

### 1.1 Multi-Task Architecture (`src/graph_bot/tasks/`)
The monolithic stream loop has been refactored into a task-driven architecture.

- **TaskSpec Protocol**: Defines the contract for loading, prompting, extracting, and validating problems.
- **TaskRegistry**: Dynamic lookup of task modules (`game24`, `wordsorting`, `mgsm`).
- **Legacy Support**: `game24` remains the default, preserving backward compatibility.

### 1.2 Task-Scoped Retrieval (Memory Isolation)
To prevent "cross-talk" between domains (e.g., retrieving math templates for a word puzzle), GraphRAG now enforces soft isolation.

- **Default**: `retrieve_paths` filters nodes where `attributes.task` matches the query task.
- **Override**: `--cross-task-retrieval` allows explicit sharing of reasoning patterns across domains.
- **Provenance**: All new nodes are tagged with `task` metadata at insertion time.

### 1.3 BoT-Aligned Distillation
The distillation pipeline now strictly follows the **Buffer of Thoughts** (BoT) specification.

- **Input**: Summarized solution steps (not raw CoT) + final candidate.
- **Output**: Structured thought template (Template Distillation prompt) prefixed with `Task: <name>` to guarantee correct indexing.
- **Cardinality**: Exactly one "thought template" node is inserted per solved problem.

### 1.4 Solve&Instantiate (Meta Reasoner)
Solve&Instantiate is aligned to BoT's per-query minimal call structure by embedding the BoT "Meta Reasoner" prompt (verbatim) into the task solver system prompt for `--mode graph_bot`.

- **No extra instantiate call**: instantiation happens inside the single solve call using retrieved context.
- **Tasks updated**: `game24`, `wordsorting`, `mgsm`.

See `docs/specs/bot_alignment_solve_instantiate_meta_reasoner.md`.

### 1.5 New Supported Domains
| Domain | Type | Validator | Characteristics |
|--------|------|-----------|-----------------|
| **Game24** | Symbolic Math | Oracle | Arithmetic expressions evaluating to 24 |
| **WordSorting** | Text Processing | Oracle | Alphabetical sorting of input words |
| **MGSM** | Math Reasoning | Oracle | Grade-school math word problems (final numeric answer) |

## 2. CLI Updates

New options in `graph-bot stream`:

```bash
# Run WordSorting task
graph-bot stream data/words.jsonl --task wordsorting

# Run MGSM with cross-task retrieval enabled
graph-bot stream data/math.jsonl --task mgsm --cross-task-retrieval
```

| Option | Default | Description |
|--------|---------|-------------|
| `--task` | `game24` | Target domain module to load. |
| `--cross-task-retrieval` | `False` | If set, search all nodes regardless of task tag. |

## 3. Developer Guide

See `docs/specs/task_module_authoring.md` for a checklist on implementing new tasks.

### Quick Start for New Tasks:
1. Implement `TaskSpec` in `src/graph_bot/tasks/<name>.py`.
2. Register in `src/graph_bot/tasks/registry.py`.
3. Add unit tests ensuring deterministic input/output handling.

## 4. Migration from v0.4

- **No Breaking Changes** for existing Game24 workflows.
- **Metagraph Compatibility**: v0.5 metagraphs include `task` attributes. v0.4 nodes (missing `task`) are treated as generic/legacy but remain retrievable if `cross-task-retrieval` is used or if fallback logic is triggered (though strict isolation is default).

## 5. Future Roadmap (v0.6+)

- **Weak Validator Integration**: Wiring `WeakLLMJudgeValidator` into the task oracle slot for open-ended domains.
- **Dense Retrieval**: Replacing Jaccard similarity with embedding-based search.
- **Template Similarity Gate**: Add a dense similarity threshold check before inserting new templates into the MetaGraph.
- **Code-Augmented Tasks**: Python interpreter integration for tasks like `HumanEval`.
