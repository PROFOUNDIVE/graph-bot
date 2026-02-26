# Mode Comparison: Flat-template-RAG vs Bot Mode

**Document Version:** 1.0  
**Last Updated:** 2026-02-26  
**Scope:** `src/graph_bot/pipelines/stream_loop.py`, `src/graph_bot/adapters/graphrag.py`, `src/graph_bot/tasks/`

---

## Executive Summary

This document analyzes the differences between **Flat-template-RAG mode** (`flat_template_rag`) and **Bot mode** (`bot`) in the Graph-Bot system. Both modes originate from the Buffer-of-Thought (BoT) paradigm and share the same Meta-Reasoner prompt architecture from the original `buffer-of-thought-llm` repository.

**Important Correction:** Bot mode was originally intended to use the same Complex prompt style (with Meta-Reasoner) as Flat-template-RAG. The current implementation mistakenly uses a simplified prompt for Bot mode. Both modes should use the identical Meta-Reasoner prompt structure.

| Aspect | Flat-template-RAG | Bot Mode |
|--------|-------------------|----------|
| **Retrieval Strategy** | Flat (single-node) templates | Single template (k=1) |
| **Graph Edges** | Ignored (`use_edges=False`) | Ignored (`use_edges=False`) |
| **Meta-Reasoning** | Enabled (Meta-Reasoner prompt) | **Should be Enabled** (currently Disabled - needs fix) |
| **Query Distillation** | Yes | Yes (with metadata preservation) |
| **Prompt Structure** | Meta-reasoner + retrieved context | **Should be identical** to Flat-template-RAG |
| **Memory Updates** | Yes | Yes |
| **Intended Use** | Full BoT pipeline | Simplified BoT baseline (fewer templates) |

---

## 1. Mode Registration

Both modes are registered in `MODE_CAPABILITIES` in `src/graph_bot/pipelines/stream_loop.py`:

```python
MODE_CAPABILITIES: dict[str, dict[str, bool]] = {
    "graph_bot": {"uses_retrieval": True, "updates_memory": True},
    "graph_bot_exec": {"uses_retrieval": True, "updates_memory": True},
    "flat_template_rag": {"uses_retrieval": True, "updates_memory": True},  # <-- Flat-template-RAG
    "bot": {"uses_retrieval": True, "updates_memory": True},               # <-- Bot mode
    "io": {"uses_retrieval": False, "updates_memory": False},
    "cot": {"uses_retrieval": False, "updates_memory": False},
    "got": {"uses_retrieval": False, "updates_memory": False},
    "tot": {"uses_retrieval": False, "updates_memory": False},
}
```

**Key Observation:** Both modes share identical capability flags—both use retrieval and update memory.

---

## 2. Mode Transformation (Stream Loop)

In `src/graph_bot/pipelines/stream_loop.py`, **Bot mode is internally transformed to Flat-template-RAG mode**:

```python
adapter_mode = active_mode
if active_mode == "bot":
    adapter_mode = "flat_template_rag"  # Bot mode uses flat_template_rag internally
    adapter.use_edges = False           # Force disable graph edges

adapter.mode = adapter_mode
```

This transformation means:
- Bot mode **is a specialization** of Flat-template-RAG
- Both modes bypass graph edge traversal
- The distinction lies in **retrieval configuration** and **prompt construction**

---

## 3. Retrieval Differences

### 3.1 Path Count (`k_paths`)

**Flat-template-RAG:**
```python
k_paths = settings.top_k_paths  # Default: 3 (configurable)
retrieval = adapter.retrieve_paths(distilled_query, k=k_paths)
```

**Bot Mode:**
```python
k_paths = 1 if active_mode == "bot" else settings.top_k_paths  # Hardcoded: 1
retrieval = adapter.retrieve_paths(distilled_query, k=k_paths)
```

**Difference:**
- Flat-template-RAG retrieves **multiple paths** (default 3) for diversity
- Bot mode retrieves **only 1 path** (single best template)

### 3.2 Graph Structure Handling

In `src/graph_bot/adapters/graphrag.py`, the `retrieve_paths` method handles modes differently:

```python
if self.mode == "flat_template_rag":
    candidate_paths = [[node_id] for node_id in seed_nodes]  # Single-node paths only
    edge_index = {}
elif self.use_edges:
    # Build candidate paths using graph edges (disabled for both modes)
    adjacency = {...}
    candidate_paths = self._build_candidate_paths(seed_nodes, adjacency, node_scores)
else:
    # Default: single-node paths
    candidate_paths = [[node_id] for node_id in seed_nodes[: settings.rerank_top_n]]
    edge_index = {}
```

**Key Point:** Both modes use **flat paths** (single-node), but Flat-template-RAG may return multiple nodes while Bot mode returns exactly one.

---

## 4. Query Handling Differences

### 4.1 Metadata Preservation

**Bot Mode** preserves original and distilled query metadata:

```python
if active_mode == "bot":
    updated_metadata = dict(query.metadata or {})
    updated_metadata.setdefault("original_question", original_question)
    updated_metadata["distilled_question"] = distilled_question
    query = query.model_copy(update={"metadata": updated_metadata})
```

**Flat-template-RAG** does not preserve this metadata distinction.

### 4.2 Distillation Behavior

Both modes support query distillation (for Game24 task):

```python
if task_spec.name == "game24":
    distilled_question = distiller.distill_query(query.question)
```

The distiller normalizes queries to enable template reuse across semantically similar problems.

---

## 5. Prompt Construction (Shared Architecture)

> **Note:** Both Flat-template-RAG and Bot mode should use the **same Meta-Reasoner prompt structure** derived from the original `buffer-of-thought-llm` repository. The current implementation incorrectly uses a simplified prompt for Bot mode - this is a known issue that should be fixed.

### 5.1 Meta-Reasoner System Prompt (Shared)

From `src/graph_bot/tasks/game24.py`, the `META_REASONER_SYSTEM` prompt:

```python
META_REASONER_SYSTEM = """[Meta Reasoner]
You are a Meta Reasoner who are extremely knowledgeable in all kinds of fields including
Computer Science, Math, Physics, Literature, History, Chemistry, Logical reasoning, Culture,
Language..... You are also able to find different high-level thought for different tasks. Here
are three reasoning sturctures:
i) Prompt-based structure:
It has a good performance when dealing with problems like Common Sense Reasoning,
Application Scheduling
ii) Procedure-based structure
It has a good performance when dealing with creative tasks like Creative Language
Generation, and Text Comprehension
iii) Programming-based:
It has a good performance when dealing with Mathematical Reasoning and Code Programming, it can also transform real-world problems into programming problem which could be
solved efficiently.
(Reasoning instantiation)
Your task is:
1. Deliberately consider the context and the problem within the distilled respond from
problem distiller and use your understanding of the question within the distilled respond to
find a domain expert who are suitable to solve the problem.
2. Consider the distilled information, choose one reasoning structures for the problem.
3. If the thought-template is provided, directly follow the thought-template to instantiate for
the given problem"""
```

### 5.2 Current Implementation Status

**Flat-template-RAG (Correct):**
```python
# Default case (includes graph_bot, flat_template_rag)
base_user = got_cot_user_template.format(input=numbers_str)
return (
    f"{META_REASONER_SYSTEM}\n\n{got_cot_system}",
    f"{base_user}\nRetrieved templates/context:\n{retrieval.concatenated_context}\n",
)
```

**Bot Mode (Incorrect - Needs Fix):**
```python
if mode == "bot":
    # ... simplified prompt without META_REASONER_SYSTEM ...
    bot_system = """You are playing the 24 Game.
    ...
```

**The Fix:** Bot mode should use the same prompt construction as Flat-template-RAG, with the only difference being `k=1` retrieval instead of `k=3`.

### 5.3 Historical Note (Deprecated Implementation)

Previously, Bot mode used a simplified prompt without the Meta-Reasoner. This was incorrect. Both modes should now use the same Complex prompt structure with Meta-Reasoner.

**Old (Incorrect) Bot Mode Prompt:**
```python
# DEPRECATED - Do not use
bot_system = """You are playing the 24 Game...
```

**Correct Approach:** Use `META_REASONER_SYSTEM` for both modes.

From `src/graph_bot/tasks/game24.py`:

```python
if mode == "bot":
    metadata = query.metadata or {}
    distilled_info = str(metadata.get("distilled_question") or query.question)
    original_input = str(metadata.get("original_question") or query.question)

    bot_system = """You are playing the 24 Game.

Given four numbers, use each number exactly once (in any order) and only the operations +, -, *, /.
Parentheses are allowed.

You MUST output exactly one line in the following format:
<python_expression> = 24

No additional text. No tags. No 'Output:' prefix."""

    bot_user_template = """Distilled information:
{distilled_info}

Original input:
{original_input}

Thought template:
{thought_template}
"""
```

**Characteristics:**
- Simple, direct system prompt
- Explicit separation of distilled vs original information
- Single thought template injected directly
- No meta-reasoning instructions

### 5.2 Flat-template-RAG Prompt (Default)

```python
# Default case (includes graph_bot, flat_template_rag)
base_user = got_cot_user_template.format(input=numbers_str)
return (
    f"{META_REASONER_SYSTEM}\n\n{got_cot_system}",
    f"{base_user}\nRetrieved templates/context:\n{retrieval.concatenated_context}\n",
)
```

**Characteristics:**
- Includes `META_REASONER_SYSTEM` with reasoning structure selection
- Chain-of-thought style prompting
- Multiple retrieved templates/context concatenated
- Meta-reasoner enables dynamic reasoning structure selection

---

## 6. Memory Update Differences

### 6.1 Edge Creation in Memory Updates

In `src/graph_bot/pipelines/stream_loop.py`, the `_insert_solution_template` function:

```python
# 2. Identify source edges if retrieval was used
edges: list[ReasoningEdge] = []
if mode != "bot" and retrieval and retrieval.paths:
    for path in retrieval.paths:
        if not path.node_ids:
            continue
        # The template used is effectively the last node in the retrieved path
        source_id = path.node_ids[-1]
        edges.append(
            ReasoningEdge(
                src=source_id,
                dst=f"episode-{problem_id}-solution",
                relation="used_for",
                attributes={...},
            )
        )
```

**Critical Difference:**
- **Flat-template-RAG:** Creates `used_for` edges linking retrieved templates to new solutions
- **Bot Mode:** Skips edge creation (`mode != "bot"`), resulting in flat node insertion only

This means Bot mode produces a **flatter memory graph** without usage tracking edges.

---

## 7. Configuration

### 7.1 CLI Options

From `src/graph_bot/cli.py`:

```python
mode: Optional[str] = typer.Option(
    None,
    "--mode",
    help="Execution mode: graph_bot, graph_bot_exec, io, cot, bot, got, tot, flat_template_rag",
)

use_edges: Optional[bool] = typer.Option(
    None, "--use-edges", help="Use graph edges for path construction"
)
```

### 7.2 Settings

From `src/graph_bot/settings.py`:

```python
mode: str = Field(
    default="graph_bot",
    description="Execution mode: graph_bot, graph_bot_exec, io, cot, bot, got, tot, flat_template_rag",
)

top_k_paths: int = Field(default=3)  # Used by flat_template_rag

use_edges: bool = Field(
    default=True, description="Use graph edges for path construction"
)
```

### 7.3 Bot-specific Config

File: `configs/bot.yaml`:

```yaml
buffer:
  index_dir: "data/processed/bot_index"
  top_k: 5
  embedder: "BAAI/bge-m3"
inference:
  use_templates: true
  strict_schema: false
```

---

## 8. Decision Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│                    User selects mode                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
   ┌───────────────┐               ┌───────────────┐
   │   bot mode    │               │ flat_template_│
   │               │               │    rag mode   │
   └───────┬───────┘               └───────┬───────┘
           │                               │
           ▼                               ▼
   ┌───────────────┐               ┌───────────────┐
   │ Transform to  │               │ Use directly  │
   │ flat_template_│               │               │
   │    rag        │               │               │
   └───────┬───────┘               └───────┬───────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
           ┌───────────────────────────────┐
           │  Retrieval Configuration      │
           │  • bot: k=1 path              │
           │  • flat_template_rag: k=3     │
           └───────────────┬───────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
   ┌───────────────┐               ┌───────────────┐
   │   Bot Prompt  │               │ Flat-template │
   │  • Should use │               │ RAG Prompt    │
   │    same Meta- │               │  • Uses Meta- │
   │    Reasoner   │               │    Reasoner   │
   │  • Currently  │               │  • Multiple   │
   │    simplified │               │    templates  │
   │    (NEEDS FIX)│               │    (k=3)      │
   │  • No meta-   │               │ RAG Prompt    │
   │    reasoning  │               │  • With meta- │
   │  • Single tpl │               │    reasoner   │
   │  • Distilled  │               │  • Multiple   │
   │    metadata   │               │    templates  │
   └───────┬───────┘               └───────┬───────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
           ┌───────────────────────────────┐
           │      Memory Update            │
           │  • bot: No edges created      │
           │  • flat_template_rag:         │
           │    used_for edges created     │
           └───────────────────────────────┘
```

---

## 9. Use Case Recommendations

- You want a **simplified BoT baseline** with fewer templates (k=1 vs k=3)
- Single-template retrieval is sufficient for your use case
- You prefer flatter memory graphs without usage edges
- Testing minimal RAG configuration (but with full Meta-Reasoner capabilities)

**Note:** Bot mode should still use the Complex Meta-Reasoner prompt, just with reduced retrieval count.
- You want a **simplified BoT baseline** without meta-reasoning overhead
- Single-template retrieval is sufficient
- You prefer flatter memory graphs without usage edges
- Testing minimal RAG configuration

### Choose **Flat-template-RAG Mode** when:
- You want **diverse template retrieval** (k > 1)
- Meta-reasoning structure selection is beneficial
- You need **usage tracking edges** for graph analytics
- Full BoT pipeline capabilities are required

---

## 10. Interview Questions for Users

To help users select the appropriate mode, ask:

### Q1: Retrieval Diversity
> "Do you need multiple diverse reasoning templates (flat_template_rag) or is a single best template sufficient (bot)?"

- **Multiple templates** → Flat-template-RAG
- **Single template** → Bot mode
- User answer. retrieval시 Top-K 개의 candidate을 받아와서 또 n개의 template을 LLM input query에 이어붙이려고 함.

### Q2: Meta-Reasoning
> "Does your task benefit from dynamic reasoning structure selection (prompt-based, procedure-based, programming-based)?"

- **Yes, dynamic selection helps** → Both Flat-template-RAG and Bot mode (both use Meta-Reasoner)
- **No, fixed approach is fine** → Consider baseline modes like `io` or `cot`

- **Yes, dynamic selection helps** → Flat-template-RAG
- **No, fixed approach is fine** → Bot mode

- User answer. Dynamic approach, 우리 논문 식으로는 Online continual stream에서 계속 진행하려는 거 맞음.

### Q3: Memory Graph Complexity
> "Do you need to track which templates were used for which solutions (usage edges)?"

- **Yes, track template usage** → Flat-template-RAG
- **No, flat storage is acceptable** → Bot mode
- User answer. graph_bot mode는 EMA update를 진행하기 때문에 template usage를 track해야 하지만 flat-template-rag나 bot mode 둘 다 단순 cosine similarity만 사용하기 때문에 필요X.

### Q4: Query Distillation
> "Do you have query normalization/clustering requirements?"

- Both modes support distillation, but Bot mode explicitly preserves original/distilled metadata
- User answer. distilled query만 있으면 됨. 모든 query를 보존하면 token을 너무 많이 먹어서.

### Q5: Baseline vs. Full Pipeline
> "Are you running a minimal baseline comparison or the full Graph-Bot pipeline?"

- **Minimal baseline** → Bot mode
- **Full pipeline** → Flat-template-RAG
- User answer. Baseline 확보용 + parameter 바꿔가면서 실험용.

### Q6: Task Type
> "What type of reasoning does your task involve?"

- **Common Sense / Scheduling** → Either (both support prompt-based)
- **Creative / Text** → Flat-template-RAG (procedure-based structure)
- **Math / Code** → Flat-template-RAG (programming-based structure)
- User answer. 다양한 task에서 성능 관찰 진행할 거임.

---

## 11. Code References

| Component | File | Key Lines |
|-----------|------|-----------|
| Mode Capabilities | `src/graph_bot/pipelines/stream_loop.py` | 33-42 |
| Mode Transformation | `src/graph_bot/pipelines/stream_loop.py` | 175-180 |
| k_paths Logic | `src/graph_bot/pipelines/stream_loop.py` | 241-242 |
| Path Retrieval | `src/graph_bot/adapters/graphrag.py` | 684-706 |
| Edge Creation | `src/graph_bot/pipelines/stream_loop.py` | 854-870 |
| Bot Prompt | `src/graph_bot/tasks/game24.py` | 118-149 |
| Default Prompt | `src/graph_bot/tasks/game24.py` | 151-155 |
| Settings | `src/graph_bot/settings.py` | 50-64 |
| CLI Options | `src/graph_bot/cli.py` | 377-384 |

---

## 12. Summary

| Feature | Flat-template-RAG | Bot Mode (Current) | Bot Mode (Intended) |
|---------|-------------------|----------|
| **Internal Mode** | `flat_template_rag` | Transforms to `flat_template_rag` | Same |
| **k_paths** | `settings.top_k_paths` (default 3) | Hardcoded 1 | Hardcoded 1 |
| **Graph Edges** | Disabled | Disabled | Disabled |
| **Meta-Reasoning** | Enabled (Meta-Reasoner) | **Disabled** (incorrect) | **Enabled** (should match Flat-template-RAG) |
| **Prompt Style** | Complex (CoT + Meta) | Simple (Direct) | **Complex (CoT + Meta)** |
| **Memory Edges** | `used_for` edges created | No edges created | No edges created |
| **Use Case** | Full BoT pipeline | Minimal baseline | Baseline with Meta-Reasoner |

**Bottom Line:** Bot mode is intended to be a retrieval-minimal version of Flat-template-RAG (k=1 instead of k=3), but should share the same Meta-Reasoner prompt architecture. The current implementation incorrectly uses a simplified prompt - this should be fixed to align with the original BoT design from `buffer-of-thought-llm`.
