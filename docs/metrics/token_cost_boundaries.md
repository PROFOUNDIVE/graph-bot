# Token & Cost Tracking Boundaries

**Status:** Draft (v0)  
**Date:** 2025-01-20  
**Context:** Distinguishing between "Reasoning Cost" and "Total System Cost" for Graph-BoT benchmarks.

## 1. Metric Definitions

We track costs in two distinct tracks to separate "Reasoning Efficiency" from "Infrastructure Overhead".

| Metric Track | Definition | Scope (Components) | Purpose |
| :--- | :--- | :--- | :--- |
| **Track A: Pipeline-Only** | Direct cost of the reasoning engine producing the solution. | • Main Agent (Distillation, Instantiation)<br>• Thought Generation (GoT)<br>• Self-Correction loops | Measure **Algorithm Efficiency**.<br>How efficiently does the logic solve the problem? |
| **Track B: Total Cost** | All costs incurred during the execution of a run. | • **Pipeline-Only** components<br>• RAG Internal Calls (Keyword/Entity Extraction)<br>• Embeddings (VectorDB queries)<br>• Auxiliary LLMs (Evaluators, Summarizers)<br>• Retries & Failures | Measure **Real-World Viability**.<br>How much does it actually cost to run? |

## 2. Component Categorization

Based on codebase analysis (`graph-bot`, `buffer-of-thought-llm`, `graph-of-thoughts`).

### What included in "Pipeline-Only"
*   **Core Reasoning**: Calls in `bot_pipeline.py` (Distill, Instantiate), `GoT` Controller thoughts.
*   **Effective Retries**: Retries that are part of the final successful chain (algorithm-level backtracks).

### What excluded from "Pipeline-Only" (Counted in Total Cost)
*   **Infrastructure / RAG**:
    *   `LightRAG` internal operations (Keyword extraction, Entity summarization).
    *   `meta_buffer.py` hardcoded calls (often `gpt-4o` for context retrieval).
*   **Embeddings**:
    *   `all-MiniLM-L6-v2` or OpenAI embedding calls for VectorDB lookup.
*   **Evaluators**:
    *   GoT "Oracle Scorer" calls used strictly for benchmarks/validation.
*   **System Failures**:
    *   5xx API errors, timeouts (wasted tokens).

## 3. Aggregation Rules

1.  **Granularity**: Costs are aggregated per `run_id` (single problem execution).
2.  **Parent-Child Spans**:
    *   A `run_id` represents one problem solving session.
    *   A `span_id` represents a specific step (e.g., "Retrieve Context", "Generate Thought").
    *   **RAG Aggregation**: `LightRAG` internal events (fine-grained) must share a common `parent_id` (the high-level "Retrieve" op). This allows collapsing them into a single cost item during analysis while preserving detail.
    *   All logs must carry `run_id` to allow aggregation.
3.  **Failures**:
    *   **System Failures** (5xx, timeout): Included in **Track B (Total Cost)**.
    *   **Algorithmic Retries** (Self-Correction): Included in **Track A (Pipeline Cost)** as they reflect the reasoning engine's efficiency.
4.  **Caching**:
    *   Cache hits (KV Cache, Semantic Cache) count as **0 cost** but must log `tokens_saved` in the usage field.

## 4. Known Implementation Gaps (to be hooked)

*   **`LightRAG` Internals**: Currently untracked. Needs hooks in `lightrag/kg/`.
*   **`meta_buffer.py`**: Hardcoded `gpt-4o` calls need to pass through the tracker.
*   **GoT Reset Bug**: Ensure token counters reset between episodes.

## 5. JSONL Log Examples

```json
{"timestamp": "2025-01-20T10:00:01Z", "run_id": "run_123", "event_type": "embedding", "component": "rag_infra", "model": "text-embedding-3-small", "usage": {"prompt_tokens": 512, "completion_tokens": 0, "total_tokens": 512}, "cost_usd": 0.00001, "latency_ms": 45}
{"timestamp": "2025-01-20T10:00:05Z", "run_id": "run_123", "event_type": "llm_completion", "component": "pipeline", "model": "gpt-4o-mini", "usage": {"prompt_tokens": 1200, "completion_tokens": 150, "total_tokens": 1350}, "cost_usd": 0.00021, "latency_ms": 1200}
{"timestamp": "2025-01-20T10:00:08Z", "run_id": "run_123", "event_type": "llm_completion", "component": "auxiliary", "model": "gpt-4o", "usage": {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250}, "cost_usd": 0.0015, "latency_ms": 800}
```

## 6. Implementation Plan

### Hook Locations
1.  **Wrapper Class**: Create `TokenTrackingClient` in `graph_bot/adapters/` to wrap `vllm/openai` calls.
2.  **LightRAG Patch**: Decorate `lightrag.query()` and `lightrag.insert()` methods.
3.  **BoT Pipeline**: Inject tracker into `Pipeline.get_respond()` in `bot_pipeline.py`.
4.  **Config Refactor**: Modify `meta_buffer.py` to read model names from a config file (e.g., `config.yaml`) instead of hardcoding `gpt-4o`.

### Aggregation Logic (5-Line Summary)
1.  Load JSONL file into Pandas DataFrame.
2.  Filter by `run_id` for specific session analysis.
3.  Group by `component` (`pipeline` vs `total`).
4.  Sum `cost_usd` and `latency_ms` per group.
5.  Output `Pipeline Cost` (Track A) and `Total Cost` (Track B).

### Aux LLM Definition
> **Aux LLM Calls**: Any LLM invocation *not* directly producing the solution text but required for system operation (e.g., self-evaluators, summarizers for memory, query re-writers). These belong to Track B (Total Cost).
