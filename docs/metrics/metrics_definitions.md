# Metrics Definitions

This document defines the metrics used in Graph-Bot experiments and logging.

## 1) Throughput & Latency

- **`latency_ms`**: End-to-end time for a single operation or problem in milliseconds.
- **`latency_total_ms`**: Aggregated latency for a problem across all attempts and operations.

## 2) Cost & Tokens

- **`prompt_tokens`**: Number of tokens in the input prompt.
- **`completion_tokens`**: Number of tokens in the model output.
- **`total_tokens`**: Sum of prompt and completion tokens.
- **`api_cost_usd`**: Estimated cost in USD based on token usage and model pricing.
- **`packed_context_tokens`** (v0.3): The number of tokens from retrieved MetaGraph paths that were injected into the solver's context window.

## 3) Graph & RAG Metrics

- **`retrieval_hit`**: Boolean indicating if retrieval returned any relevant paths.
- **`reuse_count`**: Number of existing nodes/edges from the MetaGraph reused in the current problem.
- **`edges_added_count`** (v0.3): Number of new reasoning edges created during the insertion of a new reasoning tree into the MetaGraph.
- **`memory_n_nodes`**: Total number of nodes in the MetaGraph.
- **`memory_n_edges`**: Total number of edges in the MetaGraph.

### Implementation Details: Edge Creation

**Current Behavior** (see `src/graph_bot/pipelines/stream_loop.py`):

Edges are only created when `retrieval.paths` is non-empty:

```python
edges = []
if retrieval and retrieval.paths:
    for path in retrieval.paths:
        edges.append(ReasoningEdge(src=source_id, dst=new_node, ...))
```

**Consequence**: If retrieval consistently misses (e.g., empty MetaGraph or embedding mismatch), nodes accumulate but edges remain at 0. This was observed in EXP4 where `memory_n_edges=0` despite `memory_n_nodes` growing to 31.

**Known Issue**:
- Cold-start scenarios produce isolated nodes with no inter-node connections.
- Graph structure benefits (path-based retrieval) are not realized until retrieval starts hitting.

**Future Work**:
- Consider fallback edge creation strategies (e.g., semantic similarity edges between new nodes).
- Bootstrap edges during warm-start seeding phase.

- **`contamination_rate`**: The proportion of retrieved nodes where `validator_passed=False`.

### Implementation Details: `contamination_rate`

**Current Calculation** (see `src/graph_bot/pipelines/stream_loop.py`):

```
contamination_rate = contaminated_nodes / reuse_count
```

Where:
- `reuse_count`: Total number of nodes retrieved from MetaGraph paths.
- `contaminated_nodes`: Count of retrieved nodes with `quality.validator_passed == False`.

**Limitations**:
- Measures only **retrieval-time** contamination; does not track contamination at insertion time.
- Validator prevents **new** bad insertions but does not remove **existing** contaminated nodes.
- High steady-state contamination (~75-82%) observed even with validator enabled.

**Future Work**:
- Memory pruning mechanism to remove low-quality nodes over time.
- Decay-based quality scoring for older nodes.

## 4) Accuracy & Performance

- **`solved`**: Boolean indicating if the problem was solved correctly within the allowed attempts.
- **`attempts`**: Total number of LLM attempts made for a problem.
- **`solved_attempt`**: The index of the attempt that successfully solved the problem (null if not solved).
- **`attempt_success_rate`**: Percentage of successful attempts over total attempts for a problem or stream.
