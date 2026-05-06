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

### Canonical Semantics Split (Source of Truth)

Graph-Bot telemetry MUST treat two edge-related surfaces as separate claim axes:

1. **retrieval-phase traversal evidence**: evidence about what retrieval consumed
   before solving.
2. **post-solve persisted edge growth**: evidence about what the system stored or
   accumulated after solving.

The retrieval claim gate MUST use only retrieval-phase traversal evidence. It
MUST NOT use post-solve persisted edge growth as graph-structure-use proof.

Canonical machine-readable slots:

| Claim axis | Canonical meaning | Accepted field(s) / mapping |
| --- | --- | --- |
| `retrieval_path_node_cardinality` | Number of nodes present in the retrieved path(s) consumed by the solve step | Use an explicit `retrieval_path_node_cardinality` field if emitted. Until then, map legacy `reuse_count` to this slot strictly as retrieved-node cardinality, not as stored-edge proof. |
| `retrieval_used_stored_edges` | Whether retrieval traversed one or more persisted graph edges before solve | Use an explicit `retrieval_used_stored_edges` boolean if emitted. Until then, this slot remains **unproven** unless a retrieval-path artifact directly shows multi-node path structure attributable to persisted edges. |
| `persisted_edge_snapshot` | Total persisted edge count in memory after update/snapshot | Map `memory_n_edges` to this slot. |
| `persisted_edges_added_delta` | Number of new persisted edges added during post-solve update | Map `edges_added_count` to this slot. |

Interpretation rules:

- `memory_n_edges` and `edges_added_count` describe persisted graph state or
  growth only.
- `memory_n_edges` and `edges_added_count` are insufficient as graph-structure-use
  proof on their own.
- `packed_context_tokens` may support retrieval analysis, but it is a context-size
  surface and not a direct stored-edge traversal proof by itself.
- If retrieval-phase and persistence-phase signals are both present, reporting and
  diagnostics MUST keep them on separate axes.

- **`retrieval_hit`**: Boolean indicating if retrieval returned any relevant paths.
- **`reuse_count`**: Legacy retrieval-phase path-node cardinality proxy. Count of
  existing MetaGraph nodes retrieved into the current problem context. This is
  not, by itself, proof that persisted graph edges were traversed.
- **`edges_added_count`** (v0.3): Number of new reasoning edges created during the insertion of a new reasoning tree into the MetaGraph.
- **`memory_n_nodes`**: Total number of nodes in the MetaGraph.
- **`memory_n_edges`**: Total number of persisted edges in the MetaGraph after
  update/snapshot. This is not retrieval-phase traversal evidence.

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

**Telemetry Interpretation Constraint**:
- Retrieval traversal claims must be grounded in retrieval-phase evidence.
- Post-solve persisted edge growth must be reported separately.
- Nonzero `memory_n_edges` or nonzero `edges_added_count` alone does not prove
  that retrieval used graph structure during solve.

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
