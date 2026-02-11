# Graph-Bot v0.3 Architecture Summary

**Status:** Draft (v0.3)

This document summarizes the architectural changes and features introduced in Graph-Bot v0.3, focusing on dual distillation, online edge connectivity, and budget-aware context management.

## 1) Dual Distillation Loop

The v0.3 pipeline implements a symmetric distillation loop for both input (queries) and output (solution traces).

1.  **Query Distill:** Raw user queries are distilled into structured `DistilledQuery` objects to improve retrieval precision.
2.  **Retrieve:** The system performs semantic top-K retrieval from the persistent MetaGraph using the distilled query.
3.  **Solve:** An LLM solves the problem, conditioned on the retrieved reasoning paths.
4.  **Trace Distill:** The raw solution trace is distilled into atomic `ReasoningNode` and `ReasoningEdge` primitives.
5.  **Insert:** New nodes are merged into the MetaGraph with deduplication logic.
6.  **Edge Connect:** Online edges are established between newly inserted nodes and existing nodes to maintain structural connectivity.

## 2) Online Edges & Connectivity

Unlike previous versions that relied on static hierarchies, v0.3 introduces **Online Edge Connection**:
- **Causal Edges:** Represent logical dependencies discovered during Trace Distillation.
- **Structural Edges:** Connect nodes within the same reasoning episode (ReasoningTree) or problem session.
- **Dynamic Connectivity:** Edges are added during the `Insert` phase based on co-occurrence and explicit causal hints.

## 3) Budget Fairness (Packing Logic)

To handle context window constraints efficiently, v0.3 implements **Budget Packing**:
- **`packed_context_tokens`:** A metric tracking the total tokens consumed by retrieved reasoning paths in the prompt.
- **Context Management:** Multiple reasoning paths are packed into the solver prompt up to a configured token limit, ensuring "fair" distribution of context budget across retrieved templates.

## 4) Causal Graph Ablation (EXP5)

The v0.3 architecture supports **Causal Graph Ablation** through the `use_edges` configuration:
- **`use_edges=True`:** Retrieval and scoring leverage graph connectivity (edges) to find coherent paths.
- **`use_edges=False`:** System falls back to purely node-based semantic retrieval (node-pool mode), ignoring structural dependencies.
- This allows for systematic ablation studies (EXP5) to quantify the value of graph structure over flat retrieval.

## 5) Key Metrics in v0.3

- **`edges_added_count`:** Number of new edges established during the `Insert` and `Edge Connect` phase.
- **`packed_context_tokens`:** Total token count of the context injected from the MetaGraph into the solver prompt.
- **`contamination_rate` (updated):** Tracks how many retrieved nodes were already present in the ground truth or previous successful paths for the current problem.

---

## 6) Current Implementation Details

### 6.1) Retrieval Mechanism

The retrieval pipeline (`graphrag.py:retrieve_paths`) operates in two stages:

**Stage 1: Semantic Candidate Selection**
```
query_tokens = tokenize(query.question)
node_scores = { node_id: jaccard_similarity(query_tokens, node.text) for node in graph.nodes }
seed_nodes = top_n(node_scores, k=rerank_top_n)
```

- Uses **Jaccard similarity** (token overlap) for initial semantic matching.
- **No dense embeddings** are used in the current implementation despite `embedding_model` setting.

**Stage 2: Path Scoring & Re-ranking**
```
combined_score = avg_semantic + avg_ema_success + avg_edge_ema - length_penalty - cost_penalty
```

Where:
- `avg_semantic`: Mean Jaccard similarity of nodes in the path.
- `avg_ema_success`: Mean EMA success rate from node statistics.
- `avg_edge_ema`: Mean EMA success rate of edge statistics (if edges exist).
- `length_penalty`: 0.02 * (path_length - 1).
- `cost_penalty`: avg_cost_usd * 0.01.

### 6.2) EMA Update Mechanism

Node statistics are updated after each problem attempt:
- `n_seen`, `n_used`, `n_success`, `n_fail`: Counters.
- `ema_success`: Exponential moving average of success rate.
- `avg_tokens`, `avg_latency_ms`, `avg_cost_usd`: Running averages.

### 6.3) Mode Configuration

| Mode | Behavior |
|------|----------|
| `flat_template_rag` | Single-node paths only (no edge traversal) |
| `use_edges=True` | Multi-hop path building via edge adjacency |
| `use_edges=False` | Node-pool retrieval (no edges) |
| `policy_id=semantic_only` | Ignores EMA scores, uses semantic similarity only |

---

## 7) Current Limitations

### 7.1) No Dense Embedding Retrieval
- Despite `embedding_model: all-MiniLM-L6-v2` in settings, the current implementation uses **Jaccard token overlap** for similarity.
- Dense vector similarity (cosine similarity on embeddings) is **not implemented**.
- This makes retrieval sensitive to exact token matches and less robust to paraphrasing.

### 7.2) Edge Creation Dependency on Retrieval Hit
- Edges are only created when `retrieval.paths` is non-empty (see `stream_loop.py:646-663`).
- Cold-start scenarios produce isolated nodes with no inter-node connections.
- Graph structure benefits are not realized until retrieval starts hitting.

### 7.3) No Graph Traversal Beyond 1-Hop
- Current path building uses simple BFS from seed nodes.
- Deep multi-hop reasoning chains are not explored.

### 7.4) Contamination Accumulation
- Validator prevents new bad insertions but does not remove existing contaminated nodes.
- Steady-state contamination remains high (~75-82%) even with validation enabled.

---

## 8) Future Work

### 8.1) Dense Embedding Integration
- Replace Jaccard similarity with dense vector similarity using sentence-transformers.
- Enable semantic retrieval that handles paraphrasing and synonyms.

### 8.2) Hybrid Scoring
- Combine dense embeddings (semantic) with sparse signals (EMA success, cost) in a learned re-ranker.

### 8.3) Edge Bootstrap Strategies
- Create semantic similarity edges during warm-start seeding.
- Fallback edge generation for cold-start scenarios.

### 8.4) Memory Pruning
- Implement decay-based node/edge removal for low-quality or stale entries.
- Maintain graph health over long-running streams.

### 8.5) LLM-based Distillation
- Implement prompt-based distillation to replace rule-based trace extraction.
- Enable domain-agnostic template generation through LLM reasoning.
- Key enabler for extending to domains beyond Game24 (text reasoning, math word problems, etc.).

### 8.6) Weak Validator / LLM-Judge
- Support domains without cheap oracle validators (e.g., open-ended generation).
- Implement LLM-as-a-judge for quality assessment when ground truth is unavailable.
- Critical for EXP6 (domain extension) experiments.

### 8.7) Code-augmented Execution Loop
- Extend solver to generate and execute code (Program-of-Thought style).
- Implement sandboxed execution environment (e.g., Docker-based).
- Capture execution traces as reasoning primitives for graph insertion.
- Bridges gap with BoT (Code-aug) baseline performance.
