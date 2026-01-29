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
- **Structural Edges:** Connect nodes within the same reasoning tree or problem session.
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
