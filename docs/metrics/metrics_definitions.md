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
- **`contamination_rate`**: The proportion of retrieved reasoning steps that match the ground truth or previous successful traces for the current problem.

## 4) Accuracy & Performance

- **`solved`**: Boolean indicating if the problem was solved correctly within the allowed attempts.
- **`attempts`**: Total number of LLM attempts made for a problem.
- **`solved_attempt`**: The index of the attempt that successfully solved the problem (null if not solved).
- **`attempt_success_rate`**: Percentage of successful attempts over total attempts for a problem or stream.
