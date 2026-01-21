# Latency Reporting & Outlier Policy

## 1. Latency Metrics
To ensure robust performance analysis despite experimental variance, we adopt the following reporting standards:

*   **Primary Metric**: **Median (p50)**
    *   Reason: `mean` is heavily skewed by occasional timeouts or system hiccups. `median` represents the typical user experience.
*   **Secondary Metric**: **95th Percentile (p95)**
    *   Reason: captures the "worst-case" performance for the majority of users, excluding extreme outliers.
*   **Format**: `p50 / p95` (e.g., `12.5s / 45.2s`)
*   **Prohibition**: Do not report `mean` alone without `median`.

## 2. Outlier Handling
*   **Definition**:
    *   **System Outliers**: Latency > 10x median.
    *   **Interactive/Hanging**: Latency capped by the Timeout limit (e.g., 60s).
*   **Policy**: **Include All Data (Do NOT Discard)**
    *   Since we use **Median (p50)** and **p95**, extreme outliers (caused by `input()` waits or loops) will not skew the primary metric significantly.
    *   **Do NOT** exclude failed or timed-out attempts from the dataset unless the run crashed entirely (N count mismatch).
    *   Timed-out runs should be recorded with the timeout value (e.g., 60s) or marked as failed, but kept in the N count.
*   **Notation**:
    *   Report `p50` and `p95` as is.
    *   If N is insufficient: `Latency (N=XX/100)`

## 3. Report Table Template
All experimental results must follow this schema to align with Cost/Token definitions (from Block 1).

| Method | Accuracy | Attempt success rate | Tokens/problem | API cost | Latency(problem) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Method Name** | `00.0%` | `00.0%` | `1.5k / 2.1k`<br>*(Pipeline / Total)* | `$0.01 / $0.015`<br>*(Pipeline / Total)* | `12.3s / 18.5s`<br>*(p50 / p95)* | `(N=98)` |

*   **Tokens/Cost Definitions** (Ref: `docs/metrics/token_cost_boundaries.md`):
    *   **Pipeline**: Core logic (Reasoning/Generation) + **Effective Retries** (algorithmic backtracks).
    *   **Total**: Pipeline + RAG + Embedding + Aux LLM calls + **System Failures** (5xx, timeouts).
