#!/bin/bash
set -e

set -o pipefail

# Configuration
# Preferred: use repo-local .env (GRAPH_BOT_* variables).
# Avoid mapping from OPENAI_API_KEY to prevent accidental empty override.
export GRAPH_BOT_LLM_PROVIDER=openai
export GRAPH_BOT_LLM_BASE_URL=https://api.openai.com/v1
export GRAPH_BOT_LLM_MODEL=gpt-4o-mini

if [ -z "$GRAPH_BOT_LLM_API_KEY" ]; then
    echo "Error: GRAPH_BOT_LLM_API_KEY is not set."
    echo "Set it in .env (recommended) or export it in your shell."
    exit 1
fi
export GRAPH_BOT_EXECUTION_TIMEOUT_SEC=60
export GRAPH_BOT_PRICING_PATH=configs/pricing/pricing_v0.yaml

# Data Selection
# Preference: data/game24_buffer_98.shuffle_1.jsonl -> tests/fixtures/game24_smoke.jsonl
if [ -f "data/game24_buffer_98.shuffle_1.jsonl" ]; then
    INPUT_DATA="data/game24_buffer_98.shuffle_1.jsonl"
else
    echo "Warning: data/game24_buffer_98.shuffle_1.jsonl not found. Falling back to smoke test data."
    INPUT_DATA="tests/fixtures/game24_smoke.jsonl"
fi

RUN_ID="g2_openai_1"
METRICS_DIR="outputs/stream_logs"

echo "Starting graph-bot stream execution..."
echo "Model: $GRAPH_BOT_LLM_MODEL"
echo "Input: $INPUT_DATA"
echo "Run ID: $RUN_ID"

# 1. Stream Execution
graph-bot stream \
    "$INPUT_DATA" \
    --run-id "$RUN_ID" \
    --metrics-out-dir "$METRICS_DIR" \
    --mode graph_bot \
    --use-edges \
    --policy-id semantic_topK_stats_rerank \
    --validator-mode oracle \
    --max-problems 10

# 2. Amortization (Log Analysis)
echo "Execution finished. Starting amortization..."
graph-bot amortize \
    "$METRICS_DIR/$RUN_ID.stream.jsonl" \
    --out "$METRICS_DIR/$RUN_ID.amortization.csv"

echo "Done. Results available in $METRICS_DIR"
