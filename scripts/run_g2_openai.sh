#!/bin/bash
set -e

# Check for API Key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set."
    echo "Please set it: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Configuration
export GRAPH_BOT_LLM_PROVIDER=vllm
export GRAPH_BOT_LLM_BASE_URL=https://api.openai.com/v1
export GRAPH_BOT_LLM_API_KEY=$OPENAI_API_KEY
export GRAPH_BOT_LLM_MODEL=gpt-4o-mini
export GRAPH_BOT_EXECUTION_TIMEOUT_SEC=60
export GRAPH_BOT_PRICING_PATH=configs/pricing/pricing_v0.yaml

# Data Selection
# Preference: data/game24.jsonl -> tests/fixtures/game24_smoke.jsonl
if [ -f "data/game24.jsonl" ]; then
    INPUT_DATA="data/game24.jsonl"
else
    echo "Warning: data/game24.jsonl not found. Falling back to smoke test data."
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
    --input-path "$INPUT_DATA" \
    --run-id "$RUN_ID" \
    --metrics-out-dir "$METRICS_DIR" \
    --max-problems 10

# 2. Amortization (Log Analysis)
echo "Execution finished. Starting amortization..."
graph-bot amortize \
    --metrics-dir "$METRICS_DIR" \
    --run-id "$RUN_ID"

echo "Done. Results available in $METRICS_DIR"
