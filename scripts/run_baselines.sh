#!/bin/bash
set -e

# Baseline Collection: IO and CoT

export GRAPH_BOT_LLM_PROVIDER=openai
export GRAPH_BOT_LLM_MODEL=gpt-4o-mini
export GRAPH_BOT_EXECUTION_TIMEOUT_SEC=60
export GRAPH_BOT_PRICING_PATH=/home/hyunwoo/git/graph-bot/configs/pricing/pricing_v0.yaml

DATA="/home/hyunwoo/git/graph-bot/data/game24_buffer_98.shuffle_1.jsonl"

echo ">>> Running Baseline: IO Prompting..."
graph-bot stream \
    "$DATA" \
    --run-id "baseline_io_formal" \
    --metrics-out-dir "/home/hyunwoo/git/graph-bot/outputs/stream_logs" \
    --mode io \
    --validator-mode oracle

echo ">>> Running Baseline: CoT Prompting..."
graph-bot stream \
    "$DATA" \
    --run-id "baseline_cot_formal" \
    --metrics-out-dir "/home/hyunwoo/git/graph-bot/outputs/stream_logs" \
    --mode cot \
    --validator-mode oracle

echo "Baseline Collection Complete."
