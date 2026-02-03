#!/bin/bash
set -e

# Sensitivity Analysis: Baseline IO/CoT x Retry 1/3
# Created for Task 7.1

# 1. Environment Configuration
export GRAPH_BOT_LLM_PROVIDER=openai
export GRAPH_BOT_LLM_MODEL=gpt-4o-mini
export GRAPH_BOT_EXECUTION_TIMEOUT_SEC=60
export GRAPH_BOT_PRICING_PATH=/home/hyunwoo/git/graph-bot/configs/pricing/pricing_v0.yaml

DATA="/home/hyunwoo/git/graph-bot/data/game24_buffer_98.shuffle_1.jsonl"
OUT_DIR="/home/hyunwoo/git/graph-bot/outputs/stream_logs"
mkdir -p "$OUT_DIR"

echo ">>> Launching 4 experiments in parallel..."

# Run 1: IO, Retry 1
GRAPH_BOT_RETRY_MAX_ATTEMPTS=1 nohup graph-bot stream "$DATA" \
    --run-id "baseline_io_retry1" \
    --metrics-out-dir "$OUT_DIR" \
    --mode io \
    --validator-mode oracle > "$OUT_DIR/baseline_io_retry1.log" 2>&1 &
echo "Started: baseline_io_retry1 (Retry Max Attempts: 1)"

# Run 2: IO, Retry 3
GRAPH_BOT_RETRY_MAX_ATTEMPTS=3 nohup graph-bot stream "$DATA" \
    --run-id "baseline_io_retry3" \
    --metrics-out-dir "$OUT_DIR" \
    --mode io \
    --validator-mode oracle > "$OUT_DIR/baseline_io_retry3.log" 2>&1 &
echo "Started: baseline_io_retry3 (Retry Max Attempts: 3)"

# Run 3: CoT, Retry 1
GRAPH_BOT_RETRY_MAX_ATTEMPTS=1 nohup graph-bot stream "$DATA" \
    --run-id "baseline_cot_retry1" \
    --metrics-out-dir "$OUT_DIR" \
    --mode cot \
    --validator-mode oracle > "$OUT_DIR/baseline_cot_retry1.log" 2>&1 &
echo "Started: baseline_cot_retry1 (Retry Max Attempts: 1)"

# Run 4: CoT, Retry 3
GRAPH_BOT_RETRY_MAX_ATTEMPTS=3 nohup graph-bot stream "$DATA" \
    --run-id "baseline_cot_retry3" \
    --metrics-out-dir "$OUT_DIR" \
    --mode cot \
    --validator-mode oracle > "$OUT_DIR/baseline_cot_retry3.log" 2>&1 &
echo "Started: baseline_cot_retry3 (Retry Max Attempts: 3)"

echo "All experiments launched in background. Check $OUT_DIR for .log files."
