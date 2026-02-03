#!/bin/bash
set -e

# EXP2: Warm-start Formal Experiment
# Goal: Compare Cold Start vs Warm Start (Seed 10)
# Output: outputs/stream_logs/exp2_formal_*

# Configuration
export GRAPH_BOT_LLM_PROVIDER=openai
export GRAPH_BOT_LLM_MODEL=gpt-4o-mini
export GRAPH_BOT_EXECUTION_TIMEOUT_SEC=60
export GRAPH_BOT_PRICING_PATH=configs/pricing/pricing_v0.yaml

DATA_FULL="data/game24_buffer_98.shuffle_1.jsonl"
DATA_SEED="data/exp2_seed_10.jsonl"
DATA_ONLINE="data/exp2_online_88.jsonl"

# 0. Prepare Data Splits
if [ ! -f "$DATA_SEED" ]; then
    echo "Creating data splits..."
    head -n 10 "$DATA_FULL" > "$DATA_SEED"
    tail -n +11 "$DATA_FULL" > "$DATA_ONLINE"
fi

# 1. Cold Start Run (Baseline)
# Uses fresh metagraph for the whole 88 problems
echo ">>> Running EXP2 Cold Start (Online set 88 items, Empty Memory)..."
export GRAPH_BOT_METAGRAPH_PATH="outputs/metagraphs/exp2_formal_cold.json"
rm -f "$GRAPH_BOT_METAGRAPH_PATH" # Ensure fresh start

graph-bot stream \
    "$DATA_ONLINE" \
    --run-id "exp2_formal_cold" \
    --metrics-out-dir "outputs/stream_logs" \
    --mode graph_bot \
    --use-edges \
    --policy-id semantic_topK_stats_rerank \
    --validator-mode oracle \
    --validator-gated-update

# 2. Warm Start - Phase A: Seed Build
echo ">>> Running EXP2 Warm Start - Phase A (Seed 10 items)..."
export GRAPH_BOT_METAGRAPH_PATH="outputs/metagraphs/exp2_formal_warm.json"
rm -f "$GRAPH_BOT_METAGRAPH_PATH" # Ensure fresh start

graph-bot stream \
    "$DATA_SEED" \
    --run-id "exp2_formal_warm_seed" \
    --metrics-out-dir "outputs/stream_logs" \
    --mode graph_bot \
    --use-edges \
    --policy-id semantic_topK_stats_rerank \
    --validator-mode oracle \
    --validator-gated-update

# 3. Warm Start - Phase B: Online Run
echo ">>> Running EXP2 Warm Start - Phase B (Online set 88 items, Pre-filled Memory)..."
# GRAPH_BOT_METAGRAPH_PATH is already set to the one populated by Phase A (outputs/metagraphs/exp2_formal_warm.json)
# It will Auto-Load.

graph-bot stream \
    "$DATA_ONLINE" \
    --run-id "exp2_formal_warm_online" \
    --metrics-out-dir "outputs/stream_logs" \
    --mode graph_bot \
    --use-edges \
    --policy-id semantic_topK_stats_rerank \
    --validator-mode oracle \
    --validator-gated-update

echo "EXP2 Formal Run Complete."
echo "Logs in outputs/stream_logs/exp2_formal_*"
