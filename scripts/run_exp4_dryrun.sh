#!/bin/bash
set -e

# EXP4: Memory Growth Dry-run
# Goal: Run long stream (N=300+) to check OOM/Timeout

export GRAPH_BOT_LLM_PROVIDER=openai
export GRAPH_BOT_LLM_MODEL=gpt-4o-mini
export GRAPH_BOT_EXECUTION_TIMEOUT_SEC=60
export GRAPH_BOT_PRICING_PATH=/home/hyunwoo/git/graph-bot/configs/pricing/pricing_v0.yaml
export GRAPH_BOT_METAGRAPH_PATH="/home/hyunwoo/git/graph-bot/outputs/metagraphs/exp4_dryrun.json"

DATA_SRC="/home/hyunwoo/git/graph-bot/data/game24_buffer_98.shuffle_1.jsonl"
DATA_LONG="/home/hyunwoo/git/graph-bot/data/game24_long_392.jsonl"

if [ ! -f "$DATA_LONG" ]; then
    echo "Creating long dataset (N~392)..."
    cat "$DATA_SRC" "$DATA_SRC" "$DATA_SRC" "$DATA_SRC" > "$DATA_LONG"
fi

rm -f "$GRAPH_BOT_METAGRAPH_PATH"

echo ">>> Running EXP4 Dry-run (N=392)..."
graph-bot stream \
    "$DATA_LONG" \
    --run-id "exp4_dryrun" \
    --metrics-out-dir "/home/hyunwoo/git/graph-bot/outputs/stream_logs" \
    --mode graph_bot \
    --use-edges \
    --policy-id semantic_topK_stats_rerank \
    --validator-mode oracle \
    --validator-gated-update

echo "EXP4 Dry-run Complete."
