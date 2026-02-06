#!/bin/bash
set -e

# Baseline repeats: IO/CoT with fixed dataset

export GRAPH_BOT_LLM_PROVIDER=openai
export GRAPH_BOT_LLM_MODEL=gpt-4o-mini
export GRAPH_BOT_EXECUTION_TIMEOUT_SEC=60
export GRAPH_BOT_PRICING_PATH=/home/hyunwoo/git/graph-bot/configs/pricing/pricing_v0.yaml

DATA="/home/hyunwoo/git/graph-bot/data/game24_buffer_98.shuffle_1.jsonl"
OUT_DIR="/home/hyunwoo/git/graph-bot/outputs/stream_logs"
mkdir -p "$OUT_DIR"

RUN_R1="${RUN_R1:-0}"

run_repeat() {
    local mode="$1"
    local retry="$2"
    local run_id="$3"

    echo ">>> Running ${run_id} (mode=${mode}, retry=${retry})"
    GRAPH_BOT_RETRY_MAX_ATTEMPTS="$retry" graph-bot stream "$DATA" \
        --run-id "$run_id" \
        --metrics-out-dir "$OUT_DIR" \
        --mode "$mode" \
        --validator-mode oracle > "$OUT_DIR/${run_id}.log" 2>&1
    echo "Completed: ${run_id}"
}

for rep in 01 02 03 04 05; do
    run_repeat io 3 "baseline_io_r3_rep${rep}"
done

for rep in 01 02 03 04 05; do
    run_repeat cot 3 "baseline_cot_r3_rep${rep}"
done

if [ "$RUN_R1" = "1" ]; then
    for rep in 01 02 03; do
        run_repeat io 1 "baseline_io_r1_rep${rep}"
    done

    for rep in 01 02 03; do
        run_repeat cot 1 "baseline_cot_r1_rep${rep}"
    done
fi

echo "Baseline repeats complete."
