#!/bin/bash
# Generate transferable feedbacks for all-wrong MathVerse problems (v2 prompt)
# 4 shards, 1 GPU each — same pattern as run_mathverse_data_pipeline.sh
set -euo pipefail

BASE=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder
cd "$BASE"

export TORCH_COMPILE_DISABLE=1
export VLLM_TORCH_COMPILE_LEVEL=0

echo "[$(date)] Starting feedback v2 generation (4 shards)..."

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD python feedback/step2_gen_feedback_mathverse.py \
        --shard $SHARD --num_shards 4 &
    sleep 60
done
wait

echo "[$(date)] Done!"
echo "Results:"
wc -l mathverse_feedback_v2/feedbacks_shard*.jsonl
