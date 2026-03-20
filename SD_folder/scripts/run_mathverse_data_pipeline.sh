#!/bin/bash
# Mathverse Self-Distillation Data Pipeline
# Step 1: Evaluate train set (8 samples per question)
# Step 2: Generate reflections for all-wrong problems (truncated + full versions)
# Step 3: Build training datasets
#
# Usage: bash SD_folder/scripts/run_mathverse_data_pipeline.sh

set -euo pipefail
export TORCH_COMPILE_DISABLE=1
export VLLM_TORCH_COMPILE_LEVEL=0

BASE=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder
MODEL=/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct

cd "$BASE"

# ==================== Step 1: Evaluate training set ====================
echo ""
echo "=========================================="
echo "  Step 1: Evaluate training set (n=8)"
echo "=========================================="

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD python data_pipeline/step1_eval_mathverse.py \
        --shard $SHARD --num_shards 4 &
done
wait

echo ""
echo "  Step 1 summary:"
python3 -c "
import json, glob
top1 = top8 = total = all_wrong = 0
for f in sorted(glob.glob('mathverse_eval_results_vllm/eval_shard*.jsonl')):
    with open(f) as fh:
        for line in fh:
            item = json.loads(line)
            total += 1
            if item['top1']: top1 += 1
            if item['top8']: top8 += 1
            if item['num_correct_in_8'] == 0: all_wrong += 1
print(f'  Total: {total}')
print(f'  Top-1: {top1}/{total} = {100*top1/total:.2f}%')
print(f'  Pass@8: {top8}/{total} = {100*top8/total:.2f}%')
print(f'  All-wrong (need reflection): {all_wrong}')
print(f'  Has correct (use shortest CoT): {total - all_wrong}')
"

# ==================== Step 2a: Generate reflections (truncated) ====================
echo ""
echo "=========================================="
echo "  Step 2a: Generate reflections (truncated)"
echo "=========================================="

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD python data_pipeline/step2_gen_reflection_mathverse.py \
        --shard $SHARD --num_shards 4 --truncate &
done
wait

echo "  Step 2a done!"

# ==================== Step 2b: Generate reflections (full, no truncation) ====================
echo ""
echo "=========================================="
echo "  Step 2b: Generate reflections (no truncation)"
echo "=========================================="

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD python data_pipeline/step2_gen_reflection_mathverse.py \
        --shard $SHARD --num_shards 4 --no-truncate &
done
wait

echo "  Step 2b done!"

# ==================== Step 3: Build training datasets ====================
echo ""
echo "=========================================="
echo "  Step 3: Build training datasets"
echo "=========================================="

echo "  Building truncated version..."
python data_pipeline/step3_build_mathverse_train.py \
    --reflection_dir mathverse_reflection_n8_truncated \
    --suffix _trunc

echo ""
echo "  Building full (no truncation) version..."
python data_pipeline/step3_build_mathverse_train.py \
    --reflection_dir mathverse_reflection_n8_full \
    --suffix _full

echo ""
echo "=========================================="
echo "  Pipeline complete!"
echo "  Outputs:"
echo "    Truncated:  data/mathverse_sdft_ref_only_trunc/train"
echo "                data/mathverse_sdft_ref_ans_trunc/train"
echo "    Full:       data/mathverse_sdft_ref_only_full/train"
echo "                data/mathverse_sdft_ref_ans_full/train"
echo "=========================================="
