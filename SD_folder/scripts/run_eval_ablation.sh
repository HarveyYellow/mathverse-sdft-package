#!/bin/bash
# Evaluate MathVerse ablation (ans_only) checkpoints: ep1 + ep2, greedy + sampling
# Usage: nohup bash SD_folder/scripts/run_eval_ablation.sh \
#            > /inspire/sfs/project/inf-multimodal/public/jingyuanhuang/eval_ablation.log 2>&1 &

set -euo pipefail

BASE=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder
MODEL_BASE=/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct

cd "$BASE"

export TORCH_COMPILE_DISABLE=1
export VLLM_TORCH_COMPILE_LEVEL=0

ABLATION_DIR=$BASE/outputs/mathverse_ans_only_ablation
OUT_DIR=$BASE/eval_results

mkdir -p "$OUT_DIR"

# ==================== ep1 (checkpoint-35) ====================
echo "[$(date)] Evaluating ep1 greedy..."
python evaluation/eval_mathverse_vllm.py \
    --model_path $ABLATION_DIR/checkpoint-35 \
    --processor_path $MODEL_BASE \
    --label sd_ans_only_ablation_ep1 \
    --output_dir $OUT_DIR \
    --mode greedy

echo "[$(date)] Evaluating ep1 sampling..."
python evaluation/eval_mathverse_vllm.py \
    --model_path $ABLATION_DIR/checkpoint-35 \
    --processor_path $MODEL_BASE \
    --label sd_ans_only_ablation_ep1 \
    --output_dir $OUT_DIR \
    --mode sampling

# ==================== ep2 (checkpoint-70) ====================
echo "[$(date)] Evaluating ep2 greedy..."
python evaluation/eval_mathverse_vllm.py \
    --model_path $ABLATION_DIR/checkpoint-70 \
    --processor_path $MODEL_BASE \
    --label sd_ans_only_ablation_ep2 \
    --output_dir $OUT_DIR \
    --mode greedy

echo "[$(date)] Evaluating ep2 sampling..."
python evaluation/eval_mathverse_vllm.py \
    --model_path $ABLATION_DIR/checkpoint-70 \
    --processor_path $MODEL_BASE \
    --label sd_ans_only_ablation_ep2 \
    --output_dir $OUT_DIR \
    --mode sampling

# ==================== Summary ====================
echo ""
echo "=========================================="
echo "  All evaluations done!"
echo "=========================================="
echo "Results saved to $OUT_DIR:"
ls -la $OUT_DIR/sd_ans_only_ablation_*.json
