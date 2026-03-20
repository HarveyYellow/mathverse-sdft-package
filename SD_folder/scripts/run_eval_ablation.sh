#!/bin/bash
# Evaluate MathVerse ablation (ans_only) checkpoints: ep1 + ep2, greedy + sampling
# Usage: nohup bash /scratch/jh19696/run_eval_ablation.sh > /scratch/jh19696/eval_ablation.log 2>&1 &

set -euo pipefail
cd /scratch/jh19696

export TORCH_COMPILE_DISABLE=1
export VLLM_TORCH_COMPILE_LEVEL=0

MODEL_BASE=/lscratch/jh19696/Qwen3-VL-8B-Instruct
ABLATION_DIR=/scratch/jh19696/Self-Distillation/outputs/mathverse_ans_only_ablation
OUT_DIR=/scratch/jh19696/eval_mathverse_results

# ==================== ep1 (checkpoint-35) ====================
echo "[$(date)] Evaluating ep1 greedy..."
python eval_mathverse_vllm.py \
    --model_path $ABLATION_DIR/checkpoint-35 \
    --processor_path $MODEL_BASE \
    --label sd_ans_only_ablation_ep1 \
    --output_dir $OUT_DIR \
    --mode greedy

echo "[$(date)] Evaluating ep1 sampling..."
python eval_mathverse_vllm.py \
    --model_path $ABLATION_DIR/checkpoint-35 \
    --processor_path $MODEL_BASE \
    --label sd_ans_only_ablation_ep1 \
    --output_dir $OUT_DIR \
    --mode sampling

# ==================== ep2 (checkpoint-70) ====================
echo "[$(date)] Evaluating ep2 greedy..."
python eval_mathverse_vllm.py \
    --model_path $ABLATION_DIR/checkpoint-70 \
    --processor_path $MODEL_BASE \
    --label sd_ans_only_ablation_ep2 \
    --output_dir $OUT_DIR \
    --mode greedy

echo "[$(date)] Evaluating ep2 sampling..."
python eval_mathverse_vllm.py \
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
