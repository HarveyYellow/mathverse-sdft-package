#!/bin/bash
# ============================================================================
# MathVerse Self-Distillation — NO online feedback (ablation)
# Uses DistilTrainer (not FeedbackDistilTrainer)
# Data: ans_only_ablation (reflection_with_answer → answer_only)
#
# Usage:
#   nohup bash SD_folder/scripts/run_mathverse_no_feedback.sh \
#       > /inspire/sfs/project/inf-multimodal/public/jingyuanhuang/train_mathverse_no_feedback.log 2>&1 &
# ============================================================================

set -euo pipefail

BASE=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder
MODEL=/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct

cd "$BASE"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export DS_BUILD_OPS=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/.triton_cache
export HF_DATASETS_CACHE=/home/ma-user/work/hf_cache

# trl compatibility: use patched trl (0.23 + missing modules for distil_trainer)
export PYTHONPATH=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/trl_patched:${BASE}/core:${PYTHONPATH:-}

rm -rf /home/ma-user/work/hf_cache/* 2>/dev/null || true

accelerate launch \
    --config_file config/accelerate_config_gqa.yaml \
    training/main_vlm_mathverse_qwen3_no_feedback.py \
    --model_name "$MODEL" \
    --data_dir data/mathverse_sdft_ans_only_ablation \
    --output_dir outputs/mathverse_ans_only_ablation \
    --num_train_epochs 2 \
    --learning_rate 1e-6 \
    --per_device_batch_size 1 \
    --global_batch_size 64 \
    --report_to none

echo "Done! Model saved to outputs/mathverse_ans_only_ablation/"
