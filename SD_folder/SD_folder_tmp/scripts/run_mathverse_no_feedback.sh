#!/bin/bash
# ============================================================================
# MathVerse Self-Distillation — NO online feedback (ablation)
# Uses DistilTrainer (not FeedbackDistilTrainer)
# Data: ans_only_ablation (reflection_with_answer → answer_only)
#
# Usage:
#   nohup bash /scratch/jh19696/Self-Distillation/run_mathverse_no_feedback.sh \
#       > /scratch/jh19696/train_mathverse_no_feedback.log 2>&1 &
# ============================================================================

set -euo pipefail
cd /scratch/jh19696/Self-Distillation

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export DS_BUILD_OPS=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/scratch/jh19696/.triton_cache
export HF_DATASETS_CACHE=/scratch/jh19696/datasets/hf_cache

rm -rf /scratch/jh19696/datasets/hf_cache/*

accelerate launch \
    --config_file accelerate_config_gqa.yaml \
    main_vlm_mathverse_qwen3_no_feedback.py \
    --model_name /lscratch/jh19696/Qwen3-VL-8B-Instruct \
    --data_dir data/mathverse_sdft_ans_only_ablation \
    --output_dir outputs/mathverse_ans_only_ablation \
    --num_train_epochs 2 \
    --learning_rate 1e-6 \
    --per_device_batch_size 1 \
    --global_batch_size 64 \
    --report_to none

echo "Done! Model saved to outputs/mathverse_ans_only_ablation/"
