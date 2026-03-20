#!/bin/bash
# ============================================================================
# Ablation: feedback vs no-feedback
# Step 1: MathVerse no-feedback (ans_only)
# Step 2: Geo3k no-feedback (ans_only)
# Step 3: Geo3k with-feedback (ref_ans, re-train without freeze_vision)
#
# All 3 trainings: 2 epochs, no freeze_vision, batch=64, lr=1e-6
#
# Usage:
#   nohup bash SD_folder/scripts/run_ablation_ans_only.sh \
#       > /inspire/sfs/project/inf-multimodal/public/jingyuanhuang/ablation_ans_only.log 2>&1 &
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

EPOCHS=2

# ==================== Step 0: Build ablation datasets ====================
echo ""
echo "=========================================="
echo "  Step 0: Build ablation datasets"
echo "=========================================="

python3 -c "
from datasets import load_from_disk, Dataset
from collections import Counter

for name, path in [
    ('mathverse', '$BASE/data/mathverse_sdft_ref_ans_trunc/train'),
    ('geometry3k', '$BASE/data/geometry3k_sdft_ref_ans/train'),
]:
    ds = load_from_disk(path)
    print(f'=== {name}: {len(ds)} samples ===')
    print(f'  Original: {Counter(ds[\"teacher_type\"])}')

    records = []
    changed = 0
    for i in range(len(ds)):
        item = dict(ds[i])
        if item['teacher_type'] in ('reflection', 'reflection_with_answer'):
            item['teacher_type'] = 'answer_only'
            item['teacher_extra'] = ''
            changed += 1
        records.append(item)

    out_dir = f'$BASE/data/{name}_sdft_ans_only_ablation/train'
    new_ds = Dataset.from_list(records)
    new_ds.save_to_disk(out_dir)
    print(f'  Changed {changed} reflection -> answer_only')
    print(f'  After:   {Counter(new_ds[\"teacher_type\"])}')
    print(f'  Saved to {out_dir}')
    print()
"

echo "  Datasets ready."

# ==================== Step 1: Train MathVerse ablation ====================
echo ""
echo "=========================================="
echo "  Step 1: Train MathVerse ans_only_ablation (2 epochs)"
echo "=========================================="

rm -rf /home/ma-user/work/hf_cache/* 2>/dev/null || true

accelerate launch \
    --config_file config/accelerate_config_gqa.yaml \
    training/main_vlm_mathverse_qwen3_no_feedback.py \
    --model_name "$MODEL" \
    --data_dir data/mathverse_sdft_ans_only_ablation \
    --output_dir outputs/mathverse_ans_only_ablation \
    --num_train_epochs $EPOCHS \
    --learning_rate 1e-6 \
    --per_device_batch_size 1 \
    --global_batch_size 64 \
    --report_to none

echo "  MathVerse ablation done!"

# ==================== Step 2: Train Geo3k ablation ====================
echo ""
echo "=========================================="
echo "  Step 2: Train Geo3k ans_only_ablation (2 epochs)"
echo "=========================================="

rm -rf /home/ma-user/work/hf_cache/* 2>/dev/null || true

accelerate launch \
    --config_file config/accelerate_config_gqa.yaml \
    training/main_vlm_geometry3k_no_feedback.py \
    --model_name "$MODEL" \
    --data_dir data/geometry3k_sdft_ans_only_ablation \
    --output_dir outputs/geo3k_ans_only_ablation \
    --num_train_epochs $EPOCHS \
    --learning_rate 1e-6 \
    --per_device_batch_size 1 \
    --global_batch_size 64 \
    --report_to none

echo "  Geo3k ans_only ablation done!"

# ==================== Step 3: Train Geo3k with feedback (no freeze_vision) ====================
echo ""
echo "=========================================="
echo "  Step 3: Train Geo3k ref_ans (with feedback, no freeze_vision, 2 epochs)"
echo "=========================================="

rm -rf /home/ma-user/work/hf_cache/* 2>/dev/null || true

accelerate launch \
    --config_file config/accelerate_config_gqa.yaml \
    training/main_vlm_geometry3k.py \
    --model_name "$MODEL" \
    --data_dir data/geometry3k_sdft_ref_ans \
    --output_dir outputs/geo3k_ref_ans_no_freeze \
    --num_train_epochs $EPOCHS \
    --learning_rate 1e-6 \
    --per_device_batch_size 1 \
    --global_batch_size 64 \
    --report_to none

echo "  Geo3k ref_ans (no freeze) done!"

# ==================== Summary ====================
echo ""
echo "=========================================="
echo "  All training complete!"
echo "=========================================="
echo "  1. MathVerse ans_only:        outputs/mathverse_ans_only_ablation/"
echo "  2. Geo3k ans_only:            outputs/geo3k_ans_only_ablation/"
echo "  3. Geo3k ref_ans (no freeze): outputs/geo3k_ref_ans_no_freeze/"
echo ""
echo "  Compare with:"
echo "    MathVerse ref_ans_trunc:     outputs/mathverse_ref_ans_trunc/  (已有, 无freeze)"
echo "    Geo3k ref_ans (old):         outputs/geo3k_ref_ans/            (已有, 有freeze)"
