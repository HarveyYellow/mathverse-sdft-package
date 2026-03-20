#!/bin/bash
# Evaluate only epoch-level checkpoints:
#   - Base model (greedy + sampling)
#   - 4 GRPO epoch checkpoints: step 17, 34, 51, 68 (greedy + sampling)
#   - 8 SD checkpoints: 4 models × 2 epochs (greedy + sampling)
#
# Skips models that already have results (checks for _greedy.json / _sampling.json)
#
# Usage: nohup bash SD_folder/scripts/run_eval_epoch_only.sh \
#            > /inspire/sfs/project/inf-multimodal/public/jingyuanhuang/eval_epoch_only.log 2>&1 &

set -euo pipefail

BASE=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder
VERL_ROOT=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export DS_BUILD_OPS=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/.triton_cache
export HF_DATASETS_CACHE=/home/ma-user/work/hf_cache
export TORCH_COMPILE_DISABLE=1
export VLLM_TORCH_COMPILE_LEVEL=0

BASE_MODEL=/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct
SD_OUT=$BASE/outputs
GRPO_CKPT=$VERL_ROOT/examples/grpo_trainer/checkpoints/verl_grpo_mathverse/qwen3_vl_8b_grpo_mathverse
RESULTS_DIR=$BASE/eval_results
EVAL_PY=$BASE/evaluation/eval_mathverse_vllm.py

mkdir -p "$RESULTS_DIR"

# ==================== Helper: evaluate one model (greedy + sampling) ====================
eval_one() {
    local label="$1"
    local model_dir="$2"
    local need_processor="$3"  # "yes" or "no"

    echo ""
    echo ">>> Evaluating: $label (greedy + sampling)"

    local proc_arg=""
    if [ "$need_processor" = "yes" ]; then
        proc_arg="--processor_path $BASE_MODEL"
    fi

    # Greedy
    if [ -f "$RESULTS_DIR/${label}_greedy.json" ]; then
        echo "  [$label] greedy already exists, skipping."
    else
        python3 "$EVAL_PY" \
            --model_path "$model_dir" \
            --label "$label" \
            --output_dir "$RESULTS_DIR" \
            --mode greedy \
            $proc_arg
    fi

    # Sampling
    if [ -f "$RESULTS_DIR/${label}_sampling.json" ]; then
        echo "  [$label] sampling already exists, skipping."
    else
        python3 "$EVAL_PY" \
            --model_path "$model_dir" \
            --label "$label" \
            --output_dir "$RESULTS_DIR" \
            --mode sampling \
            $proc_arg
    fi

    echo "  $label done."
}

# ============================================================
#  1. Base model
# ============================================================
echo ""
echo "######################################################"
echo "  Evaluating Base Model"
echo "######################################################"

eval_one "base_qwen3vl8b" "$BASE_MODEL" "no"

# ============================================================
#  2. GRPO epoch checkpoints (step 17, 34, 51, 68)
# ============================================================
echo ""
echo "######################################################"
echo "  Evaluating GRPO Epoch Checkpoints (step 17,34,51,68)"
echo "######################################################"

cd "$VERL_ROOT"

for step_num in 17 34 51 68; do
    step_dir="$GRPO_CKPT/global_step_${step_num}"
    merged_dir="$GRPO_CKPT/global_step_${step_num}_hf"
    label="grpo_epoch$((step_num / 17))_step${step_num}"

    if [ ! -d "$step_dir" ]; then
        echo "  WARNING: $step_dir not found, skipping."
        continue
    fi

    # Merge FSDP -> HF if not already done
    if [ ! -d "$merged_dir" ] || [ ! -f "$merged_dir/config.json" ]; then
        echo "  Merging global_step_${step_num} ..."
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$step_dir/actor" \
            --target_dir "$merged_dir"
        echo "  global_step_${step_num} merged."
    fi

    eval_one "$label" "$merged_dir" "yes"
done

# ============================================================
#  3. SD checkpoints (4 models × 2 epochs)
# ============================================================
echo ""
echo "######################################################"
echo "  Evaluating SD Checkpoints"
echo "######################################################"

eval_one "sd_ref_only_trunc_ep1" "$SD_OUT/mathverse_ref_only_trunc/checkpoint-35" "yes"
eval_one "sd_ref_only_trunc_ep2" "$SD_OUT/mathverse_ref_only_trunc/checkpoint-70" "yes"
eval_one "sd_ref_ans_trunc_ep1"  "$SD_OUT/mathverse_ref_ans_trunc/checkpoint-35"  "yes"
eval_one "sd_ref_ans_trunc_ep2"  "$SD_OUT/mathverse_ref_ans_trunc/checkpoint-70"  "yes"
eval_one "sd_ref_only_full_ep1"  "$SD_OUT/mathverse_ref_only_full/checkpoint-35"  "yes"
eval_one "sd_ref_only_full_ep2"  "$SD_OUT/mathverse_ref_only_full/checkpoint-70"  "yes"
eval_one "sd_ref_ans_full_ep1"   "$SD_OUT/mathverse_ref_ans_full/checkpoint-35"   "yes"
eval_one "sd_ref_ans_full_ep2"   "$SD_OUT/mathverse_ref_ans_full/checkpoint-70"   "yes"

# ============================================================
#  SUMMARY
# ============================================================
echo ""
echo "######################################################"
echo "  EVALUATION SUMMARY"
echo "######################################################"

python3 -c "
import json, glob, os
results_dir = '$RESULTS_DIR'

# Only show epoch-level results
targets = [
    'base_qwen3vl8b',
    'grpo_epoch1_step17', 'grpo_epoch2_step34', 'grpo_epoch3_step51', 'grpo_epoch4_step68',
    'sd_ref_only_trunc_ep1', 'sd_ref_only_trunc_ep2',
    'sd_ref_ans_trunc_ep1', 'sd_ref_ans_trunc_ep2',
    'sd_ref_only_full_ep1', 'sd_ref_only_full_ep2',
    'sd_ref_ans_full_ep1', 'sd_ref_ans_full_ep2',
]

models = {}
for f in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
    with open(f) as fh:
        r = json.load(fh)
    label = r.get('label', '')
    if label not in targets:
        continue
    if label not in models:
        models[label] = {}
    if r.get('mode') == 'sampling':
        models[label]['top1'] = r.get('accuracy_top1', 0)
        models[label]['top8'] = r.get('accuracy_top8', 0)
        models[label]['total'] = r.get('total', 0)
    else:
        models[label]['greedy'] = r.get('accuracy', 0)
        models[label]['total'] = r.get('total', 0)

print(f\"{'Model':<35} {'Greedy':>8} {'Top1(t=1)':>10} {'Top8(t=1)':>10}\")
print('-' * 65)
for label in targets:
    if label not in models:
        print(f'{label:<35} {\"  (missing)\":>8}')
        continue
    m = models[label]
    g = f\"{m.get('greedy', 0):.2%}\" if 'greedy' in m else '  -'
    t1 = f\"{m.get('top1', 0):.2%}\" if 'top1' in m else '  -'
    t8 = f\"{m.get('top8', 0):.2%}\" if 'top8' in m else '  -'
    print(f'{label:<35} {g:>8} {t1:>10} {t8:>10}')
"

echo ""
echo "All done! Results in: $RESULTS_DIR"
