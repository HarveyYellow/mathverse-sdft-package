#!/bin/bash
# Train 5 models sequentially on MathVerse with Qwen3-VL-8B:
#   1. Self-Distill ref_only_trunc (2 epochs)
#   2. Self-Distill ref_ans_trunc (2 epochs)
#   3. Self-Distill ref_only_full (2 epochs)
#   4. Self-Distill ref_ans_full (2 epochs)
#   5. GRPO (4 epochs)
#
# Usage: nohup bash SD_folder/scripts/run_all_mathverse_training.sh \
#            > /inspire/sfs/project/inf-multimodal/public/jingyuanhuang/train_mathverse.log 2>&1 &

set -euo pipefail

BASE=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl/SD_folder
VERL_ROOT=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/verl
MODEL=/home/ma-user/work/share_base_models/Qwen3-VL/Qwen3-VL-8B-Instruct

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export DS_BUILD_OPS=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/.triton_cache
export HF_DATASETS_CACHE=/home/ma-user/work/hf_cache

# ==================== Step 1: Self-Distillation ref_only_trunc (2 epochs) ====================
echo ""
echo "=========================================="
echo "  Step 1/5: Self-Distillation ref_only_trunc (2 epochs)"
echo "=========================================="

rm -rf /home/ma-user/work/hf_cache/* 2>/dev/null || true
cd "$BASE"

accelerate launch \
    --config_file config/accelerate_config_gqa.yaml \
    training/main_vlm_mathverse_qwen3.py \
    --model_name "$MODEL" \
    --data_dir data/mathverse_sdft_ref_only_trunc \
    --output_dir outputs/mathverse_ref_only_trunc \
    --num_train_epochs 2 \
    --learning_rate 1e-6 \
    --per_device_batch_size 1 \
    --global_batch_size 64 \
    --report_to none

echo "  ref_only_trunc done!"

# ==================== Step 2: Self-Distillation ref_ans_trunc (2 epochs) ====================
echo ""
echo "=========================================="
echo "  Step 2/5: Self-Distillation ref_ans_trunc (2 epochs)"
echo "=========================================="

rm -rf /home/ma-user/work/hf_cache/* 2>/dev/null || true
cd "$BASE"

accelerate launch \
    --config_file config/accelerate_config_gqa.yaml \
    training/main_vlm_mathverse_qwen3.py \
    --model_name "$MODEL" \
    --data_dir data/mathverse_sdft_ref_ans_trunc \
    --output_dir outputs/mathverse_ref_ans_trunc \
    --num_train_epochs 2 \
    --learning_rate 1e-6 \
    --per_device_batch_size 1 \
    --global_batch_size 64 \
    --report_to none

echo "  ref_ans_trunc done!"

# ==================== Step 3: Self-Distillation ref_only_full (2 epochs) ====================
echo ""
echo "=========================================="
echo "  Step 3/5: Self-Distillation ref_only_full (2 epochs)"
echo "=========================================="

rm -rf /home/ma-user/work/hf_cache/* 2>/dev/null || true
cd "$BASE"

accelerate launch \
    --config_file config/accelerate_config_gqa.yaml \
    training/main_vlm_mathverse_qwen3.py \
    --model_name "$MODEL" \
    --data_dir data/mathverse_sdft_ref_only_full \
    --output_dir outputs/mathverse_ref_only_full \
    --num_train_epochs 2 \
    --learning_rate 1e-6 \
    --per_device_batch_size 1 \
    --global_batch_size 64 \
    --report_to none

echo "  ref_only_full done!"

# ==================== Step 4: Self-Distillation ref_ans_full (2 epochs) ====================
echo ""
echo "=========================================="
echo "  Step 4/5: Self-Distillation ref_ans_full (2 epochs)"
echo "=========================================="

rm -rf /home/ma-user/work/hf_cache/* 2>/dev/null || true
cd "$BASE"

accelerate launch \
    --config_file config/accelerate_config_gqa.yaml \
    training/main_vlm_mathverse_qwen3.py \
    --model_name "$MODEL" \
    --data_dir data/mathverse_sdft_ref_ans_full \
    --output_dir outputs/mathverse_ref_ans_full \
    --num_train_epochs 2 \
    --learning_rate 1e-6 \
    --per_device_batch_size 1 \
    --global_batch_size 64 \
    --report_to none

echo "  ref_ans_full done!"

# ==================== Step 5: GRPO (4 epochs) ====================
echo ""
echo "=========================================="
echo "  Step 5/5: GRPO (4 epochs)"
echo "=========================================="

# Clean caches before GRPO
rm -rf /home/ma-user/work/hf_cache/* 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true

cd "$VERL_ROOT/examples/grpo_trainer"

TRAIN_PARQUET=$HOME/data/mathverse/train.parquet
TEST_PARQUET=$HOME/data/mathverse/test.parquet

# Force regenerate parquet to ensure latest preprocessing
echo "Generating parquet data..."
rm -rf $HOME/data/mathverse
mkdir -p $HOME/data/mathverse
python3 "$VERL_ROOT/examples/data_preprocess/mathverse.py" \
    --local_dataset_path "/inspire/sfs/project/inf-multimodal/public/jingyuanhuang/data/mathverse_SDFT" \
    --local_save_dir $HOME/data/mathverse \
    --model_path $MODEL \
    --max_prompt_length 4096
echo "Parquet data generated."

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_PARQUET \
    data.val_files=$TEST_PARQUET \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_mathverse' \
    trainer.experiment_name='qwen3_vl_8b_grpo_mathverse' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=4 \
    trainer.val_before_train=False

echo "  GRPO done!"

echo ""
echo "=========================================="
echo "  All 5 trainings completed!"
echo "  1. SD ref_only_trunc (2ep): outputs/mathverse_ref_only_trunc"
echo "  2. SD ref_ans_trunc (2ep):  outputs/mathverse_ref_ans_trunc"
echo "  3. SD ref_only_full (2ep):  outputs/mathverse_ref_only_full"
echo "  4. SD ref_ans_full (2ep):   outputs/mathverse_ref_ans_full"
echo "  5. GRPO (4ep):              checkpoints/verl_grpo_mathverse/qwen3_vl_8b_grpo_mathverse"
echo "=========================================="
