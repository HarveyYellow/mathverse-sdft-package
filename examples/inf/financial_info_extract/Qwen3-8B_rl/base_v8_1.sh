set -x

export NCCL_SOCKET_IFNAME=ens2f5
export GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
export NCCL_NET_PLUGIN=none
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=15
export NCCL_DEBUG=INFO
export MKL_THREADING_LAYER=GNU
# handle ray error, but will slightly decrease training speed
# export HYDRA_FULL_ERROR=1
# export RAY_IGNORE_UNHANDLED_ERRORS=1

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
export MKL_NUM_THREADS=16

# dependency: vllm>=0.11.0, megatron-lm>=0.13, mbridge with qwen3vl_cp branch
# environment option1: use a stable container later than docker://verlai/verl:vllm011.dev6 
    # and install mbridge in it by following the instruction in the container
            # pip remove mbridge if you have installed it
            # pip install git+https://github.com/ISEEKYAN/mbridge.git@qwen3vl_cp # for correct mbridge
# environment option2: use container docker://verlai/verl:vllm011.dev_qwenvl_cp

export VLLM_ALLREDUCE_USE_SYMM_MEM=0 # for vllm0.11.0 with TP
export PYTHONPATH="/home/ma-user/work/data_mllm/swift_share/Megatron-LM":$PYTHONPATH

NNODES=${MA_NUM_HOSTS:-"1"}
NODE_RANK="$VC_TASK_INDEX"
MASTER_DOMAIN_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_IP_ADDR=$(python -c "import socket; print(socket.gethostbyname('$MASTER_DOMAIN_ADDR'))" 2>/dev/null)
MASTER_PORT="6379"
NGPUS_PER_NODE="$MA_NUM_GPUS"

echo "=== node info in modelarts ==="
echo "node number: ${NNODES}"
echo "node rank: ${NODE_RANK}"
echo "master domain address: ${MASTER_DOMAIN_ADDR}"
echo "master ip address: ${MASTER_IP_ADDR}"
echo "master port: ${MASTER_PORT}"
echo "number of gpus per node: ${NGPUS_PER_NODE}"
echo "=============================="

HF_MODEL_PATH="/home/ma-user/work/fengjianming/work/LLMs/Qwen3-8B"
max_model_len=40960
train_path="/home/ma-user/work/fengjianming/work/INF-FinDocBench/train/v8_1_8b_dense_rl/train.parquet"
project_name="financial_info_extract"
reward_fn_path="examples/inf/reward_functions/financial_info_extract.py"
current_script="$(realpath "$0")"
script_basename=$(basename "$current_script")
experiment_name="${script_basename%.*}"
experiment_dir="${1:-${project_name}}/${experiment_name}"
export SWANLAB_LOG_DIR="${experiment_dir}/swanlog"

# start ray cluster
if [ "$NODE_RANK" = "0" ]; then
    echo "start head node..."
    # use MASTER_IP_ADDR instead of MASTER_DOMAIN_ADDR to accelarate training
    ray start --head \
        --node-ip-address=${MASTER_IP_ADDR} \
        --port=${MASTER_PORT} \
        --num-gpus=${NGPUS_PER_NODE}

    if [ ${NNODES} -gt 1 ]; then
        echo "waiting for worker nodes to join..."
        sleep 60
    fi
else
    echo "waiting for head node..."
    sleep 30

    MAX_RETRIES=30
    RETRY_COUNT=0
    echo "try to join head node"
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if ray start --address="${MASTER_IP_ADDR}:${MASTER_PORT}"; then
            echo "join head node ${MASTER_IP_ADDR} successfully"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo "join head node ${MASTER_IP_ADDR} failed, retry $RETRY_COUNT/$MAX_RETRIES"
            sleep 10
        fi
    done

    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "can not join head node, exit."
        exit 1
    fi
fi

echo "check ray status"
sleep 10
if ray status > /dev/null 2>&1; then
    echo "=== ray status ==="
    ray status
    echo "=================="
else
    echo "check ray status failed"
    exit 1
fi

# start training
if [ "$NODE_RANK" = "0" ]; then
    echo "start training in head node"

    mkdir -p ${experiment_dir}
    chmod -R 777 ${experiment_dir}
    cp -f "$current_script" ${experiment_dir}
    cp -f "$reward_fn_path" ${experiment_dir}

    python -m verl.trainer.main_ppo --config-path=config --config-name='ppo_megatron_trainer.yaml' \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        custom_reward_function.path=${reward_fn_path} \
        custom_reward_function.name=compute_score \
        data.train_files="$train_path" \
        data.val_files="$train_path" \
        data.val_max_samples=1 \
        data.shuffle=True \
        data.seed=42 \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        data.train_batch_size=128 \
        data.max_prompt_length=38912 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=True \
        data.filter_overlong_prompts_workers=128 \
        data.truncation='error' \
        actor_rollout_ref.model.path=$HF_MODEL_PATH \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_model_len \
        actor_rollout_ref.actor.megatron.use_mbridge=True \
        actor_rollout_ref.actor.megatron.param_offload=True \
        actor_rollout_ref.actor.megatron.optimizer_offload=True \
        actor_rollout_ref.actor.megatron.grad_offload=True \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
        actor_rollout_ref.actor.megatron.context_parallel_size=1 \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
        +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
        +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
        +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.max_num_batched_tokens=$max_model_len \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$max_model_len \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$max_model_len \
        actor_rollout_ref.ref.megatron.param_offload=True \
        trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
        trainer.nnodes=${NNODES} \
        trainer.total_epochs=1 \
        trainer.test_freq=-1 \
        trainer.val_before_train=False \
        trainer.default_local_dir=${experiment_dir}/checkpoints \
        trainer.save_freq=200 \
        trainer.logger='["console","swanlab"]' \
        trainer.project_name=${project_name} \
        trainer.experiment_name=${experiment_name} \
        hydra.run.dir=${experiment_dir}/outputs

    echo "finish training"
fi

ray stop
