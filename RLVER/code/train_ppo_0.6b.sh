#!/usr/bin/env bash
set -euo pipefail

############################
# ==== USER CONFIG ====   #
############################
: "${WANDB_TOKEN:=60b3bda8c6fe6553f6091ed9ec277e7da90c004c}"
: "${RUN_NAME:=RL_PPO_EXPERIMENT_01}"
: "${IF_THINK:=False}"   # "True" or "False"

# 强制使用 Instruct 0.6B
: "${BASE_MODEL:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${CRITIC_MODEL:=Qwen/Qwen2.5-0.5B-Instruct}"

# 路径对齐到 Pod 环境
: "${HF_HOME:=/checkpoints/jingquaw-sandbox/.cache/huggingface}"
: "${HF_HUB_CACHE:=/checkpoints/jingquaw-sandbox/.cache/huggingface/hub}"
: "${TRANSFORMERS_CACHE:=/checkpoints/jingquaw-sandbox/.cache/huggingface/transformers}"
: "${HF_DATASETS_CACHE:=/checkpoints/jingquaw-sandbox/.cache/huggingface/datasets}"
: "${XDG_CACHE_HOME:=/checkpoints/jingquaw-sandbox/.cache}"
: "${TORCH_HOME:=/checkpoints/jingquaw-sandbox/.cache/torch}"
: "${PIP_CACHE_DIR:=/checkpoints/jingquaw-sandbox/.cache/pip}"
: "${TRITON_CACHE_DIR:=/checkpoints/jingquaw-sandbox/.triton}"
: "${CUDA_CACHE_PATH:=/checkpoints/jingquaw-sandbox/.nv/ComputeCache}"
: "${TMPDIR:=/checkpoints/jingquaw-sandbox/tmp}"
: "${WANDB_DIR:=/checkpoints/jingquaw-sandbox/wandb}"

# 日志与 ckpt
: "${CKPT_DIR:=/checkpoints/jingquaw-sandbox/verl/${RUN_NAME}}"

# Ray
: "${RAY_ADDRESS:=http://127.0.0.1:8265}"
: "${RAY_TEMP_DIR:=/checkpoints/jingquaw-sandbox/tmp/ray}"

# GPU/NCCL
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${NCCL_P2P_DISABLE:=0}"
: "${NCCL_IB_DISABLE:=0}"
: "${OMP_NUM_THREADS:=8}"

############################
# ==== ENV & PREP ====    #
############################
export WANDB_API_KEY="$WANDB_TOKEN"
export RAY_record_ref_creation_sites=1
export HYDRA_FULL_ERROR=1
# driver 侧可以关掉 tokenizer 线程；不放进 Ray job env，避免冲突
export TOKENIZERS_PARALLELISM=false

export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE \
       XDG_CACHE_HOME TORCH_HOME PIP_CACHE_DIR TRITON_CACHE_DIR \
       CUDA_CACHE_PATH TMPDIR WANDB_DIR \
       CUDA_VISIBLE_DEVICES NCCL_P2P_DISABLE NCCL_IB_DISABLE OMP_NUM_THREADS

mkdir -p \
  "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$TMPDIR" "$WANDB_DIR" "$CKPT_DIR" "$RAY_TEMP_DIR"

# 轻量依赖
pip install -q --disable-pip-version-check torchdata || true

echo "[INFO] Base model:    $BASE_MODEL"
echo "[INFO] Critic model:  $CRITIC_MODEL"
echo "[INFO] CKPT dir:      $CKPT_DIR"
echo "[INFO] Ray address:   $RAY_ADDRESS"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

#####################################
# ==== Ensure local Ray head ====== #
#####################################
if ! curl -s "${RAY_ADDRESS}/api/version" >/dev/null; then
  echo "[INFO] Ray dashboard not reachable; starting local head..."
  ray stop --force || true
  ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_ADDRESS##*:}" \
    --num-cpus=128 \
    --num-gpus=8 \
    --temp-dir="$RAY_TEMP_DIR"
  until curl -s "${RAY_ADDRESS}/api/version" >/dev/null; do
    echo "[wait] Ray dashboard not ready yet..."; sleep 1;
  done
  echo "[INFO] Ray dashboard is ready."
fi

########################################
# ==== Build Ray runtime env JSON ==== #
########################################
RUNTIME_ENV_JSON="$(mktemp)"
cat > "$RUNTIME_ENV_JSON" <<'JSON'
{
  "env_vars": {
    "WANDB_API_KEY": "__WANDB_API_KEY__",
    "HF_HOME": "__HF_HOME__",
    "HF_HUB_CACHE": "__HF_HUB_CACHE__",
    "TRANSFORMERS_CACHE": "__TRANSFORMERS_CACHE__",
    "HF_DATASETS_CACHE": "__HF_DATASETS_CACHE__",
    "XDG_CACHE_HOME": "__XDG_CACHE_HOME__",
    "TORCH_HOME": "__TORCH_HOME__",
    "PIP_CACHE_DIR": "__PIP_CACHE_DIR__",
    "TRITON_CACHE_DIR": "__TRITON_CACHE_DIR__",
    "CUDA_CACHE_PATH": "__CUDA_CACHE_PATH__",
    "TMPDIR": "__TMPDIR__",
    "WANDB_DIR": "__WANDB_DIR__",
    "HYDRA_FULL_ERROR": "1",
    "RAY_record_ref_creation_sites": "1",
    "CUDA_VISIBLE_DEVICES": "__CUDA_VISIBLE_DEVICES__",
    "NCCL_P2P_DISABLE": "__NCCL_P2P_DISABLE__",
    "NCCL_IB_DISABLE": "__NCCL_IB_DISABLE__",
    "OMP_NUM_THREADS": "__OMP_NUM_THREADS__",
    "VLLM_NO_USAGE_STATS": "1",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"
  },
  "working_dir": "./",
  "pip": ["latex2sympy2", "word2number", "timeout_decorator"]
}
JSON

# 注：不再写入 TOKENIZERS_PARALLELISM，避免和 main_ppo.py 里的 ray.init 冲突
sed -i \
  -e "s|__WANDB_API_KEY__|${WANDB_API_KEY}|g" \
  -e "s|__HF_HOME__|${HF_HOME}|g" \
  -e "s|__HF_HUB_CACHE__|${HF_HUB_CACHE}|g" \
  -e "s|__TRANSFORMERS_CACHE__|${TRANSFORMERS_CACHE}|g" \
  -e "s|__HF_DATASETS_CACHE__|${HF_DATASETS_CACHE}|g" \
  -e "s|__XDG_CACHE_HOME__|${XDG_CACHE_HOME}|g" \
  -e "s|__TORCH_HOME__|${TORCH_HOME}|g" \
  -e "s|__PIP_CACHE_DIR__|${PIP_CACHE_DIR}|g" \
  -e "s|__TRITON_CACHE_DIR__|${TRITON_CACHE_DIR}|g" \
  -e "s|__CUDA_CACHE_PATH__|${CUDA_CACHE_PATH}|g" \
  -e "s|__TMPDIR__|${TMPDIR}|g" \
  -e "s|__WANDB_DIR__|${WANDB_DIR}|g" \
  -e "s|__CUDA_VISIBLE_DEVICES__|${CUDA_VISIBLE_DEVICES}|g" \
  -e "s|__NCCL_P2P_DISABLE__|${NCCL_P2P_DISABLE}|g" \
  -e "s|__NCCL_IB_DISABLE__|${NCCL_IB_DISABLE}|g" \
  -e "s|__OMP_NUM_THREADS__|${OMP_NUM_THREADS}|g" \
  "$RUNTIME_ENV_JSON"

#####################################
# ==== Submit Ray training job ==== #
#####################################
LOG_FILE="$CKPT_DIR/train.log"
set -x
ray job submit --address="$RAY_ADDRESS" \
  --runtime-env-json="$(cat "$RUNTIME_ENV_JSON")" \
  -- \
  PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 \
  python3 -m verl.trainer.main_ppo \
    +data.virtual_dataset_size=32000 \
    +data.val_virtual_dataset_size=320 \
    data.prompt_key=prompt \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=5000 \
    data.max_response_length=10000 \
    data.return_raw_chat=True \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.thinking="$IF_THINK" \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.use_loss_generation_mask=True \
    actor_rollout_ref.rollout.name=vllm_multi_turn_via_chat \
    +actor_rollout_ref.rollout.environment.name=url_environment \
    +actor_rollout_ref.rollout.environment.per_turn_length=5000 \
    +actor_rollout_ref.rollout.environment.max_turns=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=48000 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=48000 \
    critic.optim.lr=5e-6 \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path="$CRITIC_MODEL" \
    critic.ppo_mini_batch_size=32 \
    critic.model.use_remove_padding=False \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    critic.ppo_max_token_len_per_gpu=48000 \
    critic.forward_max_token_len_per_gpu=48000 \
    trainer.project_name=verl \
    trainer.experiment_name="$RUN_NAME" \
    trainer.default_local_dir="$CKPT_DIR" \
    "trainer.logger=[console,wandb]" \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.save_rollout=True \
    trainer.test_freq=999999 \
    trainer.total_epochs=999999 \
    trainer.total_training_steps=1000 \
    2>&1 | tee -a "$LOG_FILE"
set +x

echo "[DONE] Ray job submitted. Log -> $LOG_FILE"
