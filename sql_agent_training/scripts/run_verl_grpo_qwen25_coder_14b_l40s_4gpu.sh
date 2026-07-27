#!/usr/bin/env bash
set -euo pipefail

# Experimental verl GRPO entrypoint for 14B Qwen Coder LoRA on one 4x L40S node.
# This script may be launched from the workspace root or from the inner project root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

resolve_project_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${PROJECT_DIR}/$1" ;;
  esac
}

MODEL_PATH="$(resolve_project_path "${MODEL_PATH:-data/models/Qwen2.5-Coder-14B-Instruct}")"
LORA_ADAPTER_PATH="$(resolve_project_path "${LORA_ADAPTER_PATH:-artifacts/checkpoints/sft_qwen25_coder_14b_lora_h100_zero2/20260725_061113/checkpoint-300}")"
TRAIN_FILES="$(resolve_project_path "${TRAIN_FILES:-data/verl_spider/train.parquet}")"
VAL_FILES="$(resolve_project_path "${VAL_FILES:-data/verl_spider/validation.parquet}")"
AGENT_LOOP_CONFIG_PATH="$(resolve_project_path "${AGENT_LOOP_CONFIG_PATH:-configs/verl_sql_agent_loop.yaml}")"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
FILTER_OVERLONG_PROMPTS_WORKERS=${FILTER_OVERLONG_PROMPTS_WORKERS:-1}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_PP=${ROLLOUT_PP:-1}
MODEL_NUM_ATTENTION_HEADS=${MODEL_NUM_ATTENTION_HEADS:-40}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.32}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${ROLLOUT_MAX_MODEL_LEN}}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1}
ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON:-True}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-1}
ACTOR_LR=${ACTOR_LR:-5e-7}
USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-False}
USE_KL_LOSS=${USE_KL_LOSS:-False}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-25}
MAX_TURNS=${MAX_TURNS:-3}
PROJECT_NAME=${PROJECT_NAME:-sql_agent_training}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-verl_grpo_qwen25_coder_14b_l40s_4gpu}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-16}
RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD:-False}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-1073741824}
ACTOR_USE_TORCH_COMPILE=${ACTOR_USE_TORCH_COMPILE:-False}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-False}
MODEL_USE_REMOVE_PADDING=${MODEL_USE_REMOVE_PADDING:-False}
MODEL_ATTN_IMPLEMENTATION=${MODEL_ATTN_IMPLEMENTATION:-sdpa}
DATA_TRUST_REMOTE_CODE=${DATA_TRUST_REMOTE_CODE:-False}
MODEL_TRUST_REMOTE_CODE=${MODEL_TRUST_REMOTE_CODE:-False}
ENABLE_GPU_MONITOR=${ENABLE_GPU_MONITOR:-True}
GPU_MONITOR_INTERVAL_SEC=${GPU_MONITOR_INTERVAL_SEC:-10}

if (( MODEL_NUM_ATTENTION_HEADS % ROLLOUT_TP != 0 )); then
  echo "ERROR: ROLLOUT_TP=${ROLLOUT_TP} must divide MODEL_NUM_ATTENTION_HEADS=${MODEL_NUM_ATTENTION_HEADS}."
  echo "For Qwen2.5-Coder-14B, legal TP values include 1, 2, 4, 5, 8, 10, 20, 40."
  exit 2
fi

if [[ "${ROLLOUT_PP}" != "1" ]]; then
  echo "ERROR: current verl vLLM rollout does not support ROLLOUT_PP > 1."
  echo "Use ROLLOUT_PP=1, or move rollout to a separate vLLM deployment / use more GPUs with a legal TP size."
  exit 2
fi

export CUDA_VISIBLE_DEVICES
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

echo "verl PROJECT_DIR=${PROJECT_DIR}"
echo "verl RAY_NUM_CPUS=${RAY_NUM_CPUS} RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY} RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD}"
echo "verl TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} ROLLOUT_N=${ROLLOUT_N}"
echo "verl MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
echo "verl ROLLOUT_TP=${ROLLOUT_TP} ROLLOUT_PP=${ROLLOUT_PP} ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION} ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN}"
echo "verl ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS} ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON}"
echo "verl USE_KL_IN_REWARD=${USE_KL_IN_REWARD} USE_KL_LOSS=${USE_KL_LOSS} KL_LOSS_COEF=${KL_LOSS_COEF}"
echo "verl ACTOR_USE_TORCH_COMPILE=${ACTOR_USE_TORCH_COMPILE} ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER}"
echo "verl ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD} ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD} REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD}"
echo "verl MODEL_USE_REMOVE_PADDING=${MODEL_USE_REMOVE_PADDING} MODEL_ATTN_IMPLEMENTATION=${MODEL_ATTN_IMPLEMENTATION}"
echo "verl DATA_TRUST_REMOTE_CODE=${DATA_TRUST_REMOTE_CODE} MODEL_TRUST_REMOTE_CODE=${MODEL_TRUST_REMOTE_CODE}"
echo "verl ENABLE_GPU_MONITOR=${ENABLE_GPU_MONITOR} GPU_MONITOR_INTERVAL_SEC=${GPU_MONITOR_INTERVAL_SEC}"

DATA=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}"
  data.train_files="['${TRAIN_FILES}']"
  data.val_files="['${VAL_FILES}']"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.return_raw_chat=True
  data.filter_overlong_prompts=True
  data.filter_overlong_prompts_workers="${FILTER_OVERLONG_PROMPTS_WORKERS}"
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}"
  data.truncation=error
  data.trust_remote_code="${DATA_TRUST_REMOTE_CODE}"
)

MODEL=(
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code="${MODEL_TRUST_REMOTE_CODE}"
  actor_rollout_ref.model.use_remove_padding="${MODEL_USE_REMOVE_PADDING}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.lora_rank=64
  actor_rollout_ref.model.lora_alpha=128
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}"
  +actor_rollout_ref.model.override_config._attn_implementation="${MODEL_ATTN_IMPLEMENTATION}"
)

ACTOR=(
  actor_rollout_ref.actor.strategy=fsdp
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}"
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.use_remove_padding="${MODEL_USE_REMOVE_PADDING}"
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.use_torch_compile="${ACTOR_USE_TORCH_COMPILE}"
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.pipeline_model_parallel_size="${ROLLOUT_PP}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  ++actor_rollout_ref.rollout.layered_summon="${ROLLOUT_LAYERED_SUMMON}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}"
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG_PATH}"
  actor_rollout_ref.rollout.agent.default_agent_loop=sql_agent
  actor_rollout_ref.rollout.agent.num_workers="${TRAIN_BATCH_SIZE}"
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}"
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURNS}"
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
)

REF=(
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}"
  actor_rollout_ref.ref.fsdp_config.use_torch_compile="${ACTOR_USE_TORCH_COMPILE}"
)

REWARD=(
  reward.num_workers="${REWARD_NUM_WORKERS}"
)

TRAINER=(
  trainer.balance_batch=True
  trainer.logger='["console"]'
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.nnodes=1
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.val_before_train=False
)

RAY_INIT=(
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}"
  +ray_kwargs.ray_init.include_dashboard="${RAY_INCLUDE_DASHBOARD}"
  +ray_kwargs.ray_init.object_store_memory="${RAY_OBJECT_STORE_MEMORY}"
)

RAY_RUNTIME=()
if [[ "${ENABLE_RAY_RUNTIME_ENV:-0}" == "1" ]]; then
  RAY_RUNTIME=(
    +ray_kwargs.ray_init.runtime_env.working_dir="${PROJECT_DIR}"
    '+ray_kwargs.ray_init.runtime_env.excludes=["data/**","artifacts/**","logs/**",".venv/**",".venv-vllm/**",".uv_cache/**",".xdg_cache/**","*.safetensors","*.sqlite"]'
  )
fi

GPU_MONITOR_PID=""
start_gpu_monitor() {
  case "${ENABLE_GPU_MONITOR}" in
    1|true|True|TRUE|yes|Yes|YES) ;;
    *) return 0 ;;
  esac
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU_MONITOR_DISABLED nvidia-smi not found"
    return 0
  fi
  echo "GPU_MONITOR_HEADER timestamp,index,memory_used_mb,memory_total_mb,gpu_utilization_pct"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits |
        while IFS= read -r line; do
          echo "GPU_MONITOR ${line}"
        done
      sleep "${GPU_MONITOR_INTERVAL_SEC}"
    done
  ) &
  GPU_MONITOR_PID="$!"
}

stop_gpu_monitor() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
}

trap stop_gpu_monitor EXIT
trap 'stop_gpu_monitor; exit 130' INT
trap 'stop_gpu_monitor; exit 143' TERM
start_gpu_monitor
RUN_START_TIME=${SECONDS}

set +e
python -m verl.trainer.main_ppo \
  "${DATA[@]}" \
  "${MODEL[@]}" \
  "${ACTOR[@]}" \
  "${ROLLOUT[@]}" \
  "${REF[@]}" \
  "${REWARD[@]}" \
  "${TRAINER[@]}" \
  "${RAY_INIT[@]}" \
  "${RAY_RUNTIME[@]}" \
  "$@"
status=$?
set -e

echo "verl RUN_WALL_TIME_SEC=$((SECONDS - RUN_START_TIME))"
exit "${status}"
