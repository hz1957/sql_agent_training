#!/usr/bin/env bash
set -euo pipefail

# Rollout comparison B: S3 tree-final with branch=8, beam=1, batch=4, and 24 tree slots per root.
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_rollout_cmp_s3_b8_beam1_bs4.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
LOG_DIR="${PROJECT_DIR}/artifacts/logs/verl"
mkdir -p "${LOG_DIR}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-25}"
DATA_SEED="${DATA_SEED:-42}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
PPO_MAX_TOKEN_LEN_PER_GPU=16384
SFT_BASE_MODEL="data/models/Qwen2.5-Coder-14B-Instruct-SFT-Ratio-Gold3200-D1137-R463-LR5e5-R32-Merged"
RUN_NAME="${RUN_NAME:-verl_grpo_rollout_cmp_s3_b8_beam1_base_ratio_gold3200_d1137_r463_lr5e5_r32_14b_h100_4gpu_${TOTAL_TRAINING_STEPS}step_bs4_n24_kl01_ep2_t10_turn3_g09_dyn16k_rmpad_vllmmem020_seed${DATA_SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RUNTIME_CACHE_ROOT="/tmp/g${UID:-0}_$$"
export TMPDIR="${RUNTIME_CACHE_ROOT}/tmp"
export UV_CACHE_DIR="${RUNTIME_CACHE_ROOT}/uv"
export UV_CACHE="${UV_CACHE_DIR}"
export XDG_CACHE_HOME="${RUNTIME_CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${RUNTIME_CACHE_ROOT}/xdg_config"
export HF_HOME="${RUNTIME_CACHE_ROOT}/huggingface"
export TORCH_EXTENSIONS_DIR="${RUNTIME_CACHE_ROOT}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${RUNTIME_CACHE_ROOT}/triton"
export RAY_TMPDIR="/tmp/r${UID:-$$}"
export VLLM_CACHE_ROOT="${RUNTIME_CACHE_ROOT}/vllm"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export SQL_AGENT_FIX_LAYERED_SUMMON=1
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
mkdir -p \
  "${TMPDIR}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${HF_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${RAY_TMPDIR}" \
  "${VLLM_CACHE_ROOT}"

case "${STOP_RAY_FIRST:-1}" in
  1|true|True|TRUE|yes|Yes|YES) uv run --no-sync ray stop -f >/dev/null 2>&1 || true ;;
esac

(
  set -o pipefail
  unset ROLLOUT_N
  echo "START ${RUN_NAME} $(date)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT}"
  echo "ROLLOUT_COMPARISON_CONFIG scheme=s3_tree_final base_model=${SFT_BASE_MODEL} batch_size=4 branch_n=8 beam_size=1 max_turns=3 rollout_n=24 max_actions_per_root=24 kl_coef=0.01 ppo_epochs=2 temperature=1.0 gamma=0.9 dynamic_bsz=True remove_padding=True token_cap_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} rollout_gpu_memory_utilization=0.20 rollout_max_num_seqs=8 rollout_max_num_batched_tokens=8192 data_seed=${DATA_SEED} rollout_seed=${ROLLOUT_SEED} layered_summon=True"

  PYTHONUNBUFFERED=1 VLLM_USE_V1="${VLLM_USE_V1:-1}" \
  EXPERIMENT_NAME="${RUN_NAME}" \
  MODEL_PATH="${SFT_BASE_MODEL}" \
  LORA_ADAPTER_PATH=none \
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
  SAVE_FREQ="${SAVE_FREQ}" \
  TEST_FREQ=-1 \
  TRAIN_BATCH_SIZE=4 \
  PPO_MINI_BATCH_SIZE=4 \
  PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
  PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  LOG_PROB_USE_DYNAMIC_BSZ=True \
  MODEL_USE_REMOVE_PADDING=True \
  MODEL_ATTN_IMPLEMENTATION=sdpa \
  DATA_SEED="${DATA_SEED}" \
  ROLLOUT_SEED="${ROLLOUT_SEED}" \
  FILTER_OVERLONG_PROMPTS=False \
  ROLLOUT_TP=4 \
  ROLLOUT_PP=1 \
  ROLLOUT_TEMPERATURE=1.0 \
  ROLLOUT_TOP_P=1.0 \
  ROLLOUT_TOP_K=-1 \
  ROLLOUT_DO_SAMPLE=True \
  ROLLOUT_GPU_MEMORY_UTILIZATION=0.20 \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192 \
  ROLLOUT_MAX_NUM_SEQS=8 \
  ROLLOUT_LAYERED_SUMMON=True \
  MAX_TURNS=3 \
  MAX_PROMPT_LENGTH=2048 \
  MAX_RESPONSE_LENGTH=2048 \
  GRPO_REWARD_SCHEME=tree_final \
  GRPO_REWARD_GAMMA=0.9 \
  GRPO_TREE_BRANCH_N=8 \
  GRPO_TREE_BEAM_SIZE=1 \
  GRPO_TREE_BEAM_TAU=1.0 \
  GRPO_TREE_BEAM_EPSILON_RANDOM=0.1 \
  GRPO_TREE_PRUNE_ON_TERMINAL_PROXY=True \
  GRPO_ADV_ESTIMATOR=grpo_tree \
  USE_KL_IN_REWARD=False \
  REF_PARAM_OFFLOAD=True \
  USE_KL_LOSS=True \
  KL_LOSS_COEF=0.01 \
  PPO_EPOCHS=2 \
  ACTOR_LR=5e-5 \
  ACTOR_CHECKPOINT_SAVE_LORA_ONLY=True \
  uv run --no-sync bash "${SCRIPT_DIR}/run_verl_grpo_qwen25_coder_14b_l40s_4gpu.sh"

  status=$?
  echo "EXIT_CODE ${status} $(date)"
  exit "${status}"
) > "${LOG_FILE}" 2>&1 &

pid=$!
echo "RUN_PID ${pid}"
echo "LOG_FILE ${LOG_FILE}"
if tail --help 2>&1 | grep -q -- "--pid"; then
  tail --pid="${pid}" -f "${LOG_FILE}"
else
  tail -f "${LOG_FILE}"
fi
wait "${pid}"
