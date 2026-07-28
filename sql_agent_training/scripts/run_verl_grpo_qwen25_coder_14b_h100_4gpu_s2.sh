#!/usr/bin/env bash
set -euo pipefail

# 4x H100 GRPO Experiment S2: chain-executable reward, temperature 1.0.
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s2.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
LOG_DIR="${PROJECT_DIR}/artifacts/logs/verl"
mkdir -p "${LOG_DIR}"

RUN_NAME="${RUN_NAME:-verl_grpo_s2_chain_executable_14b_h100_4gpu_150step_bs8_n4_t10_turn3_g09_b01_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SLURM_TMPDIR:-/tmp/$USER}/triton_cache}"
export RAY_TMPDIR="${RAY_TMPDIR:-${SLURM_TMPDIR:-/tmp/$USER}/ray}"
mkdir -p "${TRITON_CACHE_DIR}" "${RAY_TMPDIR}"

case "${STOP_RAY_FIRST:-1}" in
  1|true|True|TRUE|yes|Yes|YES) uv run --no-sync ray stop -f >/dev/null 2>&1 || true ;;
esac

(
  set -o pipefail
  echo "START ${RUN_NAME} $(date)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  PYTHONUNBUFFERED=1 VLLM_USE_V1="${VLLM_USE_V1:-1}" \
  EXPERIMENT_NAME="${RUN_NAME}" \
  MODEL_PATH=data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged \
  LORA_ADAPTER_PATH=none \
  TOTAL_TRAINING_STEPS=150 \
  SAVE_FREQ=25 \
  TEST_FREQ=-1 \
  TRAIN_BATCH_SIZE=8 \
  PPO_MINI_BATCH_SIZE=8 \
  PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
  ROLLOUT_N=4 \
  ROLLOUT_TP=4 \
  ROLLOUT_PP=1 \
  ROLLOUT_TEMPERATURE=1.0 \
  ROLLOUT_TOP_P=1.0 \
  ROLLOUT_TOP_K=-1 \
  ROLLOUT_DO_SAMPLE=True \
  ROLLOUT_GPU_MEMORY_UTILIZATION=0.32 \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096 \
  ROLLOUT_MAX_NUM_SEQS=4 \
  ROLLOUT_LAYERED_SUMMON=True \
  MAX_TURNS=3 \
  MAX_PROMPT_LENGTH=2048 \
  MAX_RESPONSE_LENGTH=2048 \
  GRPO_REWARD_SCHEME=chain_executable \
  GRPO_REWARD_GAMMA=0.9 \
  GRPO_EXECUTABLE_FALLBACK_BETA=0.1 \
  GRPO_ADV_ESTIMATOR=grpo_multi_step \
  USE_KL_IN_REWARD=False \
  USE_KL_LOSS=False \
  ACTOR_LR=5e-7 \
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
