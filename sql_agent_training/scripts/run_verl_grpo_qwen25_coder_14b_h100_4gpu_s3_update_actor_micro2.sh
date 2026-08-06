#!/usr/bin/env bash
set -euo pipefail

# 4x H100 S3 update_actor benchmark: fixed batching, PPO micro batch per GPU = 2.
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_update_actor_micro2.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

resolve_project_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${PROJECT_DIR}/$1" ;;
  esac
}

LOG_DIR="${PROJECT_DIR}/artifacts/logs/verl"
mkdir -p "${LOG_DIR}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
BASELINE_LOG="$(resolve_project_path "${BASELINE_LOG:-artifacts/logs/verl/verl_grpo_s3_kl01_ep2_tree_final_14b_h100_4gpu_150step_bs2_n20_t10_turn3_g09_branch4_beam2_20260729_043123.log}")"

if (( TOTAL_TRAINING_STEPS <= WARMUP_STEPS )); then
  echo "ERROR: TOTAL_TRAINING_STEPS must be greater than WARMUP_STEPS."
  exit 2
fi

RUN_NAME="${RUN_NAME:-verl_grpo_s3_update_actor_micro2_14b_h100_4gpu_${TOTAL_TRAINING_STEPS}step_bs2_n20_ep2_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SLURM_TMPDIR:-/tmp/$USER}/triton_cache}"
export RAY_TMPDIR="${RAY_TMPDIR:-${SLURM_TMPDIR:-/tmp/$USER}/ray}"
export SQL_AGENT_FIX_LAYERED_SUMMON=1
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
mkdir -p "${TRITON_CACHE_DIR}" "${RAY_TMPDIR}"

case "${STOP_RAY_FIRST:-1}" in
  1|true|True|TRUE|yes|Yes|YES) uv run --no-sync ray stop -f >/dev/null 2>&1 || true ;;
esac

(
  set -o pipefail
  unset ROLLOUT_N
  echo "START ${RUN_NAME} $(date)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "UPDATE_ACTOR_BENCHMARK_CONFIG variant=micro2 steps=${TOTAL_TRAINING_STEPS} warmup=${WARMUP_STEPS} train_batch_size=2 ppo_mini_batch_size=2 ppo_micro_batch_size_per_gpu=2 dynamic_bsz=False remove_padding=False ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} branch_n=4 beam_size=2 rollout_n=20 ppo_epochs=2 kl_coef=0.01 layered_summon=True"

  PYTHONUNBUFFERED=1 VLLM_USE_V1="${VLLM_USE_V1:-1}" \
  EXPERIMENT_NAME="${RUN_NAME}" \
  MODEL_PATH=data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged \
  LORA_ADAPTER_PATH=none \
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
  SAVE_FREQ=-1 \
  TEST_FREQ=-1 \
  TRAIN_BATCH_SIZE=2 \
  PPO_MINI_BATCH_SIZE=2 \
  PPO_MICRO_BATCH_SIZE_PER_GPU=2 \
  PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  LOG_PROB_USE_DYNAMIC_BSZ=False \
  MODEL_USE_REMOVE_PADDING=False \
  DATA_SEED="${DATA_SEED:-42}" \
  ROLLOUT_SEED="${ROLLOUT_SEED:-42}" \
  FILTER_OVERLONG_PROMPTS=False \
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
  GRPO_REWARD_SCHEME=tree_final \
  GRPO_REWARD_GAMMA=0.9 \
  GRPO_TREE_BRANCH_N=4 \
  GRPO_TREE_BEAM_SIZE=2 \
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
  ENABLE_GPU_MONITOR=True \
  GPU_MONITOR_INTERVAL_SEC=1 \
  uv run --no-sync bash "${SCRIPT_DIR}/run_verl_grpo_qwen25_coder_14b_l40s_4gpu.sh"

  status=$?
  echo "EXIT_CODE ${status} $(date)"
  exit "${status}"
) >"${LOG_FILE}" 2>&1 &

pid=$!
echo "RUN_PID ${pid}"
echo "LOG_FILE ${LOG_FILE}"
if tail --help 2>&1 | grep -q -- "--pid"; then
  tail --pid="${pid}" -f "${LOG_FILE}"
else
  tail -f "${LOG_FILE}"
fi
wait "${pid}"

case "${PREFLIGHT:-0}" in
  1|true|True|TRUE|yes|Yes|YES)
    echo "PREFLIGHT_DONE ${RUN_NAME}"
    exit 0
    ;;
esac

case "${DRY_RUN:-0}" in
  1|true|True|TRUE|yes|Yes|YES)
    echo "DRY_RUN_DONE ${RUN_NAME}"
    exit 0
    ;;
esac

analysis_logs=()
if [[ -f "${BASELINE_LOG}" ]]; then
  analysis_logs+=("${BASELINE_LOG}")
else
  echo "WARN: baseline log not found: ${BASELINE_LOG}"
fi
analysis_logs+=("${LOG_FILE}")

echo
uv run --no-sync python "${SCRIPT_DIR}/analyze_verl_update_actor_benchmark.py" \
  "${analysis_logs[@]}" \
  --warmup-steps "${WARMUP_STEPS}"
