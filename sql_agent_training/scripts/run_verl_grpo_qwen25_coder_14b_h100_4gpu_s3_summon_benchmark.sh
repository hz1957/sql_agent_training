#!/usr/bin/env bash
set -euo pipefail

# Compare full and layered FSDP-to-vLLM LoRA summon using verl's native metrics.
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_summon_benchmark.sh full
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_summon_benchmark.sh layered

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {full|layered}"
  exit 2
fi

SUMMON_MODE="$1"
case "${SUMMON_MODE}" in
  full|layered) ;;
  *)
    echo "Usage: $0 {full|layered}"
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
LOG_DIR="${PROJECT_DIR}/artifacts/logs/verl"
mkdir -p "${LOG_DIR}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-15}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.32}"
if (( TOTAL_TRAINING_STEPS <= WARMUP_STEPS )); then
  echo "ERROR: TOTAL_TRAINING_STEPS must be greater than WARMUP_STEPS."
  exit 2
fi

case "${SUMMON_MODE}" in
  full) ROLLOUT_LAYERED_SUMMON=False ;;
  layered) ROLLOUT_LAYERED_SUMMON=True ;;
esac

RUN_NAME="${RUN_NAME:-verl_grpo_s3_summon_${SUMMON_MODE}_14b_h100_4gpu_${TOTAL_TRAINING_STEPS}step_bs2_n20_ep2_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SLURM_TMPDIR:-/tmp/$USER}/triton_cache}"
export RAY_TMPDIR="${RAY_TMPDIR:-${SLURM_TMPDIR:-/tmp/$USER}/ray}"
# vLLM 0.12.0 sleep-mode CuMemAllocator rejects expandable segments.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
# Load the same compatibility worker in both groups. The only A/B variable is
# actor_rollout_ref.rollout.layered_summon.
export SQL_AGENT_FIX_LAYERED_SUMMON=1
mkdir -p "${TRITON_CACHE_DIR}" "${RAY_TMPDIR}"

case "${STOP_RAY_FIRST:-1}" in
  1|true|True|TRUE|yes|Yes|YES) uv run --no-sync ray stop -f >/dev/null 2>&1 || true ;;
esac

(
  set -o pipefail
  unset ROLLOUT_N
  echo "START ${RUN_NAME} $(date)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "SUMMON_BENCHMARK_CONFIG mode=${SUMMON_MODE} layered_summon=${ROLLOUT_LAYERED_SUMMON} steps=${TOTAL_TRAINING_STEPS} warmup=${WARMUP_STEPS} gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} allocator_conf=default"

  PYTHONUNBUFFERED=1 VLLM_USE_V1="${VLLM_USE_V1:-1}" \
  EXPERIMENT_NAME="${RUN_NAME}" \
  MODEL_PATH=data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged \
  LORA_ADAPTER_PATH=none \
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
  FILTER_OVERLONG_PROMPTS=False \
  SAVE_FREQ=-1 \
  TEST_FREQ=-1 \
  TRAIN_BATCH_SIZE=2 \
  PPO_MINI_BATCH_SIZE=2 \
  PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
  ROLLOUT_TP=4 \
  ROLLOUT_PP=1 \
  ROLLOUT_TEMPERATURE=1.0 \
  ROLLOUT_TOP_P=1.0 \
  ROLLOUT_TOP_K=-1 \
  ROLLOUT_DO_SAMPLE=True \
  ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096 \
  ROLLOUT_MAX_NUM_SEQS=4 \
  ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON}" \
  MAX_TURNS=3 \
  MAX_PROMPT_LENGTH=2048 \
  MAX_RESPONSE_LENGTH=2048 \
  GRPO_REWARD_SCHEME=tree_final \
  GRPO_REWARD_GAMMA=0.9 \
  GRPO_TREE_BRANCH_N=4 \
  GRPO_TREE_BEAM_SIZE=2 \
  GRPO_TREE_BEAM_TAU=1.0 \
  GRPO_TREE_BEAM_EPSILON_RANDOM=0.1 \
  GRPO_TREE_PRUNE_ON_GOLD_REWARD=True \
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

echo
uv run --no-sync python "${SCRIPT_DIR}/analyze_verl_summon_benchmark.py" \
  "${LOG_FILE}" \
  --warmup-steps "${WARMUP_STEPS}"
