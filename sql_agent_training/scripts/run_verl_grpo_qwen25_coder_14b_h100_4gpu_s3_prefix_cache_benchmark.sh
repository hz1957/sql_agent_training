#!/usr/bin/env bash
set -euo pipefail

# Measure vLLM prefix caching with an otherwise identical S3 tree workload.
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_prefix_cache_benchmark.sh off
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_prefix_cache_benchmark.sh on

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {off|on}"
  exit 2
fi

PREFIX_CACHE_MODE="$1"
case "${PREFIX_CACHE_MODE}" in
  off) ROLLOUT_ENABLE_PREFIX_CACHING=False ;;
  on) ROLLOUT_ENABLE_PREFIX_CACHING=True ;;
  *)
    echo "Usage: $0 {off|on}"
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
LOG_DIR="${PROJECT_DIR}/artifacts/logs/verl"
mkdir -p "${LOG_DIR}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-12}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.32}"
BATCH_INVARIANT="${BATCH_INVARIANT:-0}"
if (( TOTAL_TRAINING_STEPS <= WARMUP_STEPS )); then
  echo "ERROR: TOTAL_TRAINING_STEPS must be greater than WARMUP_STEPS."
  exit 2
fi
case "${BATCH_INVARIANT}" in
  1|true|True|TRUE|yes|Yes|YES) BATCH_INVARIANT_ENABLED=True ;;
  0|false|False|FALSE|no|No|NO) BATCH_INVARIANT_ENABLED=False ;;
  *)
    echo "ERROR: BATCH_INVARIANT must be 0/1 or true/false."
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-verl_grpo_s3_prefix_cache_${PREFIX_CACHE_MODE}_14b_h100_4gpu_${TOTAL_TRAINING_STEPS}step_bs2_n20_ep2_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
CACHE_SHORT_TAG="${CACHE_SHORT_TAG:-$(date +%H%M%S)}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_${USER:-u}_${PREFIX_CACHE_MODE}_${CACHE_SHORT_TAG}}"
export UV_CACHE="${UV_CACHE:-${UV_CACHE_DIR}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PROJECT_DIR}/artifacts/tmp/${RUN_NAME}/triton_cache}"
# Ray appends ray/session_*/sockets/plasma_store under RAY_TMPDIR. The cluster
# workspace path plus a descriptive run name can exceed the 107-byte Unix socket
# limit, so keep this one intentionally short.
RAY_SHORT_TAG="${RAY_SHORT_TAG:-$(date +%H%M%S)}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/vr_${USER:-u}_${PREFIX_CACHE_MODE}_${RAY_SHORT_TAG}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_DIR}/artifacts/tmp/${RUN_NAME}/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${PROJECT_DIR}/artifacts/tmp/${RUN_NAME}/xdg_config}"
# vLLM batch invariance is a beta path; default to off and let exact workload
# fingerprints decide whether the OFF/ON timing comparison is valid.
if [[ "${BATCH_INVARIANT_ENABLED}" == "True" ]]; then
  export VLLM_BATCH_INVARIANT=1
else
  unset VLLM_BATCH_INVARIANT
fi
# Enable native vLLM throughput/cache statistics without writing usage data under $HOME.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_LOG_STATS_INTERVAL=1
export NO_COLOR=1
# vLLM 0.12.0 sleep-mode CuMemAllocator rejects expandable segments.
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
# Keep the proven 80 GB H100 weight-sync path identical in both groups.
export SQL_AGENT_FIX_LAYERED_SUMMON=1
# verl's async vLLM path can leave the engine default enabled even when
# RolloutConfig says False. Force the final vLLM config to match this benchmark.
export SQL_AGENT_FIX_PREFIX_CACHE_CONFIG=1
mkdir -p "${UV_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${RAY_TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}"

case "${STOP_RAY_FIRST:-1}" in
  1|true|True|TRUE|yes|Yes|YES) uv run --no-sync ray stop -f >/dev/null 2>&1 || true ;;
esac

(
  set -o pipefail
  unset ROLLOUT_N
  echo "START ${RUN_NAME} $(date)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "PREFIX_CACHE_BENCHMARK_CONFIG mode=${PREFIX_CACHE_MODE} enable_prefix_caching=${ROLLOUT_ENABLE_PREFIX_CACHING} prefix_cache_patch=True disable_log_stats=False steps=${TOTAL_TRAINING_STEPS} warmup=${WARMUP_STEPS} data_seed=42 rollout_seed=42 stable_request_seeds=True batch_invariant=${BATCH_INVARIANT_ENABLED} frozen_actor=True workload_fingerprint=True batch_size=2 rollout_n=20 branch_n=4 beam_size=2 max_turns=3 temperature=1.0 top_p=1.0 top_k=-1 gamma=0.9 ppo_epochs=2 kl_coef=0.01 actor_lr=0 layered_summon=True"

  PYTHONUNBUFFERED=1 VLLM_USE_V1="${VLLM_USE_V1:-1}" \
  EXPERIMENT_NAME="${RUN_NAME}" \
  MODEL_PATH=data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged \
  LORA_ADAPTER_PATH=none \
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
  FILTER_OVERLONG_PROMPTS=False \
  SAVE_FREQ=-1 \
  TEST_FREQ=-1 \
  DATA_SEED=42 \
  ROLLOUT_SEED=42 \
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
  ROLLOUT_LAYERED_SUMMON=True \
  ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING}" \
  ROLLOUT_DISABLE_LOG_STATS=False \
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
  GRPO_TREE_STABLE_REQUEST_SEEDS=True \
  GRPO_TREE_WORKLOAD_FINGERPRINT=True \
  GRPO_ADV_ESTIMATOR=grpo_tree \
  USE_KL_IN_REWARD=False \
  REF_PARAM_OFFLOAD=True \
  USE_KL_LOSS=True \
  KL_LOSS_COEF=0.01 \
  PPO_EPOCHS=2 \
  ACTOR_LR=0 \
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
uv run --no-sync python "${SCRIPT_DIR}/analyze_verl_prefix_cache_benchmark.py" \
  "${LOG_FILE}" \
  --warmup-steps "${WARMUP_STEPS}"
