#!/usr/bin/env bash
set -euo pipefail

# 4x H100 S3 rollout concurrency benchmark.
# Default runs a pair:
#   baseline: current best config, dynamic_bsz=True + remove_padding=True, seqs=4, tokens=4096
#   seq8_tok8192: only increase vLLM rollout concurrency capacity to seqs=8, tokens=8192
#
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_rollout_concurrency_benchmark.sh
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_rollout_concurrency_benchmark.sh baseline
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_rollout_concurrency_benchmark.sh seq8_tok8192

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/artifacts/logs/verl"
mkdir -p "${LOG_DIR}"

VARIANT="${1:-pair}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"

if (( TOTAL_TRAINING_STEPS <= WARMUP_STEPS )); then
  echo "ERROR: TOTAL_TRAINING_STEPS must be greater than WARMUP_STEPS."
  exit 2
fi

case "${VARIANT}" in
  baseline|seq8_tok8192|pair) ;;
  *)
    echo "ERROR: expected variant baseline, seq8_tok8192, or pair; got: ${VARIANT}"
    exit 2
    ;;
esac

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SLURM_TMPDIR:-/tmp/$USER}/triton_cache}"
export RAY_TMPDIR="${RAY_TMPDIR:-${SLURM_TMPDIR:-/tmp/$USER}/ray}"
export SQL_AGENT_FIX_LAYERED_SUMMON=1
unset PYTORCH_CUDA_ALLOC_CONF
unset PYTORCH_ALLOC_CONF
mkdir -p "${TRITON_CACHE_DIR}" "${RAY_TMPDIR}"

is_enabled() {
  case "${1:-0}" in
    1|true|True|TRUE|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

run_one() {
  local variant="$1"
  local rollout_max_num_seqs
  local rollout_max_num_batched_tokens
  local variant_label

  case "${variant}" in
    baseline)
      rollout_max_num_seqs=4
      rollout_max_num_batched_tokens=4096
      variant_label="baseline_optimal_dynamic_bsz_remove_padding"
      ;;
    seq8_tok8192)
      rollout_max_num_seqs=8
      rollout_max_num_batched_tokens=8192
      variant_label="seq8_tok8192_dynamic_bsz_remove_padding"
      ;;
    *)
      echo "ERROR: unknown variant: ${variant}"
      return 2
      ;;
  esac

  local run_name
  local log_file
  run_name="${RUN_NAME_PREFIX:-verl_grpo_s3_rollout_concurrency}_${variant}_14b_h100_4gpu_${TOTAL_TRAINING_STEPS}step_bs2_n20_ep2_seqs${rollout_max_num_seqs}_tok${rollout_max_num_batched_tokens}_$(date +%Y%m%d_%H%M%S)"
  log_file="${LOG_DIR}/${run_name}.log"

  case "${STOP_RAY_FIRST:-1}" in
    1|true|True|TRUE|yes|Yes|YES) uv run --no-sync ray stop -f >/dev/null 2>&1 || true ;;
  esac

  (
    set -o pipefail
    unset ROLLOUT_N
    echo "START ${run_name} $(date)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "ROLLOUT_CONCURRENCY_BENCHMARK_CONFIG variant=${variant_label} steps=${TOTAL_TRAINING_STEPS} warmup=${WARMUP_STEPS} train_batch_size=2 ppo_mini_batch_size=2 ppo_micro_batch_size_per_gpu=1 dynamic_bsz=True remove_padding=True ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} rollout_max_num_seqs=${rollout_max_num_seqs} rollout_max_num_batched_tokens=${rollout_max_num_batched_tokens} branch_n=4 beam_size=2 rollout_n=20 ppo_epochs=2 kl_coef=0.01 layered_summon=True"

    PYTHONUNBUFFERED=1 VLLM_USE_V1="${VLLM_USE_V1:-1}" \
    EXPERIMENT_NAME="${run_name}" \
    MODEL_PATH=data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged \
    LORA_ADAPTER_PATH=none \
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
    SAVE_FREQ=-1 \
    TEST_FREQ=-1 \
    TRAIN_BATCH_SIZE=2 \
    PPO_MINI_BATCH_SIZE=2 \
    PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
    PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    LOG_PROB_USE_DYNAMIC_BSZ=True \
    MODEL_USE_REMOVE_PADDING=True \
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
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="${rollout_max_num_batched_tokens}" \
    ROLLOUT_MAX_NUM_SEQS="${rollout_max_num_seqs}" \
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
  ) >"${log_file}" 2>&1 &

  local pid=$!
  echo "RUN_PID ${pid}"
  echo "LOG_FILE ${log_file}"
  if tail --help 2>&1 | grep -q -- "--pid"; then
    tail --pid="${pid}" -f "${log_file}"
  else
    tail -f "${log_file}"
  fi
  wait "${pid}"
  RUN_LOGS+=("${log_file}")
}

RUN_LOGS=()

case "${VARIANT}" in
  pair)
    run_one baseline
    run_one seq8_tok8192
    ;;
  *)
    run_one "${VARIANT}"
    ;;
esac

if is_enabled "${PREFLIGHT:-0}"; then
  echo "PREFLIGHT_DONE ${VARIANT}"
  exit 0
fi

if is_enabled "${DRY_RUN:-0}"; then
  echo "DRY_RUN_DONE ${VARIANT}"
  exit 0
fi

echo
uv run --no-sync python "${SCRIPT_DIR}/analyze_verl_update_actor_benchmark.py" \
  "${RUN_LOGS[@]}" \
  --warmup-steps "${WARMUP_STEPS}"
