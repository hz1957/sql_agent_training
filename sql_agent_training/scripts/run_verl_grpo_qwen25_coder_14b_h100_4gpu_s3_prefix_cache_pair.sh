#!/usr/bin/env bash
set -euo pipefail

# Run a controlled prefix-cache OFF/ON pair, then compare the two logs.
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_prefix_cache_pair.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-12}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
PAIR_ID="${PAIR_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${PROJECT_DIR}/artifacts/logs/verl"
mkdir -p "${LOG_DIR}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_${USER:-u}_pcpair_${PAIR_ID}}"
export UV_CACHE="${UV_CACHE:-${UV_CACHE_DIR}}"
mkdir -p "${UV_CACHE_DIR}"

OFF_NAME="verl_grpo_s3_prefix_cache_off_14b_h100_4gpu_${TOTAL_TRAINING_STEPS}step_bs2_n20_ep2_${PAIR_ID}"
ON_NAME="verl_grpo_s3_prefix_cache_on_14b_h100_4gpu_${TOTAL_TRAINING_STEPS}step_bs2_n20_ep2_${PAIR_ID}"
OFF_LOG="${LOG_DIR}/${OFF_NAME}.log"
ON_LOG="${LOG_DIR}/${ON_NAME}.log"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
WARMUP_STEPS="${WARMUP_STEPS}" \
RUN_NAME="${OFF_NAME}" \
LOG_FILE="${OFF_LOG}" \
bash "${SCRIPT_DIR}/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_prefix_cache_benchmark.sh" off

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
WARMUP_STEPS="${WARMUP_STEPS}" \
RUN_NAME="${ON_NAME}" \
LOG_FILE="${ON_LOG}" \
bash "${SCRIPT_DIR}/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_prefix_cache_benchmark.sh" on

echo
uv run --no-sync python "${SCRIPT_DIR}/analyze_verl_prefix_cache_benchmark.py" \
  "${OFF_LOG}" \
  "${ON_LOG}" \
  --warmup-steps "${WARMUP_STEPS}"
