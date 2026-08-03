#!/usr/bin/env bash
set -euo pipefail

# Regenerate the canonical SFT trajectory pool for data-ratio experiments.
#
# This intentionally overwrites the existing trajectory pool used by the Mixed-SFT
# preparation scripts. The target is larger than the current 1,600 records so that
# the next ratio experiment can include up to 2,400 trajectory records with buffer.

usage() {
  cat >&2 <<'EOF'
Usage:
  bash sql_agent_training/scripts/run_regenerate_sft_trajectories_for_ratio.sh

Default behavior:
  Overwrite the canonical trajectory directory:
    artifacts/sft_trajectory/qwen25_coder_14b_sft_merged_q1000_n4_target1600_seed43_t09

Default generation:
  question_count          = 1500
  rollouts_per_question  = 4
  target_correct         = 3000
  temperature            = 0.9
  seed                   = 44
  max_turns              = 3

Recommended 4xH100 launch:
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  VLLM_TENSOR_PARALLEL_SIZE=4 \
  bash sql_agent_training/scripts/run_regenerate_sft_trajectories_for_ratio.sh

Environment overrides:
  OUTPUT_DIR, QUESTION_COUNT, ROLLOUTS_PER_QUESTION, TARGET_CORRECT, SEED,
  TEMPERATURE, CUDA_VISIBLE_DEVICES, VLLM_TENSOR_PARALLEL_SIZE, VLLM_PORT,
  VLLM_BIN, VLLM_PROJECT_ENVIRONMENT, RUNTIME_CACHE_ROOT, UV_PROJECT_ENVIRONMENT
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CANONICAL_OUTPUT_DIR="artifacts/sft_trajectory/qwen25_coder_14b_sft_merged_q1000_n4_target1600_seed43_t09"
OUTPUT_DIR="${OUTPUT_DIR:-${CANONICAL_OUTPUT_DIR}}"
QUESTION_COUNT="${QUESTION_COUNT:-1500}"
ROLLOUTS_PER_QUESTION="${ROLLOUTS_PER_QUESTION:-4}"
TARGET_CORRECT="${TARGET_CORRECT:-3000}"
SEED="${SEED:-44}"
TEMPERATURE="${TEMPERATURE:-0.9}"
MAX_TURNS="${MAX_TURNS:-3}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LOG_DIR="${LOG_DIR:-artifacts/logs/sft_trajectory}"
GENERATOR_LOG="${GENERATOR_LOG:-${LOG_DIR}/regenerate_ratio_q${QUESTION_COUNT}_n${ROLLOUTS_PER_QUESTION}_target${TARGET_CORRECT}_seed${SEED}_t09.log}"
VLLM_LOG="${VLLM_LOG:-${LOG_DIR}/vllm_regenerate_ratio.log}"

case "${OUTPUT_DIR}" in
  artifacts/sft_trajectory/*) ;;
  *)
    echo "ERROR: refusing to overwrite outside artifacts/sft_trajectory: ${OUTPUT_DIR}" >&2
    exit 2
    ;;
esac

echo "REGENERATE_TRAJECTORIES output_dir=${OUTPUT_DIR}"
echo "REGENERATE_TRAJECTORIES questions=${QUESTION_COUNT} rollouts=${ROLLOUTS_PER_QUESTION} target=${TARGET_CORRECT}"
echo "REGENERATE_TRAJECTORIES seed=${SEED} temperature=${TEMPERATURE} max_turns=${MAX_TURNS}"
echo "REGENERATE_TRAJECTORIES cuda=${CUDA_VISIBLE_DEVICES} tp=${VLLM_TENSOR_PARALLEL_SIZE}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Remove stale trajectory and mixed-SFT files so downstream scripts cannot
# accidentally consume a mixture prepared from the previous 1,600-record pool.
rm -f \
  "${OUTPUT_DIR}/candidate_trajectories.jsonl" \
  "${OUTPUT_DIR}/verified_trajectories.jsonl" \
  "${OUTPUT_DIR}/trajectory_sft.jsonl" \
  "${OUTPUT_DIR}/summary.json" \
  "${OUTPUT_DIR}"/mixed_sft_*.jsonl \
  "${OUTPUT_DIR}"/mixed_sft_*_summary.json

export \
  OUTPUT_DIR \
  QUESTION_COUNT \
  ROLLOUTS_PER_QUESTION \
  TARGET_CORRECT \
  SEED \
  TEMPERATURE \
  MAX_TURNS \
  CUDA_VISIBLE_DEVICES \
  VLLM_TENSOR_PARALLEL_SIZE \
  LOG_DIR \
  GENERATOR_LOG \
  VLLM_LOG

bash scripts/run_generate_sft_trajectories.sh

echo "Regenerated trajectory pool:"
echo "  ${OUTPUT_DIR}/summary.json"
echo "  ${OUTPUT_DIR}/trajectory_sft.jsonl"
