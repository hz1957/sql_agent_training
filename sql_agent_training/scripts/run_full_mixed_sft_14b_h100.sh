#!/usr/bin/env bash
set -euo pipefail

# Build and train one 7,000-record Mixed-SFT dataset on four H100/H200 GPUs.
# Launch from the outer workspace root.

usage() {
  cat >&2 <<'EOF'
Usage:
  bash sql_agent_training/scripts/run_full_mixed_sft_14b_h100.sh [prepare|train|eval|all|list]

Default mode:
  train

Dataset:
  4,667 gold + 1,658 direct trajectory + 675 rewrite trajectory = 7,000 records

Recommended sequence:
  bash sql_agent_training/scripts/run_full_mixed_sft_14b_h100.sh prepare
  CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 \
    bash sql_agent_training/scripts/run_full_mixed_sft_14b_h100.sh train

Environment overrides:
  NUM_GPUS, CUDA_VISIBLE_DEVICES, DEEPSPEED_INCLUDE, CHECKPOINT_STEPS,
  EVAL_SAMPLE_SIZE, EVAL_SAMPLE_SEED, EVAL_CUDA_VISIBLE_DEVICES,
  LOG_ROOT, RUNTIME_CACHE_ROOT, UV_PROJECT_ENVIRONMENT
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODE="${1:-train}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  for candidate in \
    "${WORKSPACE_DIR}/.venv-sft" \
    "${PROJECT_DIR}/.venv-sft" \
    "${WORKSPACE_DIR}/.venv" \
    "${PROJECT_DIR}/.venv"; do
    if [[ -x "${candidate}/bin/python" ]]; then
      export UV_PROJECT_ENVIRONMENT="${candidate}"
      break
    fi
  done
fi

NUM_GPUS="${NUM_GPUS:-4}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-438}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-500}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-0}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/sft_full_mixed}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}}/sql_agent_training_sft_full_mixed}"

TRAJECTORY_DIR="artifacts/sft_trajectory/qwen25_coder_14b_sft_merged_q1000_n4_target1600_seed43_t09"
TRAJECTORY_JSONL="${TRAJECTORY_DIR}/trajectory_sft.jsonl"
MIXED_JSONL="${TRAJECTORY_DIR}/mixed_sft_full_gold4667_direct1658_rewrite675_seed42.jsonl"
MIXED_SUMMARY="${TRAJECTORY_DIR}/mixed_sft_full_gold4667_direct1658_rewrite675_seed42_summary.json"
SFT_CONFIG="configs/sft.qwen25_coder_14b_from_base_full_mixed_gold4667_direct1658_rewrite675.h100_zero2.lr5e5_r32.yaml"
EVAL_CONFIG="configs/agent_eval.qwen25_coder_14b_base.yaml"
CHECKPOINT_ROOT="artifacts/checkpoints/sft_14b_from_base_full_mixed_gold4667_d1658_r675_lr5e5_r32"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-${RUNTIME_CACHE_ROOT}/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUNTIME_CACHE_ROOT}/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${RUNTIME_CACHE_ROOT}/cuda}"

PYTHON_CMD=(uv run --no-sync python)
DEEPSPEED_CMD=(uv run --no-sync deepspeed)
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  PYTHON_CMD=("${UV_PROJECT_ENVIRONMENT}/bin/python")
fi
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && -x "${UV_PROJECT_ENVIRONMENT}/bin/deepspeed" ]]; then
  DEEPSPEED_CMD=("${UV_PROJECT_ENVIRONMENT}/bin/deepspeed")
fi

mkdir -p \
  "${LOG_ROOT}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${CUDA_CACHE_PATH}"

list_experiment() {
  echo "MIX=gold:4667 direct:1658 rewrite:675 total:7000"
  echo "BASE_MODEL=data/models/Qwen2.5-Coder-14B-Instruct"
  echo "TRAIN=global_batch:16 lr:5e-5 lora_rank:32 steps:438 epochs:1"
  echo "TRAJECTORY_JSONL=${TRAJECTORY_JSONL}"
  echo "MIXED_JSONL=${MIXED_JSONL}"
  echo "SFT_CONFIG=${SFT_CONFIG}"
  echo "CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
}

validate_trajectory_pool() {
  if [[ ! -f "${TRAJECTORY_JSONL}" ]]; then
    echo "ERROR: missing trajectory pool: ${TRAJECTORY_JSONL}" >&2
    echo "Run scripts/run_regenerate_sft_trajectories_for_ratio.sh first." >&2
    exit 1
  fi
}

run_prepare() {
  validate_trajectory_pool
  local log_path="${LOG_ROOT}/prepare_$(date +%Y%m%d_%H%M%S).log"
  echo "START_FULL_MIXED_SFT_PREPARE $(date)"
  PYTHONUNBUFFERED=1 \
    "${PYTHON_CMD[@]}" scripts/prepare_mixed_sft.py \
      --trajectory-jsonl "${TRAJECTORY_JSONL}" \
      --output-jsonl "${MIXED_JSONL}" \
      --summary-json "${MIXED_SUMMARY}" \
      --gold-count 4667 \
      --direct-count 1658 \
      --rewrite-count 675 \
      --seed 42 \
    2>&1 | tee "${log_path}"
  echo "FINISH_FULL_MIXED_SFT_PREPARE summary=${MIXED_SUMMARY} log=${log_path} $(date)"
}

run_train() {
  if [[ ! -f "${MIXED_JSONL}" ]]; then
    echo "ERROR: missing prepared Mixed-SFT data: ${MIXED_JSONL}" >&2
    echo "Run '$0 prepare' first." >&2
    exit 1
  fi

  local launcher_args=()
  if [[ -n "${DEEPSPEED_INCLUDE:-}" ]]; then
    launcher_args+=(--include "${DEEPSPEED_INCLUDE}")
  elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    launcher_args+=(--num_gpus "${NUM_GPUS}")
  fi

  local log_path="${LOG_ROOT}/train_$(date +%Y%m%d_%H%M%S).log"
  echo "START_FULL_MIXED_SFT_TRAIN config=${SFT_CONFIG} num_gpus=${NUM_GPUS} $(date)"
  PYTHONUNBUFFERED=1 \
    "${DEEPSPEED_CMD[@]}" "${launcher_args[@]}" \
      --module sql_agent_training.train.sft \
      --config "${SFT_CONFIG}" \
    2>&1 | tee "${log_path}"
  echo "FINISH_FULL_MIXED_SFT_TRAIN checkpoint_root=${CHECKPOINT_ROOT} log=${log_path} $(date)"
}

latest_run_dir() {
  [[ -d "${CHECKPOINT_ROOT}" ]] || return 1
  find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR == 1 { $1=""; sub(/^ /, ""); print; exit }'
}

run_eval() {
  local run_dir
  if ! run_dir="$(latest_run_dir)"; then
    echo "ERROR: no run found under ${CHECKPOINT_ROOT}" >&2
    exit 1
  fi

  read -r -a steps <<< "${CHECKPOINT_STEPS}"
  local step checkpoint output_dir log_path
  for step in "${steps[@]}"; do
    checkpoint="${run_dir}/checkpoint-${step}"
    if [[ ! -d "${checkpoint}" ]]; then
      echo "SKIP missing checkpoint: ${checkpoint}"
      continue
    fi
    output_dir="${checkpoint}/agent_eval_validation_${EVAL_SAMPLE_SIZE}_seed${EVAL_SAMPLE_SEED}"
    log_path="${LOG_ROOT}/checkpoint-${step}_agent_eval_$(date +%Y%m%d_%H%M%S).log"
    CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" PYTHONUNBUFFERED=1 \
      "${PYTHON_CMD[@]}" -m sql_agent_training.train.agent_eval \
        --config "${EVAL_CONFIG}" \
        --checkpoint "${checkpoint}" \
        --split validation \
        --sample-size "${EVAL_SAMPLE_SIZE}" \
        --sample-seed "${EVAL_SAMPLE_SEED}" \
        --output-dir "${output_dir}" \
      2>&1 | tee "${log_path}"
  done
}

case "${MODE}" in
  prepare) run_prepare ;;
  train) run_train ;;
  eval) run_eval ;;
  all) run_prepare; run_train; run_eval ;;
  list) list_experiment ;;
  *) usage; exit 2 ;;
esac
