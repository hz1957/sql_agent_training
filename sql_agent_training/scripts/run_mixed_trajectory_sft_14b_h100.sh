#!/usr/bin/env bash
set -euo pipefail

# Build the 3200/1137/463 mixture, train one LoRA adapter on 4 H100s, and run agent eval.
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_mixed_trajectory_sft_14b_h100.sh all

usage() {
  cat >&2 <<'EOF'
Usage:
  bash sql_agent_training/scripts/run_mixed_trajectory_sft_14b_h100.sh [all|prepare|train|eval|list]

Modes:
  all       Prepare data, train, then evaluate checkpoints 100/200/300.
  prepare   Build the fixed 3200 gold + 1137 direct + 463 rewrite JSONL.
  train     Train Mixed-SFT only; requires the prepared JSONL.
  eval      Agent-evaluate checkpoints from the latest Mixed-SFT run.
  list      Print the resolved experiment settings without running.

Environment overrides:
  NUM_GPUS                   DeepSpeed GPU count. Default: 4
  CHECKPOINT_STEPS           Space-separated eval steps. Default: 100 200 300
  EVAL_SAMPLE_SIZE           Spider validation sample size. Default: 500
  EVAL_SAMPLE_SEED           Spider validation sample seed. Default: 0
  EVAL_CUDA_VISIBLE_DEVICES  GPU used for sequential agent eval. Default: 0
  LOG_ROOT                   Log directory. Default: artifacts/logs/sft_mixed_trajectory
  RUNTIME_CACHE_ROOT         Runtime cache root. Default: node-local temporary storage
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

TRAJECTORY_DIR="artifacts/sft_trajectory/qwen25_coder_14b_sft_merged_q1000_n4_target1600_seed43_t09"
TRAJECTORY_JSONL="${TRAJECTORY_DIR}/trajectory_sft.jsonl"
MIXED_JSONL="${TRAJECTORY_DIR}/mixed_sft_gold3200_direct1137_rewrite463_seed42.jsonl"
MIXED_SUMMARY="${TRAJECTORY_DIR}/mixed_sft_gold3200_direct1137_rewrite463_seed42_summary.json"
SFT_CONFIG="configs/sft.qwen25_coder_14b_from_base_mixed_gold3200_direct1137_rewrite463.h100_zero2.yaml"
EVAL_CONFIG="configs/agent_eval.qwen25_coder_14b_base.yaml"
CHECKPOINT_ROOT="artifacts/checkpoints/sft_14b_from_base_mixed_gold3200_direct1137_rewrite463"

NUM_GPUS="${NUM_GPUS:-4}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-100 200 300}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-500}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-0}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/sft_mixed_trajectory}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}}/sql_agent_training_mixed_sft}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-${RUNTIME_CACHE_ROOT}/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUNTIME_CACHE_ROOT}/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${RUNTIME_CACHE_ROOT}/cuda}"

mkdir -p \
  "${LOG_ROOT}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${CUDA_CACHE_PATH}"

latest_run_dir() {
  if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    return 1
  fi
  find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR == 1 { $1=""; sub(/^ /, ""); print; exit }'
}

list_experiment() {
  echo "SFT_CONFIG=${SFT_CONFIG}"
  echo "EVAL_CONFIG=${EVAL_CONFIG}"
  echo "TRAJECTORY_JSONL=${TRAJECTORY_JSONL}"
  echo "MIXED_JSONL=${MIXED_JSONL}"
  echo "MIX=gold:3200 direct:1137 rewrite:463 total:4800"
  echo "BASE_MODEL=data/models/Qwen2.5-Coder-14B-Instruct"
  echo "TRAIN=4xH100 micro_batch:1 accumulation:4 global_batch:16 lr:5e-5 steps:300 epochs:1"
  echo "CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
  echo "CHECKPOINT_STEPS=${CHECKPOINT_STEPS}"
  echo "AGENT_EVAL=sample_size:${EVAL_SAMPLE_SIZE} seed:${EVAL_SAMPLE_SEED} max_turns:3 temperature:0"
}

run_prepare() {
  local log_path="${LOG_ROOT}/prepare_$(date +%Y%m%d_%H%M%S).log"
  echo "START_MIXED_SFT_PREPARE input=${TRAJECTORY_JSONL} output=${MIXED_JSONL} $(date)"
  PYTHONUNBUFFERED=1 \
    uv run --no-sync python scripts/prepare_mixed_sft.py \
      --trajectory-jsonl "${TRAJECTORY_JSONL}" \
      --output-jsonl "${MIXED_JSONL}" \
      --summary-json "${MIXED_SUMMARY}" \
      --gold-count 3200 \
      --direct-count 1137 \
      --rewrite-count 463 \
      --seed 42 \
    2>&1 | tee "${log_path}"
  echo "FINISH_MIXED_SFT_PREPARE summary=${MIXED_SUMMARY} log=${log_path} $(date)"
}

run_train() {
  if [[ ! -f "${MIXED_JSONL}" ]]; then
    echo "ERROR: missing Mixed-SFT data: ${MIXED_JSONL}" >&2
    echo "Run '$0 prepare' first." >&2
    exit 1
  fi

  local log_path="${LOG_ROOT}/train_$(date +%Y%m%d_%H%M%S).log"
  echo "START_MIXED_SFT_TRAIN config=${SFT_CONFIG} num_gpus=${NUM_GPUS} $(date)"
  PYTHONUNBUFFERED=1 \
    uv run --no-sync deepspeed --num_gpus "${NUM_GPUS}" \
      --module sql_agent_training.train.sft \
      --config "${SFT_CONFIG}" \
    2>&1 | tee "${log_path}"
  echo "FINISH_MIXED_SFT_TRAIN checkpoint_root=${CHECKPOINT_ROOT} log=${log_path} $(date)"
}

run_eval() {
  local run_dir
  if ! run_dir="$(latest_run_dir)"; then
    echo "ERROR: no Mixed-SFT run found under ${CHECKPOINT_ROOT}" >&2
    exit 1
  fi

  echo "START_MIXED_SFT_AGENT_EVAL run_dir=${run_dir} $(date)"
  read -r -a step_values <<< "${CHECKPOINT_STEPS}"
  for step in "${step_values[@]}"; do
    local checkpoint output_dir log_path
    checkpoint="${run_dir}/checkpoint-${step}"
    if [[ ! -d "${checkpoint}" ]]; then
      echo "SKIP missing checkpoint: ${checkpoint}"
      continue
    fi

    output_dir="${checkpoint}/agent_eval_validation_${EVAL_SAMPLE_SIZE}_seed${EVAL_SAMPLE_SEED}"
    log_path="${LOG_ROOT}/checkpoint-${step}_agent_eval_$(date +%Y%m%d_%H%M%S).log"
    echo "EVAL checkpoint=${checkpoint} output=${output_dir}"
    CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
      PYTHONUNBUFFERED=1 \
      uv run --no-sync python -m sql_agent_training.train.agent_eval \
        --config "${EVAL_CONFIG}" \
        --checkpoint "${checkpoint}" \
        --split validation \
        --sample-size "${EVAL_SAMPLE_SIZE}" \
        --sample-seed "${EVAL_SAMPLE_SEED}" \
        --output-dir "${output_dir}" \
      2>&1 | tee "${log_path}"
  done
  echo "FINISH_MIXED_SFT_AGENT_EVAL run_dir=${run_dir} $(date)"
}

case "${MODE}" in
  list)
    list_experiment
    ;;
  prepare)
    run_prepare
    ;;
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
  all)
    run_prepare
    run_train
    run_eval
    ;;
  *)
    usage
    exit 2
    ;;
esac
