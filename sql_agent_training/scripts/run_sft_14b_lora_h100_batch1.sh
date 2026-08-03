#!/usr/bin/env bash
set -euo pipefail

# Run the first SFT quality sweep for Qwen2.5-Coder-14B LoRA on H100 ZeRO-2.
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_sft_14b_lora_h100_batch1.sh all

usage() {
  cat >&2 <<'EOF'
Usage:
  bash sql_agent_training/scripts/run_sft_14b_lora_h100_batch1.sh [all|train|eval|list]

Default mode:
  all

Environment overrides:
  NUM_GPUS           DeepSpeed GPU count. Default: 4
  SFT_CONFIGS        Space-separated config list. Default: first batch configs
  CHECKPOINT_STEPS   Space-separated checkpoint steps for eval. Default: 100 200 300 400 500 600
  EVAL_SAMPLE_SIZE   Spider validation sample size for execution eval. Default: 500
  EVAL_SAMPLE_SEED   Spider validation sample seed. Default: 0
  EVAL_SPLIT         Eval split. Default: validation
  LOG_ROOT           Log directory. Default: artifacts/logs/sft_batch1
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

DEFAULT_CONFIGS=(
  configs/sft.qwen25_coder_14b_lora.h100_zero2.batch1_lr2e5_r32.yaml
  configs/sft.qwen25_coder_14b_lora.h100_zero2.batch1_lr3e5_r32.yaml
  configs/sft.qwen25_coder_14b_lora.h100_zero2.batch1_lr2e5_r64.yaml
  configs/sft.qwen25_coder_14b_lora.h100_zero2.batch1_lr3e5_r64.yaml
)

if [[ -n "${SFT_CONFIGS:-}" ]]; then
  read -r -a CONFIGS <<< "${SFT_CONFIGS}"
else
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

NUM_GPUS="${NUM_GPUS:-4}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-100 200 300 400 500 600}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-500}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-0}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/sft_batch1}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SLURM_TMPDIR:-/tmp/$USER}/triton_cache}"

export UV_LINK_MODE TRITON_CACHE_DIR
mkdir -p "${LOG_ROOT}" "${TRITON_CACHE_DIR}"

checkpoint_root_from_config() {
  awk -F': *' '$1 ~ /^[[:space:]]*checkpoint_dir$/ { print $2; exit }' "$1"
}

latest_run_dir() {
  local root="$1"
  if [[ ! -d "${root}" ]]; then
    return 1
  fi
  find "${root}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | awk 'NR == 1 { $1=""; sub(/^ /, ""); print; exit }'
}

list_configs() {
  for config in "${CONFIGS[@]}"; do
    local root
    root="$(checkpoint_root_from_config "${config}")"
    printf '%s -> %s\n' "${config}" "${root}"
  done
}

run_train() {
  local config="$1"
  local root slug log_path
  root="$(checkpoint_root_from_config "${config}")"
  slug="$(basename "${root}")"
  log_path="${LOG_ROOT}/${slug}_train_$(date +%Y%m%d_%H%M%S).log"

  echo "START_SFT_TRAIN config=${config} checkpoint_root=${root} num_gpus=${NUM_GPUS} $(date)"
  PYTHONUNBUFFERED=1 \
    uv run --no-sync deepspeed --num_gpus "${NUM_GPUS}" --module sql_agent_training.train.sft --config "${config}" \
    2>&1 | tee "${log_path}"
  echo "FINISH_SFT_TRAIN config=${config} log=${log_path} $(date)"
}

run_eval() {
  local config="$1"
  local root slug run_dir
  root="$(checkpoint_root_from_config "${config}")"
  slug="$(basename "${root}")"
  if ! run_dir="$(latest_run_dir "${root}")"; then
    echo "WARN: no run directory found under ${root}; skipping eval for ${config}" >&2
    return 0
  fi

  echo "START_SFT_EVAL config=${config} run_dir=${run_dir} sample_size=${EVAL_SAMPLE_SIZE} seed=${EVAL_SAMPLE_SEED} $(date)"
  read -r -a STEP_VALUES <<< "${CHECKPOINT_STEPS}"
  for step in "${STEP_VALUES[@]}"; do
    local checkpoint output_dir log_path
    checkpoint="${run_dir}/checkpoint-${step}"
    if [[ ! -d "${checkpoint}" ]]; then
      echo "SKIP missing checkpoint: ${checkpoint}"
      continue
    fi
    output_dir="${checkpoint}/eval_${EVAL_SPLIT}_${EVAL_SAMPLE_SIZE}_seed${EVAL_SAMPLE_SEED}"
    log_path="${LOG_ROOT}/${slug}_checkpoint-${step}_eval_${EVAL_SPLIT}_${EVAL_SAMPLE_SIZE}_seed${EVAL_SAMPLE_SEED}.log"
    echo "EVAL checkpoint=${checkpoint} output_dir=${output_dir}"
    PYTHONUNBUFFERED=1 \
      uv run --no-sync python -m sql_agent_training.train.sft_eval \
        --config "${config}" \
        --checkpoint "${checkpoint}" \
        --split "${EVAL_SPLIT}" \
        --sample-size "${EVAL_SAMPLE_SIZE}" \
        --sample-seed "${EVAL_SAMPLE_SEED}" \
        --output-dir "${output_dir}" \
      2>&1 | tee "${log_path}"
  done
  echo "FINISH_SFT_EVAL config=${config} run_dir=${run_dir} $(date)"
}

case "${MODE}" in
  list)
    list_configs
    ;;
  train)
    for config in "${CONFIGS[@]}"; do
      run_train "${config}"
    done
    ;;
  eval)
    for config in "${CONFIGS[@]}"; do
      run_eval "${config}"
    done
    ;;
  all)
    for config in "${CONFIGS[@]}"; do
      run_train "${config}"
      run_eval "${config}"
    done
    ;;
  *)
    usage
    exit 2
    ;;
esac
