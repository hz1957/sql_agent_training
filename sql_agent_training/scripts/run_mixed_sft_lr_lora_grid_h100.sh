#!/usr/bin/env bash
set -euo pipefail

# Train the five missing cells in the Mixed-SFT learning-rate x LoRA-rank grid.
# The completed lr=5e-5, r=32 run is intentionally excluded.

usage() {
  cat >&2 <<'EOF'
Usage:
  bash sql_agent_training/scripts/run_mixed_sft_lr_lora_grid_h100.sh [list|train|eval|all]

Default mode:
  train

Environment overrides:
  NUM_GPUS           DeepSpeed GPU count. Default: 4
  SFT_CONFIGS        Space-separated config list. Default: five missing grid cells
  CHECKPOINT_STEPS   Space-separated eval steps. Default: 300
  EVAL_SAMPLE_SIZE   Spider validation sample size. Default: 500
  EVAL_SAMPLE_SEED   Spider validation sample seed. Default: 0
  EVAL_CUDA_VISIBLE_DEVICES GPU used for sequential agent eval. Default: 0
  LOG_ROOT           Log directory. Default: artifacts/logs/sft_mixed_lr_lora_grid
  RUNTIME_CACHE_ROOT Runtime cache root. Default: node-local temporary storage
  DEEPSPEED_INCLUDE  Explicit DeepSpeed include string, e.g. localhost:0,1,2,3

The fixed grid is:
  learning_rate = {2e-5, 3e-5, 5e-5}
  LoRA           = {r32/alpha64, r64/alpha128}

Already complete and not rerun:
  lr=5e-5, r=32, alpha=64
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
    if [[ -x "${candidate}/bin/deepspeed" ]]; then
      export UV_PROJECT_ENVIRONMENT="${candidate}"
      break
    fi
  done
fi

DEFAULT_CONFIGS=(
  configs/sft.qwen25_coder_14b_from_base_mixed_gold3200_direct1137_rewrite463.h100_zero2.lr2e5_r32.yaml
  configs/sft.qwen25_coder_14b_from_base_mixed_gold3200_direct1137_rewrite463.h100_zero2.lr3e5_r32.yaml
  configs/sft.qwen25_coder_14b_from_base_mixed_gold3200_direct1137_rewrite463.h100_zero2.lr2e5_r64.yaml
  configs/sft.qwen25_coder_14b_from_base_mixed_gold3200_direct1137_rewrite463.h100_zero2.lr3e5_r64.yaml
  configs/sft.qwen25_coder_14b_from_base_mixed_gold3200_direct1137_rewrite463.h100_zero2.lr5e5_r64.yaml
)

if [[ -n "${SFT_CONFIGS:-}" ]]; then
  read -r -a CONFIGS <<< "${SFT_CONFIGS}"
else
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

NUM_GPUS="${NUM_GPUS:-4}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-300}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-500}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-0}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/sft_mixed_lr_lora_grid}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}}/sql_agent_training_mixed_sft_grid}"
MIXED_JSONL="artifacts/sft_trajectory/qwen25_coder_14b_sft_merged_q1000_n4_target1600_seed43_t09/mixed_sft_gold3200_direct1137_rewrite463_seed42.jsonl"
EVAL_CONFIG="configs/agent_eval.qwen25_coder_14b_base.yaml"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-${RUNTIME_CACHE_ROOT}/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUNTIME_CACHE_ROOT}/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${RUNTIME_CACHE_ROOT}/cuda}"

DEEPSPEED_CMD=(uv run --no-sync deepspeed)
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && -x "${UV_PROJECT_ENVIRONMENT}/bin/deepspeed" ]]; then
  DEEPSPEED_CMD=("${UV_PROJECT_ENVIRONMENT}/bin/deepspeed")
fi

PYTHON_CMD=(uv run --no-sync python)
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  PYTHON_CMD=("${UV_PROJECT_ENVIRONMENT}/bin/python")
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

checkpoint_root_from_config() {
  awk -F': *' '$1 ~ /^[[:space:]]*checkpoint_dir$/ { print $2; exit }' "$1"
}

learning_rate_from_config() {
  awk -F': *' '$1 ~ /^[[:space:]]*learning_rate$/ { print $2; exit }' "$1"
}

lora_value_from_config() {
  local field="$1"
  local config="$2"
  awk -F': *' -v field="${field}" '$1 ~ "^[[:space:]]*" field "$" { print $2; exit }' "${config}"
}

validate_inputs() {
  if [[ ! -f "${MIXED_JSONL}" ]]; then
    echo "ERROR: missing fixed Mixed-SFT data: ${MIXED_JSONL}" >&2
    echo "Prepare it first with:" >&2
    echo "  bash scripts/run_mixed_trajectory_sft_14b_h100.sh prepare" >&2
    exit 1
  fi

  local config
  for config in "${CONFIGS[@]}"; do
    if [[ ! -f "${config}" ]]; then
      echo "ERROR: missing config: ${config}" >&2
      exit 1
    fi
  done

  if [[ ! -f "${EVAL_CONFIG}" ]]; then
    echo "ERROR: missing eval config: ${EVAL_CONFIG}" >&2
    exit 1
  fi
}

list_configs() {
  local config root lr rank alpha
  printf 'deepspeed_cmd=%s\n' "${DEEPSPEED_CMD[*]}"
  printf 'python_cmd=%s\n' "${PYTHON_CMD[*]}"
  printf 'uv_project_environment=%s\n' "${UV_PROJECT_ENVIRONMENT:-<unset>}"
  printf 'checkpoint_steps=%s eval_sample_size=%s eval_sample_seed=%s\n' \
    "${CHECKPOINT_STEPS}" "${EVAL_SAMPLE_SIZE}" "${EVAL_SAMPLE_SEED}"
  for config in "${CONFIGS[@]}"; do
    root="$(checkpoint_root_from_config "${config}")"
    lr="$(learning_rate_from_config "${config}")"
    rank="$(lora_value_from_config r "${config}")"
    alpha="$(lora_value_from_config alpha "${config}")"
    printf 'lr=%s r=%s alpha=%s config=%s checkpoint_root=%s\n' \
      "${lr}" "${rank}" "${alpha}" "${config}" "${root}"
  done
  echo "SKIP completed cell: lr=0.00005 r=32 alpha=64"
}

latest_run_dir_for_root() {
  local root="$1"
  if [[ ! -d "${root}" ]]; then
    return 1
  fi
  find "${root}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR == 1 { $1=""; sub(/^ /, ""); print; exit }'
}

run_train() {
  local config="$1"
  local root slug log_path
  local launcher_args=()
  root="$(checkpoint_root_from_config "${config}")"
  slug="$(basename "${root}")"
  log_path="${LOG_ROOT}/${slug}_train_$(date +%Y%m%d_%H%M%S).log"

  if [[ -n "${DEEPSPEED_INCLUDE:-}" ]]; then
    launcher_args+=(--include "${DEEPSPEED_INCLUDE}")
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; not passing --num_gpus to DeepSpeed."
  else
    launcher_args+=(--num_gpus "${NUM_GPUS}")
  fi

  echo "START_MIXED_SFT_GRID_TRAIN config=${config} checkpoint_root=${root} num_gpus=${NUM_GPUS} $(date)"
  PYTHONUNBUFFERED=1 \
    "${DEEPSPEED_CMD[@]}" "${launcher_args[@]}" \
      --module sql_agent_training.train.sft \
      --config "${config}" \
    2>&1 | tee "${log_path}"
  echo "FINISH_MIXED_SFT_GRID_TRAIN config=${config} checkpoint_root=${root} log=${log_path} $(date)"
}

run_eval_for_config() {
  local config="$1"
  local root lr rank run_dir
  root="$(checkpoint_root_from_config "${config}")"
  lr="$(learning_rate_from_config "${config}")"
  rank="$(lora_value_from_config r "${config}")"

  if ! run_dir="$(latest_run_dir_for_root "${root}")"; then
    echo "SKIP no run directory under checkpoint_root=${root}"
    return 0
  fi

  read -r -a step_values <<< "${CHECKPOINT_STEPS}"
  for step in "${step_values[@]}"; do
    local checkpoint output_dir log_path slug
    checkpoint="${run_dir}/checkpoint-${step}"
    if [[ ! -d "${checkpoint}" ]]; then
      echo "SKIP missing checkpoint: ${checkpoint}"
      continue
    fi

    slug="$(basename "${root}")_checkpoint-${step}"
    output_dir="${checkpoint}/agent_eval_validation_${EVAL_SAMPLE_SIZE}_seed${EVAL_SAMPLE_SEED}"
    log_path="${LOG_ROOT}/${slug}_agent_eval_$(date +%Y%m%d_%H%M%S).log"
    echo "EVAL_MIXED_SFT_GRID lr=${lr} r=${rank} checkpoint=${checkpoint} output=${output_dir}"
    CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
      PYTHONUNBUFFERED=1 \
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

run_eval() {
  local config
  echo "START_MIXED_SFT_GRID_AGENT_EVAL steps=${CHECKPOINT_STEPS} sample_size=${EVAL_SAMPLE_SIZE} seed=${EVAL_SAMPLE_SEED} $(date)"
  for config in "${CONFIGS[@]}"; do
    run_eval_for_config "${config}"
  done
  echo "FINISH_MIXED_SFT_GRID_AGENT_EVAL $(date)"
}

case "${MODE}" in
  list)
    list_configs
    ;;
  train)
    validate_inputs
    list_configs
    for config in "${CONFIGS[@]}"; do
      run_train "${config}"
    done
    ;;
  eval)
    validate_inputs
    list_configs
    run_eval
    ;;
  all)
    validate_inputs
    list_configs
    for config in "${CONFIGS[@]}"; do
      run_train "${config}"
    done
    run_eval
    ;;
  *)
    usage
    exit 2
    ;;
esac
