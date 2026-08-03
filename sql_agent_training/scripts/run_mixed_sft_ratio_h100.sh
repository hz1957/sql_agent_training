#!/usr/bin/env bash
set -euo pipefail

# Prepare, train, and optionally evaluate the Mixed-SFT data-ratio experiment.
# All cells keep 4,800 training records, 300 optimizer steps, lr=5e-5, and
# LoRA r=32/alpha=64. Only the gold/trajectory composition changes.

usage() {
  cat >&2 <<'EOF'
Usage:
  bash sql_agent_training/scripts/run_mixed_sft_ratio_h100.sh [list|prepare|train|eval|all]

Default mode:
  train

Recommended 4xH100 training:
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NUM_GPUS=4 \
  bash sql_agent_training/scripts/run_mixed_sft_ratio_h100.sh train

Modes:
  list     Print the fixed ratio cells and generated config paths.
  prepare  Build the four 4,800-record SFT JSONL files and generated configs.
  train    Train the four generated configs sequentially. Run prepare first.
  eval     Agent-evaluate checkpoint-300 for the latest run of each ratio.
  all      prepare, train, then eval.

Environment overrides:
  NUM_GPUS                   DeepSpeed GPU count. Default: 4
  RATIO_CELLS                Space-separated cell slugs to run.
  CHECKPOINT_STEPS           Space-separated eval steps. Default: 300
  EVAL_SAMPLE_SIZE           Spider validation sample size. Default: 500
  EVAL_SAMPLE_SEED           Spider validation sample seed. Default: 0
  EVAL_CUDA_VISIBLE_DEVICES  GPU used for sequential agent eval. Default: 0
  LOG_ROOT                   Log directory. Default: artifacts/logs/sft_ratio
  RUNTIME_CACHE_ROOT         Runtime cache root. Default: node-local temporary storage
  DEEPSPEED_INCLUDE          Explicit DeepSpeed include string, e.g. localhost:0,1,2,3

Fixed cells:
  gold4800_d0_r0       4,800 gold + 0 direct + 0 rewrite
  gold4000_d568_r232   4,000 gold + 568 direct + 232 rewrite
  gold3200_d1137_r463  3,200 gold + 1,137 direct + 463 rewrite
  gold2400_d1706_r694  2,400 gold + 1,706 direct + 694 rewrite
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
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-300}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-500}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-0}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
LOG_ROOT="${LOG_ROOT:-artifacts/logs/sft_ratio}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}}/sql_agent_training_sft_ratio}"

TRAJECTORY_DIR="artifacts/sft_trajectory/qwen25_coder_14b_sft_merged_q1000_n4_target1600_seed43_t09"
TRAJECTORY_JSONL="${TRAJECTORY_DIR}/trajectory_sft.jsonl"
GENERATED_CONFIG_DIR="artifacts/configs/sft_ratio"
EVAL_CONFIG="configs/agent_eval.qwen25_coder_14b_base.yaml"

CELL_SPECS=(
  "gold4800_d0_r0 4800 0 0"
  "gold4000_d568_r232 4000 568 232"
  "gold3200_d1137_r463 3200 1137 463"
  "gold2400_d1706_r694 2400 1706 694"
)

if [[ -n "${RATIO_CELLS:-}" ]]; then
  read -r -a SELECTED_CELLS <<< "${RATIO_CELLS}"
else
  SELECTED_CELLS=(
    gold4800_d0_r0
    gold4000_d568_r232
    gold3200_d1137_r463
    gold2400_d1706_r694
  )
fi

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
  "${GENERATED_CONFIG_DIR}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${CUDA_CACHE_PATH}"

cell_selected() {
  local slug="$1"
  local selected
  for selected in "${SELECTED_CELLS[@]}"; do
    [[ "${selected}" == "${slug}" ]] && return 0
  done
  return 1
}

cell_config_path() {
  local slug="$1"
  printf '%s/sft.qwen25_coder_14b_from_base_ratio_%s.h100_zero2.lr5e5_r32.yaml\n' \
    "${GENERATED_CONFIG_DIR}" "${slug}"
}

cell_jsonl_path() {
  local slug="$1"
  printf '%s/mixed_sft_ratio_%s_seed42.jsonl\n' "${TRAJECTORY_DIR}" "${slug}"
}

cell_summary_path() {
  local slug="$1"
  printf '%s/mixed_sft_ratio_%s_seed42_summary.json\n' "${TRAJECTORY_DIR}" "${slug}"
}

cell_checkpoint_root() {
  local slug="$1"
  printf 'artifacts/checkpoints/sft_14b_from_base_ratio_%s_lr5e5_r32\n' "${slug}"
}

write_config() {
  local slug="$1"
  local jsonl="$2"
  local config_path root
  config_path="$(cell_config_path "${slug}")"
  root="$(cell_checkpoint_root "${slug}")"
  cat > "${config_path}" <<EOF
model:
  path: data/models/Qwen2.5-Coder-14B-Instruct

tokenizer:
  kind: hf

data:
  data_dir: data/spider
  train_file: train_spider.json
  validation_file: dev.json
  preformatted_sft_jsonl: ${jsonl}

output:
  sft_jsonl: ${jsonl}
  checkpoint_dir: ${root}

eval:
  sample_size: 100
  train_sample_size: 100
  sample_seed: 0

training:
  max_prompt_length: 3584
  max_response_length: 512
  learning_rate: 0.00005
  num_train_epochs: 1
  max_steps: 300
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  warmup_ratio: 0.03
  weight_decay: 0.0
  lr_scheduler_type: linear
  max_grad_norm: 1.0
  seed: 42
  data_seed: 42
  logging_steps: 10
  eval_strategy: steps
  eval_steps: 100
  per_device_eval_batch_size: 1
  eval_accumulation_steps: 1
  save_strategy: steps
  save_steps: 100
  bf16: true
  fp16: false
  report_to: none
  deepspeed: configs/deepspeed_zero2_h100.json

lora:
  enabled: true
  r: 32
  alpha: 64
  dropout: 0.05
  bias: none
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
EOF
}

validate_common_inputs() {
  if [[ ! -f "${TRAJECTORY_JSONL}" ]]; then
    echo "ERROR: missing trajectory SFT pool: ${TRAJECTORY_JSONL}" >&2
    echo "Regenerate it first with: bash scripts/run_regenerate_sft_trajectories_for_ratio.sh" >&2
    exit 1
  fi
  if [[ ! -f "${EVAL_CONFIG}" ]]; then
    echo "ERROR: missing eval config: ${EVAL_CONFIG}" >&2
    exit 1
  fi
}

list_cells() {
  echo "python_cmd=${PYTHON_CMD[*]}"
  echo "deepspeed_cmd=${DEEPSPEED_CMD[*]}"
  echo "uv_project_environment=${UV_PROJECT_ENVIRONMENT:-<unset>}"
  echo "trajectory_jsonl=${TRAJECTORY_JSONL}"
  local spec slug gold direct rewrite total
  for spec in "${CELL_SPECS[@]}"; do
    read -r slug gold direct rewrite <<< "${spec}"
    cell_selected "${slug}" || continue
    total=$((gold + direct + rewrite))
    printf 'slug=%s total=%s gold=%s direct=%s rewrite=%s data=%s config=%s checkpoint_root=%s\n' \
      "${slug}" "${total}" "${gold}" "${direct}" "${rewrite}" \
      "$(cell_jsonl_path "${slug}")" "$(cell_config_path "${slug}")" "$(cell_checkpoint_root "${slug}")"
  done
}

run_prepare() {
  validate_common_inputs
  local spec slug gold direct rewrite jsonl summary log_path
  for spec in "${CELL_SPECS[@]}"; do
    read -r slug gold direct rewrite <<< "${spec}"
    cell_selected "${slug}" || continue
    jsonl="$(cell_jsonl_path "${slug}")"
    summary="$(cell_summary_path "${slug}")"
    log_path="${LOG_ROOT}/${slug}_prepare_$(date +%Y%m%d_%H%M%S).log"
    echo "START_SFT_RATIO_PREPARE slug=${slug} gold=${gold} direct=${direct} rewrite=${rewrite} $(date)"
    PYTHONUNBUFFERED=1 \
      "${PYTHON_CMD[@]}" scripts/prepare_mixed_sft.py \
        --trajectory-jsonl "${TRAJECTORY_JSONL}" \
        --output-jsonl "${jsonl}" \
        --summary-json "${summary}" \
        --gold-count "${gold}" \
        --direct-count "${direct}" \
        --rewrite-count "${rewrite}" \
        --seed 42 \
      2>&1 | tee "${log_path}"
    write_config "${slug}" "${jsonl}"
    echo "FINISH_SFT_RATIO_PREPARE slug=${slug} data=${jsonl} config=$(cell_config_path "${slug}") summary=${summary} log=${log_path} $(date)"
  done
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

run_train_for_cell() {
  local slug="$1"
  local config root log_path
  local launcher_args=()
  config="$(cell_config_path "${slug}")"
  root="$(cell_checkpoint_root "${slug}")"
  if [[ ! -f "${config}" ]]; then
    echo "ERROR: missing generated config for ${slug}: ${config}" >&2
    echo "Run '$0 prepare' first." >&2
    exit 1
  fi

  if [[ -n "${DEEPSPEED_INCLUDE:-}" ]]; then
    launcher_args+=(--include "${DEEPSPEED_INCLUDE}")
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; not passing --num_gpus to DeepSpeed."
  else
    launcher_args+=(--num_gpus "${NUM_GPUS}")
  fi

  log_path="${LOG_ROOT}/${slug}_train_$(date +%Y%m%d_%H%M%S).log"
  echo "START_SFT_RATIO_TRAIN slug=${slug} config=${config} checkpoint_root=${root} num_gpus=${NUM_GPUS} $(date)"
  PYTHONUNBUFFERED=1 \
    "${DEEPSPEED_CMD[@]}" "${launcher_args[@]}" \
      --module sql_agent_training.train.sft \
      --config "${config}" \
    2>&1 | tee "${log_path}"
  echo "FINISH_SFT_RATIO_TRAIN slug=${slug} checkpoint_root=${root} log=${log_path} $(date)"
}

run_train() {
  validate_common_inputs
  local spec slug gold direct rewrite
  for spec in "${CELL_SPECS[@]}"; do
    read -r slug gold direct rewrite <<< "${spec}"
    cell_selected "${slug}" || continue
    run_train_for_cell "${slug}"
  done
}

run_eval_for_cell() {
  local slug="$1"
  local root run_dir
  root="$(cell_checkpoint_root "${slug}")"
  if ! run_dir="$(latest_run_dir_for_root "${root}")"; then
    echo "SKIP no run directory under checkpoint_root=${root}"
    return 0
  fi

  read -r -a step_values <<< "${CHECKPOINT_STEPS}"
  local step checkpoint output_dir log_path
  for step in "${step_values[@]}"; do
    checkpoint="${run_dir}/checkpoint-${step}"
    if [[ ! -d "${checkpoint}" ]]; then
      echo "SKIP missing checkpoint: ${checkpoint}"
      continue
    fi
    output_dir="${checkpoint}/agent_eval_validation_${EVAL_SAMPLE_SIZE}_seed${EVAL_SAMPLE_SEED}"
    log_path="${LOG_ROOT}/${slug}_checkpoint-${step}_agent_eval_$(date +%Y%m%d_%H%M%S).log"
    echo "EVAL_SFT_RATIO slug=${slug} checkpoint=${checkpoint} output=${output_dir}"
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
  validate_common_inputs
  local spec slug gold direct rewrite
  echo "START_SFT_RATIO_AGENT_EVAL steps=${CHECKPOINT_STEPS} sample_size=${EVAL_SAMPLE_SIZE} seed=${EVAL_SAMPLE_SEED} $(date)"
  for spec in "${CELL_SPECS[@]}"; do
    read -r slug gold direct rewrite <<< "${spec}"
    cell_selected "${slug}" || continue
    run_eval_for_cell "${slug}"
  done
  echo "FINISH_SFT_RATIO_AGENT_EVAL $(date)"
}

case "${MODE}" in
  list)
    list_cells
    ;;
  prepare)
    run_prepare
    ;;
  train)
    list_cells
    run_train
    ;;
  eval)
    list_cells
    run_eval
    ;;
  all)
    list_cells
    run_prepare
    run_train
    run_eval
    ;;
  *)
    usage
    exit 2
    ;;
esac
