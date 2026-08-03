#!/usr/bin/env bash
set -euo pipefail

# Generate up to 800 verified trajectories from 500 Spider questions x 4 rollouts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

resolve_project_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${PROJECT_DIR}/$1" ;;
  esac
}

MODEL_PATH="$(resolve_project_path "${MODEL_PATH:-data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged}")"
TOKENIZER_PATH="$(resolve_project_path "${TOKENIZER_PATH:-${MODEL_PATH}}")"
OUTPUT_DIR="$(resolve_project_path "${OUTPUT_DIR:-artifacts/sft_trajectory/qwen25_coder_14b_sft_merged_q500_n4_target800_seed42}")"
LOG_DIR="$(resolve_project_path "${LOG_DIR:-artifacts/logs/sft_trajectory}")"
GENERATOR_LOG="${GENERATOR_LOG:-${LOG_DIR}/generate_q500_n4_target800.log}"
VLLM_LOG="${VLLM_LOG:-${LOG_DIR}/vllm_qwen25_coder_14b_sft_merged.log}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp/$USER}/sql_agent_trajectory_cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/uv}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_CACHE_ROOT}/xdg}"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${RUNTIME_CACHE_ROOT}/xdg_config}"
HF_HOME="${HF_HOME:-${RUNTIME_CACHE_ROOT}/huggingface}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/torchinductor}"
TORCH_HOME="${TORCH_HOME:-${RUNTIME_CACHE_ROOT}/torch}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${RUNTIME_CACHE_ROOT}/torch_extensions}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${RUNTIME_CACHE_ROOT}/cuda}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/numba}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${RUNTIME_CACHE_ROOT}/vllm}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
DO_NOT_TRACK="${DO_NOT_TRACK:-1}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen25-coder-14b-sft-merged}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-120}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-5}"

QUESTION_COUNT="${QUESTION_COUNT:-500}"
ROLLOUTS_PER_QUESTION="${ROLLOUTS_PER_QUESTION:-4}"
TARGET_CORRECT="${TARGET_CORRECT:-800}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_TURNS="${MAX_TURNS:-3}"
MAX_TOKENS_PER_CALL="${MAX_TOKENS_PER_CALL:-512}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
WORKERS="${WORKERS:-32}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
LOG_EVERY="${LOG_EVERY:-25}"

PYTHON_CMD=(uv run --no-sync python)
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  PYTHON_CMD=("${UV_PROJECT_ENVIRONMENT}/bin/python")
fi

if [[ -z "${VLLM_BIN:-}" ]]; then
  for candidate in \
    "${VLLM_PROJECT_ENVIRONMENT:-}/bin/vllm" \
    "${WORKSPACE_DIR}/.venv-vllm/bin/vllm" \
    "${PROJECT_DIR}/.venv-vllm/bin/vllm" \
    "${WORKSPACE_DIR}/.venv-verl/bin/vllm" \
    "${PROJECT_DIR}/.venv-verl/bin/vllm" \
    "${WORKSPACE_DIR}/.venv/bin/vllm" \
    "${PROJECT_DIR}/.venv/bin/vllm" \
    "${WORKSPACE_DIR}/.venv-sft/bin/vllm" \
    "${PROJECT_DIR}/.venv-sft/bin/vllm"; do
    if [[ -x "${candidate}" ]]; then
      VLLM_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -n "${VLLM_BIN:-}" ]]; then
  VLLM_CMD=("${VLLM_BIN}")
elif command -v vllm >/dev/null 2>&1; then
  VLLM_CMD=("$(command -v vllm)")
else
  VLLM_CMD=(uv run --no-sync vllm)
fi

mkdir -p \
  "${LOG_DIR}" \
  "${OUTPUT_DIR}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${HF_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${NUMBA_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}"
export \
  CUDA_VISIBLE_DEVICES \
  UV_LINK_MODE \
  UV_CACHE_DIR \
  XDG_CACHE_HOME \
  XDG_CONFIG_HOME \
  HF_HOME \
  TRITON_CACHE_DIR \
  TORCHINDUCTOR_CACHE_DIR \
  TORCH_HOME \
  TORCH_EXTENSIONS_DIR \
  CUDA_CACHE_PATH \
  NUMBA_CACHE_DIR \
  VLLM_CACHE_ROOT \
  VLLM_USE_V1 \
  VLLM_NO_USAGE_STATS \
  DO_NOT_TRACK

VLLM_PID=""
STARTED_VLLM=0

cleanup() {
  if [[ "${STARTED_VLLM}" == "1" && -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM pid=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

fetch_vllm_models() {
  curl -fsS "${VLLM_BASE_URL}/models" 2>/dev/null || true
}

echo "trajectory_generation MODEL_PATH=${MODEL_PATH}"
echo "trajectory_generation QUESTIONS=${QUESTION_COUNT} ROLLOUTS=${ROLLOUTS_PER_QUESTION} TARGET=${TARGET_CORRECT}"
echo "trajectory_generation TEMPERATURE=${TEMPERATURE} MAX_TURNS=${MAX_TURNS} WORKERS=${WORKERS}"
echo "trajectory_generation CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} TP=${VLLM_TENSOR_PARALLEL_SIZE}"
echo "trajectory_generation RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT}"
echo "trajectory_generation OUTPUT_DIR=${OUTPUT_DIR}"
echo "trajectory_generation PYTHON_CMD=${PYTHON_CMD[*]}"
echo "trajectory_generation VLLM_CMD=${VLLM_CMD[*]}"

MODELS_JSON="$(fetch_vllm_models)"
if [[ "${MODELS_JSON}" == *"${SERVED_MODEL_NAME}"* ]]; then
  echo "Reusing existing vLLM server at ${VLLM_BASE_URL}"
elif [[ -n "${MODELS_JSON}" ]]; then
  echo "ERROR: ${VLLM_BASE_URL} is already serving a different model."
  echo "Stop that vLLM process or set VLLM_PORT to another free port."
  exit 2
else
  echo "Starting vLLM at ${VLLM_BASE_URL}"
  PYTHONUNBUFFERED=1 \
    "${VLLM_CMD[@]}" serve "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --dtype "${VLLM_DTYPE}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
      --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
      > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  STARTED_VLLM=1
  echo "VLLM_PID=${VLLM_PID}"

  for attempt in $(seq 1 "${WAIT_ATTEMPTS}"); do
    MODELS_JSON="$(fetch_vllm_models)"
    if [[ "${MODELS_JSON}" == *"${SERVED_MODEL_NAME}"* ]]; then
      echo "vLLM_READY $(date)"
      break
    fi
    if grep -qE "Engine core initialization failed|EngineCore failed to start" "${VLLM_LOG}" 2>/dev/null; then
      echo "ERROR: vLLM reported a fatal startup error. Last vLLM log lines:"
      tail -80 "${VLLM_LOG}" || true
      exit 1
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "ERROR: vLLM exited before ready. Last vLLM log lines:"
      tail -80 "${VLLM_LOG}" || true
      exit 1
    fi
    echo "waiting for vLLM... ${attempt}/${WAIT_ATTEMPTS}"
    sleep "${WAIT_INTERVAL_SEC}"
  done

  MODELS_JSON="$(fetch_vllm_models)"
  if [[ "${MODELS_JSON}" != *"${SERVED_MODEL_NAME}"* ]]; then
    echo "ERROR: vLLM did not become ready. Last vLLM log lines:"
    tail -80 "${VLLM_LOG}" || true
    exit 1
  fi
fi

echo "START trajectory generation $(date)"
PYTHONUNBUFFERED=1 \
  "${PYTHON_CMD[@]}" scripts/generate_sft_trajectories.py \
    --base-url "${VLLM_BASE_URL}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${SERVED_MODEL_NAME}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --question-count "${QUESTION_COUNT}" \
    --rollouts-per-question "${ROLLOUTS_PER_QUESTION}" \
    --target-correct "${TARGET_CORRECT}" \
    --seed "${SEED}" \
    --temperature "${TEMPERATURE}" \
    --max-turns "${MAX_TURNS}" \
    --max-tokens-per-call "${MAX_TOKENS_PER_CALL}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --workers "${WORKERS}" \
    --request-retries "${REQUEST_RETRIES}" \
    --log-every "${LOG_EVERY}" \
    --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${GENERATOR_LOG}"
echo "FINISH trajectory generation $(date)"
echo "Summary: ${OUTPUT_DIR}/summary.json"
echo "Verified trajectories: ${OUTPUT_DIR}/verified_trajectories.jsonl"
echo "SFT records: ${OUTPUT_DIR}/trajectory_sft.jsonl"
