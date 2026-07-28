#!/usr/bin/env bash
set -euo pipefail

# Start vLLM for the merged SFT 14B model, run rollout temperature probes, then stop vLLM.
# Defaults are intentionally small: 16 Spider tasks x 4 rollouts x 3 temperatures.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

resolve_project_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "${PROJECT_DIR}/$1" ;;
  esac
}

MODEL_PATH="$(resolve_project_path "${MODEL_PATH:-data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged}")"
TOKENIZER_PATH="$(resolve_project_path "${TOKENIZER_PATH:-${MODEL_PATH}}")"
OUTPUT_DIR="$(resolve_project_path "${OUTPUT_DIR:-artifacts/rollout_temperature_probe/qwen25_14b_sft_merged_n4_limit16_seed13}")"
LOG_DIR="$(resolve_project_path "${LOG_DIR:-logs}")"
VLLM_LOG="${VLLM_LOG:-${LOG_DIR}/vllm_qwen25_14b_sft_merged_temperature_probe.log}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SLURM_TMPDIR:-/tmp/$USER}/triton_cache}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen25-coder-14b-sft-merged}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-120}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-5}"

TEMPERATURES="${TEMPERATURES:-0.8 1.0 1.2}"
ROLLOUT_N="${ROLLOUT_N:-4}"
LIMIT="${LIMIT:-16}"
MAX_TURNS="${MAX_TURNS:-3}"
MAX_TOKENS_PER_CALL="${MAX_TOKENS_PER_CALL:-512}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
SEED="${SEED:-13}"
SHUFFLE="${SHUFFLE:-1}"
INCLUDE_TURNS="${INCLUDE_TURNS:-0}"
LOG_EVERY="${LOG_EVERY:-1}"

mkdir -p "${LOG_DIR}" "${TRITON_CACHE_DIR}" "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES UV_LINK_MODE TRITON_CACHE_DIR VLLM_USE_V1

read -r -a TEMPERATURE_ARGS <<< "${TEMPERATURES}"
PROBE_ARGS=(
  --backend vllm
  --base-url "${VLLM_BASE_URL}"
  --model-path "${MODEL_PATH}"
  --model-name "${SERVED_MODEL_NAME}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --temperatures "${TEMPERATURE_ARGS[@]}"
  --rollout-n "${ROLLOUT_N}"
  --max-turns "${MAX_TURNS}"
  --max-tokens-per-call "${MAX_TOKENS_PER_CALL}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --limit "${LIMIT}"
  --seed "${SEED}"
  --output-dir "${OUTPUT_DIR}"
  --log-every "${LOG_EVERY}"
)

if [[ "${SHUFFLE}" == "1" || "${SHUFFLE}" == "true" || "${SHUFFLE}" == "True" ]]; then
  PROBE_ARGS+=(--shuffle)
fi
if [[ "${INCLUDE_TURNS}" == "1" || "${INCLUDE_TURNS}" == "true" || "${INCLUDE_TURNS}" == "True" ]]; then
  PROBE_ARGS+=(--include-turns)
fi

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

echo "temperature_probe MODEL_PATH=${MODEL_PATH}"
echo "temperature_probe SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "temperature_probe CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "temperature_probe TEMPERATURES=${TEMPERATURES} ROLLOUT_N=${ROLLOUT_N} LIMIT=${LIMIT}"
echo "temperature_probe OUTPUT_DIR=${OUTPUT_DIR}"
echo "temperature_probe VLLM_LOG=${VLLM_LOG}"

fetch_vllm_models() {
  curl -fsS "${VLLM_BASE_URL}/models" 2>/dev/null || true
}

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
    uv run --no-sync vllm serve "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --dtype "${VLLM_DTYPE}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
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

echo "START temperature probe $(date)"
PYTHONUNBUFFERED=1 uv run --no-sync python scripts/probe_rollout_temperature.py "${PROBE_ARGS[@]}"
echo "FINISH temperature probe $(date)"

echo "SUMMARY:"
cat "${OUTPUT_DIR}/summary.md"
