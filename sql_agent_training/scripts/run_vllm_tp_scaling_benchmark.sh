#!/usr/bin/env bash
set -euo pipefail

# vLLM TP scaling benchmark for Qwen2.5-Coder-14B SQL-agent prompts.
#
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_vllm_tp_scaling_benchmark.sh single
#   bash sql_agent_training/scripts/run_vllm_tp_scaling_benchmark.sh fixed
#   bash sql_agent_training/scripts/run_vllm_tp_scaling_benchmark.sh all
#
# Cases:
#   single      TP=1 on 1 GPU, TP=2 on 2 GPUs, TP=4 on 4 GPUs.
#   fixed       4 replicas x TP=1, 2 replicas x TP=2, 1 replica x TP=4.
#   tp1,tp2,tp4,rep4tp1,rep2tp2,rep1tp4 run one case only.

LAUNCH_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MODE="${1:-all}"
case "${MODE}" in
  all|single|fixed|tp1|tp2|tp4|rep4tp1|rep2tp2|rep1tp4) ;;
  *)
    echo "Usage: $0 {all|single|fixed|tp1|tp2|tp4|rep4tp1|rep2tp2|rep1tp4}"
    exit 2
    ;;
esac

MODEL_PATH="${MODEL_PATH:-data/models/Qwen2.5-Coder-14B-Instruct-SFT-Ratio-Gold3200-D1137-R463-LR5e5-R32-Merged}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen25-coder-14b-sql}"
DEFAULT_DATASET_PARQUET="data/verl_spider/validation.parquet"
if [[ ! -f "${DEFAULT_DATASET_PARQUET}" && -f "sql_agent_training/data/verl_spider/validation.parquet" ]]; then
  DEFAULT_DATASET_PARQUET="sql_agent_training/data/verl_spider/validation.parquet"
fi
DATASET_PARQUET="${DATASET_PARQUET:-${DEFAULT_DATASET_PARQUET}}"
RESULT_ROOT_RAW="${RESULT_ROOT:-artifacts/logs/vllm_tp_scaling/$(date +%Y%m%d_%H%M%S)}"
if [[ "${RESULT_ROOT_RAW}" = /* ]]; then
  RESULT_ROOT="${RESULT_ROOT_RAW}"
else
  RESULT_ROOT="${LAUNCH_DIR}/${RESULT_ROOT_RAW}"
fi

LIMIT="${LIMIT:-128}"
REPETITIONS="${REPETITIONS:-1}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-900}"

DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PORT_BASE="${PORT_BASE:-8100}"
STREAM="${STREAM:-1}"
SHUFFLE="${SHUFFLE:-1}"
SEED="${SEED:-0}"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "${RESULT_ROOT}"

SERVER_PIDS=()
MONITOR_PID=""

cleanup_case() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
  for pid in "${SERVER_PIDS[@]:-}"; do
    # Every server is started in its own session. Terminating the process group
    # also stops uv, the API server, and vLLM engine worker descendants.
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  for _attempt in {1..15}; do
    local any_alive=0
    for pid in "${SERVER_PIDS[@]:-}"; do
      if kill -0 -- "-${pid}" 2>/dev/null; then
        any_alive=1
      fi
    done
    (( any_alive == 0 )) && break
    sleep 1
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 -- "-${pid}" 2>/dev/null; then
      echo "WARNING: force-stopping vLLM process group ${pid}"
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
  SERVER_PIDS=()
}

trap cleanup_case EXIT
trap 'cleanup_case; exit 130' INT
trap 'cleanup_case; exit 143' TERM

start_gpu_monitor() {
  local output_csv="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU monitor skipped: nvidia-smi not found"
    return 0
  fi
  (
    echo "timestamp,index,memory_used_mb,memory_total_mb,gpu_utilization_pct"
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
      sleep 1
    done
  ) > "${output_csv}" &
  MONITOR_PID="$!"
}

start_server() {
  local name="$1"
  local cuda_devices="$2"
  local tp="$3"
  local port="$4"
  local log_file="$5"

  if ! command -v setsid >/dev/null 2>&1; then
    echo "ERROR: setsid is required for reliable vLLM process cleanup." >&2
    return 1
  fi

  echo "START_SERVER name=${name} cuda=${cuda_devices} tp=${tp} port=${port} $(date)" > "${log_file}"
  setsid env CUDA_VISIBLE_DEVICES="${cuda_devices}" \
    PYTHONUNBUFFERED=1 \
    uv run --no-sync python -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" \
      --tokenizer "${TOKENIZER_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${tp}" \
      --dtype "${DTYPE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --host 127.0.0.1 \
      --port "${port}" >> "${log_file}" 2>&1 &
  local pid="$!"
  SERVER_PIDS+=("${pid}")
  echo "SERVER_PID ${pid} ${name} ${log_file}"
}

wait_server() {
  local port="$1"
  local pid="$2"
  local log_file="$3"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
  until uv run --no-sync python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${port}/v1/models', timeout=2).read()" >/dev/null 2>&1; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: server on port ${port} exited before readiness."
      echo "SERVER LOG (last 120 lines): ${log_file}"
      tail -n 120 "${log_file}" 2>/dev/null || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "ERROR: timed out waiting for server on port ${port}."
      return 1
    fi
    sleep 5
  done
}

run_case() {
  local case_name="$1"
  local server_specs="$2"
  local case_dir="${RESULT_ROOT}/${case_name}"
  local server_dir="${case_dir}/servers"
  local gpu_csv="${case_dir}/gpu_monitor.csv"
  local urls=()
  local stream_args=()
  local shuffle_args=()

  cleanup_case
  mkdir -p "${server_dir}"
  echo
  echo "CASE ${case_name}"
  echo "RESULT_DIR ${case_dir}"
  echo "CONFIG model=${MODEL_PATH} dataset=${DATASET_PARQUET} limit=${LIMIT} repetitions=${REPETITIONS} concurrency=${CONCURRENCY} max_tokens=${MAX_TOKENS} stream=${STREAM}"

  start_gpu_monitor "${gpu_csv}"

  for spec in ${server_specs}; do
    IFS=: read -r cuda_devices tp port <<< "${spec}"
    start_server "${case_name}_tp${tp}_${port}" "${cuda_devices}" "${tp}" "${port}" "${server_dir}/server_tp${tp}_port${port}.log"
    urls+=("http://127.0.0.1:${port}/v1")
  done

  for index in "${!SERVER_PIDS[@]}"; do
    local spec_index=0
    for spec in ${server_specs}; do
      if (( spec_index == index )); then
        IFS=: read -r _cuda_devices _tp port <<< "${spec}"
        wait_server \
          "${port}" \
          "${SERVER_PIDS[$index]}" \
          "${server_dir}/server_tp${_tp}_port${port}.log"
      fi
      spec_index=$((spec_index + 1))
    done
  done

  case "${STREAM}" in
    1|true|True|TRUE|yes|Yes|YES) stream_args=(--stream) ;;
    *) stream_args=() ;;
  esac
  case "${SHUFFLE}" in
    1|true|True|TRUE|yes|Yes|YES) shuffle_args=(--shuffle) ;;
    *) shuffle_args=() ;;
  esac

  uv run --no-sync python "${SCRIPT_DIR}/bench_vllm_tp_scaling.py" \
    --case-name "${case_name}" \
    --base-urls "${urls[@]}" \
    --model-name "${SERVED_MODEL_NAME}" \
    --dataset-parquet "${DATASET_PARQUET}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --limit "${LIMIT}" \
    --repetitions "${REPETITIONS}" \
    --seed "${SEED}" \
    "${shuffle_args[@]}" \
    --concurrency "${CONCURRENCY}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --timeout-seconds "${TIMEOUT_SECONDS}" \
    "${stream_args[@]}" \
    --gpu-monitor-csv "${gpu_csv}" \
    --output-dir "${case_dir}" | tee "${case_dir}/benchmark_stdout.log"

  cleanup_case
}

if [[ "${MODE}" == "all" || "${MODE}" == "single" ]]; then
  run_case "single_tp1" "0:1:$((PORT_BASE + 1))"
  run_case "single_tp2" "0,1:2:$((PORT_BASE + 2))"
  run_case "single_tp4" "0,1,2,3:4:$((PORT_BASE + 4))"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "fixed" ]]; then
  run_case "fixed_4rep_tp1" "0:1:$((PORT_BASE + 11)) 1:1:$((PORT_BASE + 12)) 2:1:$((PORT_BASE + 13)) 3:1:$((PORT_BASE + 14))"
  run_case "fixed_2rep_tp2" "0,1:2:$((PORT_BASE + 21)) 2,3:2:$((PORT_BASE + 22))"
  run_case "fixed_1rep_tp4" "0,1,2,3:4:$((PORT_BASE + 31))"
fi

case "${MODE}" in
  tp1) run_case "single_tp1" "0:1:$((PORT_BASE + 1))" ;;
  tp2) run_case "single_tp2" "0,1:2:$((PORT_BASE + 2))" ;;
  tp4) run_case "single_tp4" "0,1,2,3:4:$((PORT_BASE + 4))" ;;
  rep4tp1) run_case "fixed_4rep_tp1" "0:1:$((PORT_BASE + 11)) 1:1:$((PORT_BASE + 12)) 2:1:$((PORT_BASE + 13)) 3:1:$((PORT_BASE + 14))" ;;
  rep2tp2) run_case "fixed_2rep_tp2" "0,1:2:$((PORT_BASE + 21)) 2,3:2:$((PORT_BASE + 22))" ;;
  rep1tp4) run_case "fixed_1rep_tp4" "0,1,2,3:4:$((PORT_BASE + 31))" ;;
esac

echo
echo "DONE result_root=${RESULT_ROOT}"
find "${RESULT_ROOT}" -name summary.json -print
