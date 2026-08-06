#!/usr/bin/env bash
set -euo pipefail

# Follow-up vLLM TP experiments:
#   saturation  fixed 4-GPU budget at global concurrency 32/64, three independent runs.
#   agent       complete SQL write/check/rewrite traces, three independent runs.
#   all         run both phases and aggregate mean/sample standard deviation.
#   summarize   rebuild aggregate_summary.json/csv from existing runs.

LAUNCH_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-all}"

case "${MODE}" in
  all|saturation|agent|summarize) ;;
  *)
    echo "Usage: $0 {all|saturation|agent|summarize}" >&2
    exit 2
    ;;
esac

RESULT_ROOT_RAW="${RESULT_ROOT:-artifacts/logs/vllm_tp_followup/$(date +%Y%m%d_%H%M%S)}"
if [[ "${RESULT_ROOT_RAW}" = /* ]]; then
  FOLLOWUP_ROOT="${RESULT_ROOT_RAW}"
else
  FOLLOWUP_ROOT="${LAUNCH_DIR}/${RESULT_ROOT_RAW}"
fi

SATURATION_CONCURRENCIES="${SATURATION_CONCURRENCIES:-32 64}"
SATURATION_RUNS="${SATURATION_RUNS:-3}"
AGENT_CONCURRENCIES="${AGENT_CONCURRENCIES:-8}"
AGENT_RUNS="${AGENT_RUNS:-3}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

LIMIT="${LIMIT:-300}"
SEED="${SEED:-0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TURNS="${MAX_TURNS:-3}"
DATASET_PARQUET="${DATASET_PARQUET:-data/verl_spider/validation.parquet}"
DATA_DIR="${DATA_DIR:-data/spider}"

RUNNER="${SCRIPT_DIR}/run_vllm_tp_scaling_benchmark.sh"
SUMMARIZER="${SCRIPT_DIR}/summarize_vllm_tp_followup.py"
TOPOLOGY_MODES=(rep4tp1 rep2tp2 rep1tp4)

if (( SATURATION_RUNS < 1 || AGENT_RUNS < 1 )); then
  echo "ERROR: SATURATION_RUNS and AGENT_RUNS must be positive." >&2
  exit 2
fi

mkdir -p "${FOLLOWUP_ROOT}"
cd "${PROJECT_DIR}"

case_name_for_mode() {
  case "$1" in
    rep4tp1) echo "fixed_4rep_tp1" ;;
    rep2tp2) echo "fixed_2rep_tp2" ;;
    rep1tp4) echo "fixed_1rep_tp4" ;;
    *) return 1 ;;
  esac
}

is_completed() {
  local run_root="$1"
  local topology_mode="$2"
  local case_name
  case_name="$(case_name_for_mode "${topology_mode}")"
  [[ "${SKIP_COMPLETED}" == "1" && -f "${run_root}/${case_name}/summary.json" ]]
}

run_saturation() {
  local concurrency
  local run_index
  local run_root
  local topology_offset
  local topology_index
  local topology_mode
  for concurrency in ${SATURATION_CONCURRENCIES}; do
    for ((run_index = 1; run_index <= SATURATION_RUNS; run_index++)); do
      run_root="${FOLLOWUP_ROOT}/saturation/concurrency_${concurrency}/run_${run_index}"
      echo
      echo "FOLLOWUP phase=saturation concurrency=${concurrency} run=${run_index}/${SATURATION_RUNS}"
      for ((topology_offset = 0; topology_offset < ${#TOPOLOGY_MODES[@]}; topology_offset++)); do
        topology_index=$(((run_index - 1 + topology_offset) % ${#TOPOLOGY_MODES[@]}))
        topology_mode="${TOPOLOGY_MODES[${topology_index}]}"
        echo "FOLLOWUP_TOPOLOGY mode=${topology_mode} order=$((topology_offset + 1))/${#TOPOLOGY_MODES[@]}"
        if is_completed "${run_root}" "${topology_mode}"; then
          echo "SKIP_COMPLETED ${run_root}/$(case_name_for_mode "${topology_mode}")/summary.json"
          continue
        fi
        BENCHMARK_KIND=prompt \
        LIMIT="${LIMIT}" \
        REPETITIONS=1 \
        SEED="${SEED}" \
        SHUFFLE=1 \
        STREAM=1 \
        CONCURRENCY="${concurrency}" \
        MAX_TOKENS="${MAX_TOKENS}" \
        TEMPERATURE="${TEMPERATURE}" \
        DATASET_PARQUET="${DATASET_PARQUET}" \
        RESULT_ROOT="${run_root}" \
        bash "${RUNNER}" "${topology_mode}"
      done
    done
  done
}

run_agent_traces() {
  local concurrency
  local run_index
  local run_root
  local topology_offset
  local topology_index
  local topology_mode
  if [[ ! -f "${DATASET_PARQUET}" ]]; then
    echo "ERROR: missing agent dataset: ${DATASET_PARQUET}" >&2
    return 1
  fi
  if [[ ! -d "${DATA_DIR}/database" ]]; then
    echo "ERROR: missing Spider database directory: ${DATA_DIR}/database" >&2
    return 1
  fi
  for concurrency in ${AGENT_CONCURRENCIES}; do
    for ((run_index = 1; run_index <= AGENT_RUNS; run_index++)); do
      run_root="${FOLLOWUP_ROOT}/agent_trace/concurrency_${concurrency}/run_${run_index}"
      echo
      echo "FOLLOWUP phase=agent_trace concurrency=${concurrency} run=${run_index}/${AGENT_RUNS}"
      for ((topology_offset = 0; topology_offset < ${#TOPOLOGY_MODES[@]}; topology_offset++)); do
        topology_index=$(((run_index - 1 + topology_offset) % ${#TOPOLOGY_MODES[@]}))
        topology_mode="${TOPOLOGY_MODES[${topology_index}]}"
        echo "FOLLOWUP_TOPOLOGY mode=${topology_mode} order=$((topology_offset + 1))/${#TOPOLOGY_MODES[@]}"
        if is_completed "${run_root}" "${topology_mode}"; then
          echo "SKIP_COMPLETED ${run_root}/$(case_name_for_mode "${topology_mode}")/summary.json"
          continue
        fi
        BENCHMARK_KIND=agent \
        LIMIT="${LIMIT}" \
        SEED="${SEED}" \
        SHUFFLE=1 \
        STREAM=0 \
        CONCURRENCY="${concurrency}" \
        MAX_TURNS="${MAX_TURNS}" \
        MAX_TOKENS="${MAX_TOKENS}" \
        TEMPERATURE="${TEMPERATURE}" \
        DATASET_PARQUET="${DATASET_PARQUET}" \
        DATA_DIR="${DATA_DIR}" \
        RESULT_ROOT="${run_root}" \
        bash "${RUNNER}" "${topology_mode}"
      done
    done
  done
}

summarize_runs() {
  if [[ -z "$(find "${FOLLOWUP_ROOT}" -name summary.json -print -quit)" ]]; then
    echo "ERROR: no summary.json files found under ${FOLLOWUP_ROOT}" >&2
    return 1
  fi
  uv run --no-sync python "${SUMMARIZER}" --result-root "${FOLLOWUP_ROOT}"
}

case "${MODE}" in
  saturation)
    run_saturation
    summarize_runs
    ;;
  agent)
    run_agent_traces
    summarize_runs
    ;;
  all)
    run_saturation
    run_agent_traces
    summarize_runs
    ;;
  summarize)
    summarize_runs
    ;;
esac

echo
echo "FOLLOWUP_DONE result_root=${FOLLOWUP_ROOT}"
echo "AGGREGATE_JSON ${FOLLOWUP_ROOT}/aggregate_summary.json"
echo "AGGREGATE_CSV ${FOLLOWUP_ROOT}/aggregate_summary.csv"
