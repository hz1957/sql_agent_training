#!/usr/bin/env bash
set -euo pipefail

# Export verl FSDP actor checkpoints to eval_models, then run SQL agent eval.
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/eval_verl_grpo_agent_steps.sh \
#     verl/<run_name>

usage() {
  cat >&2 <<'EOF'
Usage:
  bash sql_agent_training/scripts/eval_verl_grpo_agent_steps.sh RUN_PATH [STEP ...]

Example:
  bash sql_agent_training/scripts/eval_verl_grpo_agent_steps.sh \
    verl/verl_grpo_s1_chain_final_14b_h100_4gpu_150step_bs4_n8_t10_turn3_g09_20260728_172036

Environment overrides:
  MODEL_PATH              Base HF model path. Default: data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged
  EVAL_CONFIG             Existing eval config path. If unset, a 14B config is written under artifacts/eval/.
  EVAL_SAMPLE_SIZE        Default: 300
  EVAL_SAMPLE_SEED        Default: 0
  MAX_TURNS               Default: 3
  MAX_RESPONSE_LENGTH     Default: 2048
  EVAL_INFERENCE_MODE     chain or tree. Default: chain
  EVAL_TEMPERATURE        Default: 0.0 for chain, 1.0 for tree
  TREE_BRANCH_N           Default: 4
  TREE_BEAM_SIZE          Default: 2
  TREE_BEAM_TAU           Default: 1.0
  TREE_BEAM_EPSILON_RANDOM Default: 0.0
  TREE_SEED               Default: 42
  CHECKER_BACKEND          Optional separate checker backend: openai_chat or sglang
  CHECKER_API_URL          Optional checker API base URL, e.g. https://api.deepseek.com
  CHECKER_MODEL_NAME       Optional checker model name, e.g. deepseek-chat
  CHECKER_API_KEY_ENV      Optional checker API key env var. Default in Python: LLM_API_KEY_AGENT
  CHECKER_TEMPERATURE      Optional checker temperature. Default in Python: 0.0 when checker is set
  EVAL_SPLIT              Default: validation
  EVAL_MODEL_ROOT         Default: artifacts/eval_models/<run_name>
  EVAL_OUTPUT_ROOT        Default: artifacts/eval/<run_name> for chain,
                           artifacts/eval/<run_name>_tree_b*_beam*_t*_seed* for tree;
                           appends checker_<model/backend> when CHECKER_* is set
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_ARG="${1:-}"
if [[ -z "${RUN_ARG}" ]]; then
  usage
  exit 2
fi
shift

if [[ "$#" -gt 0 ]]; then
  STEP_VALUES=("$@")
elif [[ -n "${EVAL_STEPS:-}" ]]; then
  read -r -a STEP_VALUES <<< "${EVAL_STEPS}"
else
  STEP_VALUES=(100)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

case "${RUN_ARG}" in
  /*)
    RUN_DIR="${RUN_ARG}"
    ;;
  *)
    if [[ -d "${RUN_ARG}" ]]; then
      RUN_DIR="${RUN_ARG}"
    elif [[ -d "artifacts/checkpoints/${RUN_ARG}" ]]; then
      RUN_DIR="artifacts/checkpoints/${RUN_ARG}"
    elif [[ -d "artifacts/checkpoints/verl/${RUN_ARG}" ]]; then
      RUN_DIR="artifacts/checkpoints/verl/${RUN_ARG}"
    else
      RUN_DIR="artifacts/checkpoints/${RUN_ARG}"
    fi
    ;;
esac

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run path does not exist: ${RUN_ARG}" >&2
  echo "       resolved as: ${RUN_DIR}" >&2
  exit 1
fi

RUN_NAME="$(basename "${RUN_DIR}")"
MODEL_PATH="${MODEL_PATH:-data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-300}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-0}"
MAX_TURNS="${MAX_TURNS:-3}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
EVAL_INFERENCE_MODE="${EVAL_INFERENCE_MODE:-chain}"
if [[ -z "${EVAL_TEMPERATURE+x}" ]]; then
  if [[ "${EVAL_INFERENCE_MODE}" == "tree" ]]; then
    EVAL_TEMPERATURE="1.0"
  else
    EVAL_TEMPERATURE="0.0"
  fi
fi
TREE_BRANCH_N="${TREE_BRANCH_N:-4}"
TREE_BEAM_SIZE="${TREE_BEAM_SIZE:-2}"
TREE_BEAM_TAU="${TREE_BEAM_TAU:-1.0}"
TREE_BEAM_EPSILON_RANDOM="${TREE_BEAM_EPSILON_RANDOM:-0.0}"
TREE_SEED="${TREE_SEED:-42}"
CHECKER_BACKEND="${CHECKER_BACKEND:-}"
CHECKER_API_URL="${CHECKER_API_URL:-}"
CHECKER_MODEL_NAME="${CHECKER_MODEL_NAME:-}"
CHECKER_API_KEY_ENV="${CHECKER_API_KEY_ENV:-}"
CHECKER_REQUEST_TIMEOUT_SECONDS="${CHECKER_REQUEST_TIMEOUT_SECONDS:-}"
CHECKER_TEMPERATURE="${CHECKER_TEMPERATURE:-}"
CHECKER_OUTPUT_TAG="${CHECKER_OUTPUT_TAG:-}"
if [[ -z "${CHECKER_OUTPUT_TAG}" && -n "${CHECKER_BACKEND}${CHECKER_API_URL}${CHECKER_MODEL_NAME}${CHECKER_API_KEY_ENV}" ]]; then
  CHECKER_OUTPUT_TAG="checker_${CHECKER_MODEL_NAME:-${CHECKER_BACKEND:-remote}}"
fi
CHECKER_OUTPUT_TAG="${CHECKER_OUTPUT_TAG//\//_}"
CHECKER_OUTPUT_TAG="${CHECKER_OUTPUT_TAG//:/_}"
CHECKER_OUTPUT_TAG="${CHECKER_OUTPUT_TAG// /_}"
CHECKER_OUTPUT_SUFFIX=""
if [[ -n "${CHECKER_OUTPUT_TAG}" ]]; then
  CHECKER_OUTPUT_SUFFIX="_${CHECKER_OUTPUT_TAG}"
fi
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
EVAL_MODEL_ROOT="${EVAL_MODEL_ROOT:-artifacts/eval_models/${RUN_NAME}}"
if [[ -z "${EVAL_OUTPUT_ROOT+x}" ]]; then
  if [[ "${EVAL_INFERENCE_MODE}" == "tree" ]]; then
    TEMP_TAG="${EVAL_TEMPERATURE//./}"
    EVAL_OUTPUT_ROOT="artifacts/eval/${RUN_NAME}_tree_b${TREE_BRANCH_N}_beam${TREE_BEAM_SIZE}_t${TEMP_TAG}_seed${TREE_SEED}${CHECKER_OUTPUT_SUFFIX}"
  else
    EVAL_OUTPUT_ROOT="artifacts/eval/${RUN_NAME}${CHECKER_OUTPUT_SUFFIX}"
  fi
fi
if [[ -n "${EVAL_CONFIG:-}" ]]; then
  WRITE_EVAL_CONFIG=0
else
  EVAL_CONFIG="artifacts/eval/agent_eval_qwen25_coder_14b_sft_merged.yaml"
  WRITE_EVAL_CONFIG=1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$(dirname "${EVAL_CONFIG}")" "${EVAL_MODEL_ROOT}" "${EVAL_OUTPUT_ROOT}"

if [[ "${WRITE_EVAL_CONFIG}" == "1" ]]; then
  cat > "${EVAL_CONFIG}" <<YAML
model:
  backend: hf
  path: ${MODEL_PATH}
  device: cuda
  torch_dtype: bf16

data:
  data_dir: data/spider
  train_file: train_spider.json
  validation_file: dev.json

rollout:
  max_turns: ${MAX_TURNS}
  max_response_length: ${MAX_RESPONSE_LENGTH}
  temperature: ${EVAL_TEMPERATURE}
  inference_mode: ${EVAL_INFERENCE_MODE}
  tree_branch_n: ${TREE_BRANCH_N}
  tree_beam_size: ${TREE_BEAM_SIZE}
  tree_beam_tau: ${TREE_BEAM_TAU}
  tree_beam_epsilon_random: ${TREE_BEAM_EPSILON_RANDOM}
  tree_seed: ${TREE_SEED}

eval:
  sample_size: ${EVAL_SAMPLE_SIZE}
  sample_seed: ${EVAL_SAMPLE_SEED}
YAML
elif [[ ! -f "${EVAL_CONFIG}" ]]; then
  echo "ERROR: EVAL_CONFIG does not exist: ${EVAL_CONFIG}" >&2
  exit 1
fi

echo "RUN_DIR=${RUN_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "EVAL_CONFIG=${EVAL_CONFIG}"
echo "EVAL_INFERENCE_MODE=${EVAL_INFERENCE_MODE}"
if [[ -n "${CHECKER_OUTPUT_TAG}" ]]; then
  echo "CHECKER_OUTPUT_TAG=${CHECKER_OUTPUT_TAG}"
fi
echo "STEPS=${STEP_VALUES[*]}"

AGENT_EVAL_EXTRA_ARGS=()
if [[ -n "${CHECKER_BACKEND}" ]]; then
  AGENT_EVAL_EXTRA_ARGS+=(--checker-backend "${CHECKER_BACKEND}")
fi
if [[ -n "${CHECKER_API_URL}" ]]; then
  AGENT_EVAL_EXTRA_ARGS+=(--checker-api-url "${CHECKER_API_URL}")
fi
if [[ -n "${CHECKER_MODEL_NAME}" ]]; then
  AGENT_EVAL_EXTRA_ARGS+=(--checker-model-name "${CHECKER_MODEL_NAME}")
fi
if [[ -n "${CHECKER_API_KEY_ENV}" ]]; then
  AGENT_EVAL_EXTRA_ARGS+=(--checker-api-key-env "${CHECKER_API_KEY_ENV}")
fi
if [[ -n "${CHECKER_REQUEST_TIMEOUT_SECONDS}" ]]; then
  AGENT_EVAL_EXTRA_ARGS+=(--checker-request-timeout-seconds "${CHECKER_REQUEST_TIMEOUT_SECONDS}")
fi
if [[ -n "${CHECKER_TEMPERATURE}" ]]; then
  AGENT_EVAL_EXTRA_ARGS+=(--checker-temperature "${CHECKER_TEMPERATURE}")
fi

for STEP in "${STEP_VALUES[@]}"; do
  ACTOR_DIR="${RUN_DIR}/global_step_${STEP}/actor"
  HF_DIR="${EVAL_MODEL_ROOT}/global_step_${STEP}_hf"
  OUTPUT_DIR="${EVAL_OUTPUT_ROOT}/global_step_${STEP}"

  if [[ ! -d "${ACTOR_DIR}" ]]; then
    echo "ERROR: missing actor checkpoint for step ${STEP}: ${ACTOR_DIR}" >&2
    exit 1
  fi

  echo "EXPORT step=${STEP} actor=${ACTOR_DIR} target=${HF_DIR}"
  uv run --no-sync python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${ACTOR_DIR}" \
    --target_dir "${HF_DIR}" \
    --trust-remote-code \
    --use_cpu_initialization

  if find "${HF_DIR}" -maxdepth 1 -type f \
    \( -name "model.safetensors" -o -name "pytorch_model.bin" -o -name "model-*.safetensors" \) \
    | grep -q .; then
    echo "ERROR: full-model weight files appeared in ${HF_DIR}. Check that the checkpoint was saved with save_lora_only=True." >&2
    exit 1
  fi

  if [[ ! -d "${HF_DIR}/lora_adapter" ]]; then
    echo "ERROR: missing LoRA adapter after export: ${HF_DIR}/lora_adapter" >&2
    exit 1
  fi

  echo "EVAL step=${STEP} checkpoint=${HF_DIR}/lora_adapter output=${OUTPUT_DIR}"
  uv run --no-sync python -m sql_agent_training.train.agent_eval \
    --config "${EVAL_CONFIG}" \
    --checkpoint "${HF_DIR}/lora_adapter" \
    --tokenizer "${HF_DIR}" \
    --split "${EVAL_SPLIT}" \
    --output-dir "${OUTPUT_DIR}" \
    --inference-mode "${EVAL_INFERENCE_MODE}" \
    --tree-branch-n "${TREE_BRANCH_N}" \
    --tree-beam-size "${TREE_BEAM_SIZE}" \
    --tree-beam-tau "${TREE_BEAM_TAU}" \
    --tree-beam-epsilon-random "${TREE_BEAM_EPSILON_RANDOM}" \
    --tree-seed "${TREE_SEED}" \
    "${AGENT_EVAL_EXTRA_ARGS[@]}"
done
