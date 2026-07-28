#!/usr/bin/env bash
set -euo pipefail

# Conservative 2x H100 entrypoint for Qwen2.5-Coder 14B LoRA verl GRPO.
# Defaults are smoke-test oriented; override env vars for pilot/main runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
export MODEL_PATH="${MODEL_PATH:-data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged}"
export LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-none}"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export TEST_FREQ="${TEST_FREQ:--1}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}"

export ROLLOUT_N="${ROLLOUT_N:-4}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export ROLLOUT_PP="${ROLLOUT_PP:-1}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-4}"
export ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-False}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.2}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
export ROLLOUT_DO_SAMPLE="${ROLLOUT_DO_SAMPLE:-True}"
export LOG_PROB_USE_DYNAMIC_BSZ="${LOG_PROB_USE_DYNAMIC_BSZ:-False}"

export USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
export USE_KL_LOSS="${USE_KL_LOSS:-False}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-False}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
export MAX_TURNS="${MAX_TURNS:-1}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-verl_grpo_qwen25_coder_14b_h100_2gpu}"

case "${PYTORCH_CUDA_ALLOC_CONF:-}" in
  *expandable_segments:True*) unset PYTORCH_CUDA_ALLOC_CONF ;;
esac
case "${PYTORCH_ALLOC_CONF:-}" in
  *expandable_segments:True*) unset PYTORCH_ALLOC_CONF ;;
esac

exec bash "${SCRIPT_DIR}/run_verl_grpo_qwen25_coder_14b_l40s_4gpu.sh" "$@"
