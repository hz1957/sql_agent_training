#!/usr/bin/env bash
set -euo pipefail

# S3 temperature sweep: temperature=1.2, gamma=0.9, KL=0.01, PPO epochs=2.
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_kl01_ep2_t12_g09_seed42.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-verl_grpo_s3_kl01_ep2_temp12_gamma09_seed42_14b_h100_4gpu_100step_bs2_n20_branch4_beam2_$(date +%Y%m%d_%H%M%S)}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export DATA_SEED="${DATA_SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export ROLLOUT_TEMPERATURE=1.2
export GRPO_REWARD_GAMMA=0.9

exec bash "${SCRIPT_DIR}/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_kl01_ep2.sh"
