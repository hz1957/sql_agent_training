#!/usr/bin/env bash
set -euo pipefail

# Fixed-seed control for the S3 temperature/gamma sweep.
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_kl01_ep2_t10_g09_seed42.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-verl_grpo_s3_kl01_ep2_temp10_gamma09_seed42_14b_h100_4gpu_100step_bs2_n20_branch4_beam2_$(date +%Y%m%d_%H%M%S)}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export DATA_SEED="${DATA_SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export ROLLOUT_TEMPERATURE=1.0
export GRPO_REWARD_GAMMA=0.9

exec bash "${SCRIPT_DIR}/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_kl01_ep2.sh"
