#!/usr/bin/env bash
set -euo pipefail

# 4x H100 S3 update_actor benchmark: dynamic batching + remove padding,
# with a larger PPO token budget per GPU.
#
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_update_actor_dynamic_bsz_tok32768.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
export RUN_NAME="${RUN_NAME:-verl_grpo_s3_update_actor_dynamic_bsz_tok${PPO_MAX_TOKEN_LEN_PER_GPU}_14b_h100_4gpu_${TOTAL_TRAINING_STEPS:-10}step_bs2_n20_ep2_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_verl_grpo_qwen25_coder_14b_h100_4gpu_s3_update_actor_dynamic_bsz.sh"
