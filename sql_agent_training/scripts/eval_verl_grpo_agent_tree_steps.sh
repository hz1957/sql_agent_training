#!/usr/bin/env bash
set -euo pipefail

# Gold-free tree inference wrapper for exported verl GRPO checkpoints.
# Launch from the outer workspace root:
#   bash sql_agent_training/scripts/eval_verl_grpo_agent_tree_steps.sh verl/<run_name> 100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EVAL_INFERENCE_MODE="${EVAL_INFERENCE_MODE:-tree}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
export TREE_BRANCH_N="${TREE_BRANCH_N:-4}"
export TREE_BEAM_SIZE="${TREE_BEAM_SIZE:-2}"
export TREE_BEAM_TAU="${TREE_BEAM_TAU:-1.0}"
export TREE_BEAM_EPSILON_RANDOM="${TREE_BEAM_EPSILON_RANDOM:-0.0}"
export TREE_SEED="${TREE_SEED:-42}"

exec bash "${SCRIPT_DIR}/eval_verl_grpo_agent_steps.sh" "$@"
