#!/usr/bin/env bash
set -euo pipefail

# Experimental verl GRPO entrypoint for 14B Qwen Coder LoRA on one 3x L40S node.
# Run from the sql_agent_training project root after preparing data/verl_spider/*.parquet.

MODEL_PATH=${MODEL_PATH:-data/models/Qwen2.5-Coder-14B-Instruct}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-artifacts/checkpoints/sft_qwen25_coder_14b_lora_h100_zero2/20260725_061113/checkpoint-300}
TRAIN_FILES=${TRAIN_FILES:-data/verl_spider/train.parquet}
VAL_FILES=${VAL_FILES:-data/verl_spider/validation.parquet}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.82}
ACTOR_LR=${ACTOR_LR:-5e-7}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-25}
MAX_TURNS=${MAX_TURNS:-3}
PROJECT_NAME=${PROJECT_NAME:-sql_agent_training}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-verl_grpo_qwen25_coder_14b_l40s_3gpu}

export CUDA_VISIBLE_DEVICES

DATA=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=False
  data.train_files="['${TRAIN_FILES}']"
  data.val_files="['${VAL_FILES}']"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.return_raw_chat=True
  data.filter_overlong_prompts=True
  data.truncation=error
)

MODEL=(
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.lora_rank=64
  actor_rollout_ref.model.lora_alpha=128
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}"
)

ACTOR=(
  actor_rollout_ref.actor.strategy=fsdp
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}"
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/verl_sql_agent_loop.yaml
  actor_rollout_ref.rollout.agent.default_agent_loop=sql_agent
  actor_rollout_ref.rollout.agent.num_workers="${TRAIN_BATCH_SIZE}"
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}"
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURNS}"
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
)

REF=(
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
  trainer.balance_batch=True
  trainer.logger='["console"]'
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.nnodes=1
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.val_before_train=False
)

RAY_RUNTIME=(
  '+ray_kwargs.ray_init.runtime_env.excludes=["data/**","artifacts/**","logs/**",".venv/**",".venv-vllm/**",".uv_cache/**",".xdg_cache/**","*.safetensors","*.sqlite"]'
)

python -m verl.trainer.main_ppo \
  "${DATA[@]}" \
  "${MODEL[@]}" \
  "${ACTOR[@]}" \
  "${ROLLOUT[@]}" \
  "${REF[@]}" \
  "${TRAINER[@]}" \
  "${RAY_RUNTIME[@]}" \
  "$@"
