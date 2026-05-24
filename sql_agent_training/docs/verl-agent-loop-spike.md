# VERL Agent Loop Spike

Date: 2026-05-17

Local probe result:

```text
verl version: 0.7.1
agent loop module: verl.experimental.agent_loop.agent_loop
AgentLoopBase: verl.experimental.agent_loop.agent_loop.AgentLoopBase
AgentLoopOutput: verl.experimental.agent_loop.agent_loop.AgentLoopOutput
AgentLoopMetrics: verl.experimental.agent_loop.agent_loop.AgentLoopMetrics
registry decorator: register(agent_name)
```

`AgentLoopBase.run` signature:

```text
(self, sampling_params: dict[str, typing.Any], **kwargs) -> AgentLoopOutput
```

Required `AgentLoopOutput` fields:

```text
prompt_ids: list[int]
response_ids: list[int]
response_mask: list[int]
metrics: AgentLoopMetrics
```

Useful optional fields:

```text
reward_score: float | None
num_turns: int
extra_fields: dict[str, Any]
response_logprobs: list[float] | None
```

Implemented project bridge:

```text
sql_agent_training.train.verl_sql_agent_loop.SqlAgentVerlLoop
registered name: sql_agent
config: configs/verl_sql_agent_loop.yaml
```

SQL agent loop protocol:

```text
write SQL -> execute SQL -> if reward passes, final -> otherwise rewrite SQL -> repeat until max_turns
```

The model emits one read-only SQLite `SELECT`/`WITH` query each turn.

VERL dataset contract:

```text
prompt: list[{"role": "user", "content": "..."}]
agent_name: sql_agent
extra_info:
  uid
  question
  db_id
  schema_prompt
  gold_sql
  sqlite_path
```

Prepare a small local parquet:

```powershell
uv run python scripts/prepare_verl_spider.py --data-dir data/spider --split-file train_spider.json --output artifacts/verl/spider_train_smoke.parquet --limit 2
```

Verified locally:

```text
VERL RLHFDataset can load the parquet.
raw_prompt, agent_name, and extra_info are preserved.
Hydra can resolve SqlAgentVerlLoop from configs/verl_sql_agent_loop.yaml.
The generated prompt uses the simple SQL rewrite protocol.
```

AutoDL one-batch command:

```bash
PYTHONPATH=. uv run python -m sql_agent_training.train.verl_agent_grpo --config configs/agent_grpo.autodl.yaml --run-verl
```

The project entrypoint prepares the Spider parquet files, then runs the following VERL shape:

```bash
PYTHONPATH=. uv run python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen2.5-Coder-0.5B-Instruct \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length=512 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.agent.default_agent_loop=sql_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/verl_sql_agent_loop.yaml \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  data.train_files=/root/autodl-tmp/data/verl/spider_train_smoke.parquet \
  data.val_files=/root/autodl-tmp/data/verl/spider_train_smoke.parquet \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.return_raw_chat=true \
  data.filter_overlong_prompts=false \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.n_gpus_per_node=1 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.logger='[console]' \
  reward.reward_model.enable=false
```

Remaining Milestone 5 server-side check:

```text
Run the AutoDL command with CUDA/vLLM and confirm one actor optimizer step completes.
```
