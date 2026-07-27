# SQL Agent Training

Minimal Spider text-to-SQL training stack.

The project now keeps only the code needed to understand one small loop:

```text
Spider data -> schema prompt -> SFT records -> SQL agent rollout -> execution reward -> GRPO loss -> optimizer step
```

The local GRPO path remains readable: `train.grpo_rollouts` prepares rollouts, while `train.grpo_train` performs online rollout/update steps. An experimental verl entrypoint is also available for server-side async AgentLoop GRPO.

## Layout

```text
sql_agent_training/
  data/       Spider examples, schema prompts, SFT formatting
  env/        SQL safety checks and read-only SQLite execution
  reward/     Spider execution-match reward
  agent/      prompts, SQL extraction, rollouts, trajectory tokenization
  train/      SFT, SFT eval, rollout grouping, minimal complete GRPO trainer
```

## Setup

```powershell
cd sql_agent_training
uv sync --group dev
uv run pytest
```

If `uv run` hits a cache permission issue, keep the cache inside the repo:

```powershell
UV_CACHE="$(pwd)/.cache_uv" XDG_CACHE_HOME="$(pwd)/.cache_xdg" uv run pytest
```

## Data

The real Spider flow expects:

```text
data/spider/
  train_spider.json
  dev.json
  tables.json
  database/
    {db_id}/
      {db_id}.sqlite
```

Prepare or verify data:

```powershell
uv run python scripts/prepare_spider.py --data-dir data/spider --download-hf-text
uv run python scripts/download_spider_assets.py --data-dir data/spider
uv run python scripts/prepare_spider.py --data-dir data/spider --verify-only
```

## Minimal Flow

Format SFT data without training:

```powershell
uv run python -m sql_agent_training.train.sft --config configs/sft.local_dryrun.yaml --dry-run
```

Run the built-in local GRPO rollout demo without updating weights:

```powershell
uv run python -m sql_agent_training.train.grpo_rollouts --config configs/grpo.local_dryrun.yaml
```

Run the built-in complete GRPO trainer. This uses a tiny local causal LM, computes advantages, caches old/reference log-probs, backpropagates clipped GRPO loss, and writes a checkpoint:

```powershell
uv run python -m sql_agent_training.train.grpo_train --config configs/grpo.local_dryrun.yaml
```

Run the same trainer with Qwen2.5-Coder-0.5B for a one-step local smoke test:

```powershell
uv run python scripts/download_model.py
uv run python -m sql_agent_training.train.grpo_train --config configs/grpo.qwen_smoke.yaml
```

Each GRPO trainer run saves a timestamped checkpoint under the configured checkpoint root, for example `artifacts/checkpoints/grpo_qwen25_coder_05b/<timestamp>/`.
The same directory contains that run's `rollouts.jsonl`, `metrics.jsonl`, `metrics.json`, and `run_config.yaml`. Rollout JSONL files include prompt and response text by default for live debugging; set `output.include_text: false` to keep only token counts, rewards, and metadata.

Run formal local GRPO training with the Qwen2.5-Coder-0.5B training config:

```powershell
uv run python -m sql_agent_training.train.grpo_train --config configs/grpo.yaml
```

`configs/grpo.yaml` uses PPO-style online GRPO: every training step samples a task batch, generates `rollout.n` rollouts from the current policy, caches old/reference log-probs, then reuses that rollout batch for `training.update_epochs` clipped actor updates.

## Experimental verl GRPO

Prepare Spider parquet files in verl's default RLHF dataset format:

```powershell
# From the workspace root, the directory that contains uv.lock.
uv run --no-sync python sql_agent_training/scripts/prepare_verl_spider.py
```

Run the 14B LoRA verl GRPO script on a 4x L40S node after installing a compatible verl/vLLM environment.
For the first smoke test, keep both Ray and GRPO very small:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TOTAL_TRAINING_STEPS=2 \
SAVE_FREQ=-1 \
TEST_FREQ=-1 \
TRAIN_BATCH_SIZE=1 \
PPO_MINI_BATCH_SIZE=1 \
ROLLOUT_N=1 \
ROLLOUT_TP=4 \
ROLLOUT_PP=1 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.32 \
ROLLOUT_MAX_NUM_SEQS=1 \
ROLLOUT_LAYERED_SUMMON=True \
USE_KL_IN_REWARD=False \
USE_KL_LOSS=False \
ACTOR_PARAM_OFFLOAD=False \
ACTOR_OPTIMIZER_OFFLOAD=False \
REF_PARAM_OFFLOAD=False \
MAX_PROMPT_LENGTH=2048 \
MAX_RESPONSE_LENGTH=512 \
uv run --no-sync bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_l40s_4gpu.sh
```

For a real GRPO pilot after the smoke succeeds, increase `TRAIN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`, and `ROLLOUT_N`
back to group values such as `4`.

The script assumes a single local Ray node by default and does not pass a Ray `runtime_env`, so Ray workers inherit the
current project checkout directly. Set `ENABLE_RAY_RUNTIME_ENV=1` only for a deployment that needs Ray to package and
ship the working directory. The script also sets `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` by default, because Ray's automatic
`uv run` runtime hook can otherwise rewrite `runtime_env.working_dir` and package the whole project.
For SLURM smoke tests, it also fixes Ray at `RAY_NUM_CPUS=16`, `RAY_OBJECT_STORE_MEMORY=1073741824`, and
`RAY_INCLUDE_DASHBOARD=False` by default to avoid slow local worker/dashboard startup and oversized Ray object-store
allocation inside memory-limited jobs; override those environment variables when more CPU-side rollout workers are
needed.
The smoke defaults also keep CPU subprocess fan-out conservative with `DATALOADER_NUM_WORKERS=0` and
`REWARD_NUM_WORKERS=1`; increase them only after the first GPU smoke succeeds. Smoke runs also default to
`ACTOR_USE_TORCH_COMPILE=False`, `ROLLOUT_ENFORCE_EAGER=True`, `MODEL_TRUST_REMOTE_CODE=False`, and
`MODEL_ATTN_IMPLEMENTATION=sdpa` to reduce initialization-time memory pressure and avoid optional FlashAttention
dependencies. For this verl vLLM rollout path, `ROLLOUT_PP` must remain `1`; the installed verl version rejects
`pipeline_model_parallel_size > 1`. Qwen2.5-Coder-14B has 40 attention heads, so `ROLLOUT_TP` must divide 40. On a
4x L40S node this means the conservative smoke path uses `ROLLOUT_TP=4`, no FSDP offload, `ROLLOUT_MAX_NUM_SEQS=1`,
reference/KL disabled via `USE_KL_LOSS=False` and `USE_KL_IN_REWARD=False`, and a default `ROLLOUT_MAX_MODEL_LEN` of
`MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH`. It also enables `ROLLOUT_LAYERED_SUMMON=True` to reduce peak memory while
syncing LoRA weights into vLLM. If this still cannot fit, lower `ROLLOUT_GPU_MEMORY_UTILIZATION` first, then consider
QLoRA or moving rollout to separate vLLM GPUs.

This path uses `sql_agent_training.train.verl_sql_agent_loop.SpiderSqlAgentLoop` as a custom verl AgentLoop. SQL write/rewrite tokens are trainable, checker/environment tokens are masked out, and the final reward is computed with the existing Spider SQLite execution reward.
The 4x L40S script writes lightweight experiment metrics into the same log: `GPU_MONITOR` lines sample
`memory_used_mb` and `gpu_utilization_pct`, while the SQL AgentLoop reports `rollout_time_sec`, `generate_time_sec`,
`tool_time_sec`, `reward_time_sec`, `prompt_tokens`, `response_tokens`, `trainable_tokens`,
`tokens_per_sec_total`, `tokens_per_sec_trainable`, and `trajectories_per_sec` through verl's console metrics.

Evaluate a trained GRPO checkpoint on Spider validation:

```powershell
uv run python -m sql_agent_training.train.agent_eval --config configs/agent_eval.yaml --checkpoint artifacts/checkpoints/grpo_qwen25_coder_05b/<timestamp>
```

This writes `eval_predictions.jsonl` and `eval_metrics.json` into the checkpoint directory. Run the same command with `--checkpoint data/models/Qwen2.5-Coder-0.5B-Instruct --output-dir artifacts/eval/base_qwen` for a base-model baseline.

Run SFT formatting and the tiny GRPO trainer through the pipeline:

```powershell
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.local_dryrun.yaml
```

For real SFT, install the SFT extra and use `configs/sft.yaml`, which points at the same Qwen2.5-Coder-0.5B checkpoint:

```powershell
uv sync --group dev --extra sft
uv run python scripts/download_model.py
uv run python -m sql_agent_training.train.sft --config configs/sft.yaml
uv run python -m sql_agent_training.train.sft_eval --config configs/sft.yaml --checkpoint artifacts/checkpoints/sft/<timestamp> --split validation --limit 10 --output-dir artifacts/eval/sft_<timestamp>
```

Each SFT run saves an eval-ready model and tokenizer under a timestamped directory below `output.checkpoint_dir`,
for example `artifacts/checkpoints/sft/<timestamp>/`. If intermediate Hugging Face Trainer checkpoints are enabled,
they are written as `checkpoint-*` subdirectories inside that same timestamped run directory.
SFT eval writes `eval_predictions.jsonl` and `eval_metrics.json` into `--output-dir`; without `--output-dir`, it writes
to the checkpoint's `eval/` directory.
