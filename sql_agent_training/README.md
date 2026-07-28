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
ROLLOUT_N=4 \
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

With `trainer.balance_batch=True`, `TRAIN_BATCH_SIZE * ROLLOUT_N` must be at least the GPU count. On 4 GPUs, use
`ROLLOUT_N=4` with `TRAIN_BATCH_SIZE=1`, or use a larger `TRAIN_BATCH_SIZE`. Run a no-GPU config dry run first:

```bash
DRY_RUN=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TOTAL_TRAINING_STEPS=2 \
TRAIN_BATCH_SIZE=1 \
PPO_MINI_BATCH_SIZE=1 \
ROLLOUT_N=4 \
ROLLOUT_TP=4 \
MAX_PROMPT_LENGTH=2048 \
MAX_RESPONSE_LENGTH=512 \
uv run --no-sync bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_l40s_4gpu.sh
```

For a real GRPO pilot after the smoke succeeds, increase `TRAIN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`, and `ROLLOUT_N`
carefully while keeping `PPO_MINI_BATCH_SIZE <= TRAIN_BATCH_SIZE`.

Before running the 2x H100 wrapper, merge the SFT LoRA checkpoint into a normal HF model directory. This avoids PEFT
loading the old SFT adapter into verl/FSDP meta tensors during actor startup:

```bash
uv run --no-sync python sql_agent_training/scripts/merge_lora_adapter.py \
  --base-model sql_agent_training/data/models/Qwen2.5-Coder-14B-Instruct \
  --adapter sql_agent_training/artifacts/checkpoints/sft_qwen25_coder_14b_lora_h100_zero2/20260725_061113/checkpoint-300 \
  --output-dir sql_agent_training/data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged \
  --dtype bfloat16 \
  --device-map auto
```

For a 2x H100 node, use the H100 wrapper. It defaults to the same small smoke shape, but uses the merged SFT model
as `MODEL_PATH`, sets `LORA_ADAPTER_PATH=none`, initializes a fresh trainable GRPO LoRA, and uses `ROLLOUT_TP=2`,
`NGPUS_PER_NODE=2`, no offload, `ROLLOUT_GPU_MEMORY_UTILIZATION=0.30`, and `ROLLOUT_LAYERED_SUMMON=True` so LoRA
parameter sync does not need to summon the full FSDP model at once. H100 smoke defaults use `ROLLOUT_N=4` and
`ROLLOUT_TEMPERATURE=1.2` to make same-prompt rollouts more likely to produce non-zero GRPO advantages.
The wrapper unsets allocator
`expandable_segments:True` when present because it is incompatible with vLLM's CuMem memory pool:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
DRY_RUN=1 \
uv run --no-sync bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_2gpu.sh
```

Remove `DRY_RUN=1` to start Ray/vLLM. The launch scripts validate the common verl batch-size relationships before
starting GPU work:

- `TRAIN_BATCH_SIZE`: number of prompts sampled from the dataset per trainer step.
- `ROLLOUT_N`: number of sampled responses per prompt. GRPO uses these as the comparison group.
- `TRAIN_BATCH_SIZE * ROLLOUT_N`: number of rollout trajectories produced for a step; with batch balancing, this must
  be at least the number of GPUs.
- `PPO_MINI_BATCH_SIZE`: number of original prompts used in one PPO/GRPO actor update mini-batch; verl requires this to
  be no larger than `TRAIN_BATCH_SIZE`.
- `PPO_MICRO_BATCH_SIZE_PER_GPU`: per-GPU chunk size used inside actor forward/backward to control memory.
- `LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`: per-GPU chunk size for old/reference log-prob recomputation when
  `LOG_PROB_USE_DYNAMIC_BSZ=False`. It defaults to `1` for smoke tests because this path avoids the optional
  FlashAttention padding helper.

The script assumes a single local Ray node by default and does not pass a Ray `runtime_env`, so Ray workers inherit the
current project checkout directly. Set `ENABLE_RAY_RUNTIME_ENV=1` only for a deployment that needs Ray to package and
ship the working directory. The script also sets `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` by default, because Ray's automatic
`uv run` runtime hook can otherwise rewrite `runtime_env.working_dir` and package the whole project.
verl checkpoints are written under `artifacts/checkpoints/verl/<experiment_name>/` by default. Put run logs under
`artifacts/logs/` as well so generated training outputs stay under the single `artifacts/` root; for example, from the
outer workspace root redirect to `sql_agent_training/artifacts/logs/<run_name>.log`.
By default the verl actor checkpoint attempts to save only LoRA adapter model weights via
`actor_rollout_ref.actor.checkpoint.save_lora_only=True` and
`actor_rollout_ref.actor.checkpoint.save_contents=["model"]`. The launch script checks the installed verl
`CheckpointConfig` before starting Ray and fails early if that verl build does not support `save_lora_only`.
Set `ACTOR_CHECKPOINT_SAVE_LORA_ONLY=False` to run with model-only full-model checkpoints on older verl builds, or
upgrade verl if small LoRA-only checkpoints are required. LoRA-only checkpoints keep storage small, but do not preserve
optimizer or extra RNG/scheduler state for exact training resume. Override `ACTOR_CHECKPOINT_SAVE_CONTENTS` and
`ACTOR_CHECKPOINT_LOAD_CONTENTS` if a fully resumable checkpoint is needed.
The server-side H100/CUDA 12.6 verl environment is recorded as the `verl-cu126` extra in
`sql_agent_training/pyproject.toml`. Use it from the outer workspace root:

```bash
cd /home/hice1/hzhang961/scratch/sql_agent_training
module load cuda/12.6.1
export UV_LINK_MODE=copy
export UV_TORCH_BACKEND=cu126
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"

uv run --no-sync ray stop -f || true
uv sync --python 3.12 --package sql-agent-training --extra verl-cu126 --group dev

uv run --no-sync python - <<'PY'
import dataclasses
import importlib.metadata as md

import flash_attn
import peft
import torch
import transformers
import vllm
from verl.trainer.config.config import CheckpointConfig

fields = [field.name for field in dataclasses.fields(CheckpointConfig)]
print("torch:", torch.__version__, torch.version.cuda)
print("vllm:", vllm.__version__)
print("peft:", peft.__version__)
print("transformers:", transformers.__version__)
print("flash_attn:", getattr(flash_attn, "__version__", "unknown"))
print("verl:", md.version("verl"))
print("CheckpointConfig fields:", fields)
assert "save_lora_only" in fields, fields
PY

uv pip check
PREFLIGHT=1 ACTOR_CHECKPOINT_SAVE_LORA_ONLY=True \
  uv run --no-sync bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_l40s_4gpu.sh
```

The outer workspace `pyproject.toml` marks `verl-cu126` as incompatible with the older `sft` and `train` extras, so uv
will refuse to install both stacks into one environment. Do not run `uv sync --extra train` in the verl environment:
that extra is for the simpler local training stack and pins `torch<2.6`, while this verl/vLLM path has been validated
with the CUDA 12.6 `torch 2.9.0` / `vllm 0.12.0` stack. The workspace config also sets
`no-build-isolation-package = ["flash-attn"]`, so `flash-attn` can build against the already-resolved PyTorch/CUDA
environment during `uv sync`.

If an existing server environment is already good and only the PyPI `verl` package lacks
`checkpoint.save_lora_only`, the lowest-churn repair remains replacing only the verl package without touching
torch/vLLM:

```bash
uv pip install --python .venv/bin/python --no-deps --force-reinstall \
  "verl @ git+https://github.com/verl-project/verl.git@main"
```
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
