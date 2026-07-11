# SQL Agent Training

Minimal Spider text-to-SQL training stack.

The project now keeps only the code needed to understand one small loop:

```text
Spider data -> schema prompt -> SFT records -> SQL agent rollout -> execution reward -> GRPO loss -> optimizer step
```

There is no AutoDL launcher and no VERL integration in this branch. The GRPO path is local and readable: `train.grpo_rollouts` prepares rollouts, while `train.grpo_train` performs online rollout/update steps.

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
uv run python -m sql_agent_training.train.sft_eval --config configs/sft.yaml --split validation --limit 10
```
