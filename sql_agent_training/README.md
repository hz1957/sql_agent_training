# SQL Agent Training

Minimal Spider text-to-SQL training stack.

The project now keeps only the code needed to understand one small loop:

```text
Spider data -> schema prompt -> SFT records -> SQL agent rollout -> execution reward -> GRPO loss -> optimizer step
```

There is no AutoDL launcher and no VERL integration in this branch. The GRPO path is local and readable: `train.grpo` prepares rollouts, while `train.grpo_trainer` performs the actual weight update.

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
uv run python -m sql_agent_training.train.grpo --config configs/grpo.local_dryrun.yaml
```

Run the built-in complete GRPO trainer. This uses a tiny local causal LM, computes advantages, caches old/reference log-probs, backpropagates clipped GRPO loss, and writes a checkpoint:

```powershell
uv run python -m sql_agent_training.train.grpo_trainer --config configs/grpo.local_dryrun.yaml
```

Run the same trainer with the real 135M parameter causal LM configured in `configs/grpo.yaml`:

```powershell
uv run python scripts/download_model.py
uv run python -m sql_agent_training.train.grpo_trainer --config configs/grpo.yaml
```

Run SFT formatting and the tiny GRPO trainer through the pipeline:

```powershell
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.local_dryrun.yaml
```

For real SFT, install the SFT extra and use `configs/sft.yaml`, which points at the same SmolLM2-135M checkpoint:

```powershell
uv sync --group dev --extra sft
uv run python scripts/download_model.py
uv run python -m sql_agent_training.train.sft --config configs/sft.yaml
uv run python -m sql_agent_training.train.sft_eval --config configs/sft.yaml --split validation --limit 10
```
