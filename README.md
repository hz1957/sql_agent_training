# SQL Agent Training

This repository now contains a standalone Spider SQL agent training project under:

```text
sql_agent_training/
```

The old Agent-Lightning framework code has been removed. The new project supports:

- Spider data preparation and schema prompts
- SFT data formatting and training skeleton
- SQLite read-only execution
- Spider execution reward
- SQL agent rollout primitives
- GRPO grouping contracts
- VERL Agent Loop probe/adaptation skeleton

## Environment With uv

Install `uv` first if it is not already available:

```powershell
python -m pip install uv
```

Create/sync the local development environment:

```powershell
cd sql_agent_training
uv sync --group dev
```

Run tests:

```powershell
uv run pytest
```

Install training extras when needed, preferably on AutoDL:

```powershell
uv sync --group dev --extra sft
# AutoDL GRPO work will also need:
uv sync --group dev --extra train
```

## Local Smoke Tests

```powershell
cd sql_agent_training
uv run pytest
uv run python -m sql_agent_training.train.verl_agent_grpo --config configs/agent_grpo.local_dryrun.yaml
```

## Spider Data

Download text labels from Hugging Face `xlangai/spider`, then download the official Spider database assets from the Spider mirror used by this project:

```powershell
cd sql_agent_training
uv run python scripts/prepare_spider.py --data-dir data/spider --download-hf-text
uv run python scripts/download_spider_assets.py --data-dir data/spider
uv run python scripts/prepare_spider.py --data-dir data/spider --verify-only
```

The local `sql_agent_training/data/` directory is ignored by git.

## SFT Smoke Test

Download Qwen Coder 0.5B and run the local SFT smoke path:

```powershell
cd sql_agent_training
uv sync --group dev --extra sft
uv run python scripts/download_model.py --model-id Qwen/Qwen2.5-Coder-0.5B-Instruct --output-dir data/models/Qwen2.5-Coder-0.5B-Instruct
uv run python -m sql_agent_training.train.sft --config configs/sft.local_smoke.yaml
uv run python -m sql_agent_training.train.sft_eval --config configs/sft.local_smoke.yaml --split validation --limit 1
```

## AutoDL

See:

```text
sql_agent_training/docs/autodl-runbook.md
```
