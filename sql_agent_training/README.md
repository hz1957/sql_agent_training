# SQL Agent Training

Standalone training stack for Spider text-to-SQL agents.

The project supports three modes:

- SFT only: `question + schema -> gold SQL`
- Agent GRPO only: SQL agent rollout with execution reward
- SFT followed by Agent GRPO

Default base model:

```text
Qwen/Qwen2.5-Coder-0.5B-Instruct
```

## Local vs AutoDL

Local RTX 3070 is for development only: data verification, unit tests, schema formatting, SQLite reward checks, and dry runs.
Real SFT and Agent GRPO are intended to run on AutoDL.

## Local Setup

```powershell
cd sql_agent_training
uv sync --group dev
uv run pytest
```

Run commands from this directory so `uv`, pytest, and module imports all resolve against the subproject.

## Data

Text samples come from Hugging Face `xlangai/spider`. Execution reward also requires official Spider SQLite databases:

```text
data/spider/
  tables.json
  database/
    {db_id}/
      {db_id}.sqlite
```

Verify a prepared dataset:

```powershell
python scripts/prepare_spider.py --data-dir data/spider --verify-only
```

Current local preparation workflow:

```powershell
uv run python scripts/prepare_spider.py --data-dir data/spider --download-hf-text
uv run python scripts/download_spider_assets.py --data-dir data/spider
uv run python scripts/prepare_spider.py --data-dir data/spider --verify-only
```

Expected verified counts:

```text
train examples: 7000
validation examples: 1034
schemas: 166
missing db ids: 0
```

## Commands

SFT formatting / dry run:

```powershell
uv run python -m sql_agent_training.train.sft --config configs/sft.local_dryrun.yaml --dry-run
```

Local Qwen Coder 0.5B SFT smoke:

```powershell
uv sync --group dev --extra sft
uv run python scripts/download_model.py --model-id Qwen/Qwen2.5-Coder-0.5B-Instruct --output-dir data/models/Qwen2.5-Coder-0.5B-Instruct
uv run python -m sql_agent_training.train.sft --config configs/sft.local_smoke.yaml
uv run python -m sql_agent_training.train.sft_eval --config configs/sft.local_smoke.yaml --split validation --limit 1
```

SFT eval dry run:

```powershell
uv run python -m sql_agent_training.train.sft_eval --config configs/sft.local_dryrun.yaml --dry-run-gold
```

Agent GRPO entrypoint:

```powershell
uv run python -m sql_agent_training.train.verl_agent_grpo --config configs/agent_grpo.local_dryrun.yaml
```

Agent rollout protocol:

```text
model writes one SQLite SELECT/WITH query
runner executes it
if execution reward passes, the SQL is final
otherwise the execution feedback is returned and the model rewrites SQL
max_turns falls back to the last candidate SQL
```

Pipeline:

```powershell
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.yaml --stages sft,grpo
```

Print local dry-run commands without running training:

```powershell
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.local_dryrun.yaml --print-commands
```

Run one stage only:

```powershell
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.yaml --stages sft --print-commands
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.yaml --stages grpo --print-commands
```

AutoDL full sequence, assuming VERL/CUDA environment is ready:

```bash
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.autodl.yaml
```

## AutoDL

See [docs/autodl-runbook.md](docs/autodl-runbook.md).

Useful probes:

```bash
bash scripts/autodl_bootstrap.sh
python scripts/probe_verl_agent_loop.py
```
