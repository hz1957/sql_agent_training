# SQL Agent Training Workspace

This repository contains a minimal standalone Spider SQL-agent training project under:

```text
sql_agent_training/
```

The branch is intentionally small. It keeps:

- Spider data loading and schema prompt rendering
- SFT formatting/training/evaluation
- Read-only SQLite execution and Spider execution reward
- A local SQL agent rollout loop
- Minimal GRPO-style trajectory grouping and a complete tiny GRPO trainer

It removes the previous AutoDL and VERL launcher code so the main logic is easier to read.

## Quick Start

```powershell
cd sql_agent_training
uv sync --group dev
uv run pytest
uv run python -m sql_agent_training.train.grpo_trainer --config configs/grpo.local_dryrun.yaml
```

For a real small model demo, download `HuggingFaceTB/SmolLM2-135M-Instruct` with `scripts/download_model.py` and run `sql_agent_training/configs/grpo.yaml`.

See `sql_agent_training/README.md` for the full minimal flow.
