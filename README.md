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
- GRPO-style trajectory grouping and an online PPO-style GRPO trainer

It keeps the local trainer readable while also carrying a server-side verl/vLLM launch path for larger GRPO runs.

## Project Files

There are two `pyproject.toml` files on purpose:

- `pyproject.toml`: workspace-level uv configuration. It declares the inner package as a workspace member and keeps
  resolver-wide rules such as mutually exclusive extras and `flash-attn` build isolation handling.
- `sql_agent_training/pyproject.toml`: package-level metadata. It declares the actual Python package dependencies,
  including separate extras for SFT/local training and the server-side `verl-cu126` stack.

The verl/vLLM GPU environment should be reproduced from `sql_agent_training[verl-cu126]`, not from a committed
`.venv` or a full `pip freeze`.

## Quick Start

```powershell
cd sql_agent_training
uv sync --group dev
uv run pytest
uv run python -m sql_agent_training.train.grpo_train --config configs/grpo.local_dryrun.yaml
```

For a real small model demo, download `Qwen/Qwen2.5-Coder-0.5B-Instruct` with `scripts/download_model.py` and run `sql_agent_training/configs/grpo.qwen_smoke.yaml`.

See `sql_agent_training/README.md` for the full minimal flow.
