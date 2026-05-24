# SQL Agent Training Plan

Date: 2026-05-17

This project is now a standalone rewrite under `sql_agent_training/`. The old Agent-Lightning framework code is not part of the active runtime path.

## Current Status

Completed locally:

- Milestone 0: standalone uv subproject.
- Milestone 1: Spider text data, SQLite databases, `tables.json`, and schema cache verified.
- Milestone 2: Qwen2.5-Coder-0.5B-Instruct local SFT smoke completed.
- Milestone 3: read-only SQLite execution and Spider execution reward.
- Milestone 4: simple SQL rewrite agent loop.
- Milestone 5: local VERL AgentLoop API adapter and parquet contract.
- Milestone 6: old framework cleanup and pipeline command orchestration.

Not locally verifiable:

- Real CUDA/vLLM/VERL one-batch Agent GRPO actor update. This must be run on AutoDL with `configs/agent_grpo.autodl.yaml`.

## Core Scope

Use Spider as the single data source:

```text
question + schema prompt -> gold SQL
```

The gold SQL is used as:

- SFT label during supervised fine-tuning.
- Reward reference inside the evaluator during GRPO.

The gold SQL must never be included in the Agent GRPO rollout prompt.

Default model:

```text
Qwen/Qwen2.5-Coder-0.5B-Instruct
```

Supported modes:

```text
SFT only
Agent GRPO only
SFT -> Agent GRPO
```

## Directory

```text
sql_agent_training/
  configs/
  docs/
  scripts/
  tests/
  sql_agent_training/
    agent/
    data/
    env/
    reward/
    train/
```

Runtime code must not import:

```text
agentlightning
agl.*
```

## Spider Data

Text labels are downloaded from Hugging Face `xlangai/spider`.

Execution reward also requires Spider database assets:

```text
data/spider/
  train_spider.json
  dev.json
  tables.json
  schema_cache.json
  database/
    {db_id}/
      {db_id}.sqlite
```

Local verified counts:

```text
train examples: 7000
validation examples: 1034
schemas: 166
missing db ids: 0
```

Verification:

```powershell
uv run python scripts/prepare_spider.py --data-dir data/spider --verify-only
```

## SFT

SFT trains:

```text
question + schema prompt -> gold SQL
```

Local smoke:

```powershell
uv run python -m sql_agent_training.train.sft --config configs/sft.local_smoke.yaml
uv run python -m sql_agent_training.train.sft_eval --config configs/sft.local_smoke.yaml --split validation --limit 1
```

AutoDL:

```bash
uv run python -m sql_agent_training.train.sft --config configs/sft.autodl.yaml
```

## Agent Loop

The current agent protocol is intentionally simple:

```text
model writes one read-only SQLite SELECT/WITH query
runner executes the query
if execution reward passes, that SQL is final
otherwise execution feedback is returned
model rewrites SQL
repeat until pass or max_turns
max_turns falls back to the last SQL candidate
```

There is no required `execute_sql`, `submit_sql`, or `final_answer` action format. The model should output SQL directly.

## Reward

Primary reward:

```text
generated SQL execution result equals gold SQL execution result: 1.0
otherwise: 0.0
```

SQL execution is read-only. The first safety policy allows:

```text
SELECT
WITH ... SELECT
```

and rejects mutating or administrative statements such as:

```text
INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, TRUNCATE,
ATTACH, DETACH, PRAGMA, VACUUM, REINDEX
```

## GRPO Semantics

One GRPO group is multiple independent rollouts for the same Spider sample.

Example:

```text
train_batch_size = 32
rollout.n = 4

one update step:
  32 Spider samples
  each sample produces 4 independent trajectories
  total trajectories = 128
```

Training rhythm:

```text
batch rollout -> reward -> GRPO advantage -> actor update -> next batch
```

The system does not roll out a full epoch before updating.

Tool/environment feedback tokens use `response_mask = 0`; model-generated SQL tokens use `response_mask = 1`.

## VERL Integration

Implemented local contract:

```text
sql_agent_training.train.verl_sql_agent_loop.SqlAgentVerlLoop
registered name: sql_agent
config: configs/verl_sql_agent_loop.yaml
```

Dataset rows for VERL contain:

```text
prompt
agent_name = sql_agent
extra_info:
  uid
  question
  db_id
  schema_prompt
  gold_sql
  sqlite_path
```

Prepare VERL parquet:

```powershell
uv run python scripts/prepare_verl_spider.py --data-dir data/spider --split-file train_spider.json --output artifacts/verl/spider_train_smoke.parquet --limit 2
```

AutoDL one-batch Agent GRPO:

```bash
PYTHONPATH=. uv run python -m sql_agent_training.train.verl_agent_grpo --config configs/agent_grpo.autodl.yaml --run-verl
```

## Pipeline

Local command preview:

```powershell
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.local_dryrun.yaml --print-commands
```

Run one stage:

```powershell
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.yaml --stages sft
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.yaml --stages grpo
```

AutoDL full sequence:

```bash
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.autodl.yaml --stages sft,grpo
```

## Milestones

### Milestone 0: Project Skeleton

Acceptance:

- uv project works.
- pytest runs.
- no runtime import of `agentlightning`.

Status: complete.

### Milestone 1: Spider Data

Acceptance:

- text data exists.
- SQLite databases exist.
- schema cache exists.
- `prepare_spider.py --verify-only` passes.

Status: complete.

### Milestone 2: SFT Baseline

Acceptance:

- SFT JSONL generation works.
- local tiny SFT smoke runs.
- local eval path reports executable rate and execution accuracy.

Status: complete locally.

### Milestone 3: SQL Environment And Reward

Acceptance:

- read-only SQL execution works.
- unsafe SQL is rejected.
- Spider execution reward returns 0/1.

Status: complete.

### Milestone 4: Agent Rollout

Acceptance:

- one Spider sample can run through the SQL rewrite loop.
- max-turn fallback is deterministic.
- trajectory contains prompt tokens, response tokens, response mask, final SQL, and reward metadata.

Status: complete.

### Milestone 5: VERL Agent GRPO

Acceptance:

- current VERL AgentLoop API is resolved.
- `SqlAgentVerlLoop` returns `AgentLoopOutput`.
- VERL parquet format is generated.
- AutoDL one-batch actor update runs.

Status: local contract complete; AutoDL actor update pending.

### Milestone 6: Pipeline And Cleanup

Acceptance:

- `--stages sft` works.
- `--stages grpo` works.
- `--stages sft,grpo` works.
- monitor scripts exist.
- old Agent-Lightning code is removed from active runtime.

Status: complete locally, assuming AutoDL GRPO run passes.

## Verification Commands

```powershell
uv run pytest
uv run python scripts/prepare_spider.py --data-dir data/spider --verify-only
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.local_dryrun.yaml --print-commands
rg "agentlightning|Agent-Lightning|Agent Lightning|agl\." sql_agent_training/sql_agent_training sql_agent_training/tests sql_agent_training/scripts sql_agent_training/configs
```
