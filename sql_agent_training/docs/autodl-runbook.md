# AutoDL Runbook

This runbook is the target workflow for real SFT and Agent GRPO training.

## 1. Create Instance

Choose a GPU instance with enough memory for Qwen Coder 0.5B, vLLM/SGLang rollout, and VERL training.
The local RTX 3070 workflow is only for dry runs and tests.

## 2. Clone And Install

```bash
git clone <repo-url>
cd agent-lightning/sql_agent_training
uv sync --extra train --group dev
```

Quick bootstrap:

```bash
bash scripts/autodl_bootstrap.sh
```

Probe VERL Agent Loop API:

```bash
python scripts/probe_verl_agent_loop.py
```

## 3. Prepare Spider

Place official Spider assets under:

```text
/root/autodl-tmp/data/spider/
  tables.json
  database/
    {db_id}/
      {db_id}.sqlite
```

Verify:

```bash
uv run python scripts/prepare_spider.py --data-dir /root/autodl-tmp/data/spider --verify-only
```

## 4. Run SFT

```bash
uv run python -m sql_agent_training.train.sft --config configs/sft.autodl.yaml
```

## 5. Run GRPO Smoke Test

Prepare VERL parquet and print the exact VERL command:

```bash
uv run python -m sql_agent_training.train.verl_agent_grpo --config configs/agent_grpo.autodl.yaml --prepare-only
```

Probe the installed VERL Agent Loop API:

```bash
uv run python scripts/probe_verl_agent_loop.py
```

See `docs/verl-agent-loop-spike.md` for the one-batch command template.

```bash
PYTHONPATH=. uv run python -m sql_agent_training.train.verl_agent_grpo --config configs/agent_grpo.autodl.yaml --run-verl
```

## 6. Run Pipeline

Run only SFT:

```bash
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.autodl.yaml --stages sft
```

Run only Agent GRPO:

```bash
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.autodl.yaml --stages grpo
```

Run SFT followed by Agent GRPO:

```bash
uv run python -m sql_agent_training.train.run_pipeline --config configs/pipeline.autodl.yaml --stages sft,grpo
```

## 7. Collect Artifacts

Download:

```text
/root/autodl-tmp/checkpoints/
/root/autodl-tmp/logs/
```
