#!/usr/bin/env bash
set -euo pipefail

python -V
nvidia-smi

python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
except Exception as exc:
    print("torch probe failed:", exc)
PY

uv sync --group dev --extra train
uv run pytest

uv run python scripts/probe_verl_agent_loop.py || true
