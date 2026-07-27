# verl GRPO 3x L40S Debug Notes, 2026-07-27

## Context

Target run:

- Model: `data/models/Qwen2.5-Coder-14B-Instruct`
- Adapter: `artifacts/checkpoints/sft_qwen25_coder_14b_lora_h100_zero2/20260725_061113/checkpoint-300`
- Training path: verl GRPO with async vLLM rollout and custom SQL AgentLoop
- Hardware requested during debugging: 3x L40S on one SLURM node
- Initial smoke target: 2 training steps, tiny rollout settings

The goal of this debugging round was not to tune GRPO quality yet. It was to make the
14B LoRA verl pipeline pass initialization and one tiny smoke run.

## Problem Chain

### 1. Ray tried to package the whole project

Early verl runs passed a Ray `runtime_env.working_dir`. Ray then attempted to package
the local project tree and encountered large files under `data/`, including models and
Spider SQLite databases.

Symptoms included Ray packaging messages about large files and slow startup.

Resolution:

- Defaulted the verl script to not pass Ray `runtime_env`.
- Kept `ENABLE_RAY_RUNTIME_ENV=1` as an opt-in path only.
- Set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` so Ray's `uv run` hook does not rewrite the
  runtime environment and package the project unexpectedly.

### 2. Dashboard-dependent Ray commands became unavailable

We disabled the Ray dashboard to reduce startup and memory pressure. After that,
`ray list actors --detail` failed because it talks to the dashboard/state API at
`127.0.0.1:8265`.

This was not a training failure.

Resolution:

- Use `ray status` for coarse resource status.
- Use `tail -f logs/verl_grpo_smoke.log` for main training logs.
- Use `nvidia-smi` sparingly for GPU process checks.
- Use `/tmp/$USER/ray/ray/session_latest/logs` only when the main log is insufficient.

### 3. LoRA config looked missing, but the wrong config block was being read

The TaskRunner config dump showed blocks like:

```text
'adapter_path': None
'lora_rank': 0
'path': '~/models/deepseek-llm-7b-chat'
```

This caused concern that the Qwen model and LoRA checkpoint were not being used.

What we found:

- Those values came from disabled/default config sections such as critic, distillation,
  or reward-model defaults.
- GRPO disables the critic path because `algorithm.adv_estimator=grpo`, and the logs
  showed `Disabled critic as algorithm.adv_estimator != gae`.
- The actual actor policy block was correct.

Expected actor policy values in the config dump:

```text
actor_rollout_ref.model.path=.../data/models/Qwen2.5-Coder-14B-Instruct
actor_rollout_ref.model.lora_adapter_path=.../checkpoint-300
actor_rollout_ref.model.lora_rank=64
actor_rollout_ref.model.lora_alpha=128
actor_rollout_ref.model.target_modules=all-linear
```

Conclusion:

- The Qwen 14B model path and LoRA checkpoint were passed correctly.
- The `adapter_path=None` lines were not the active actor policy.

### 4. Ray placement group could not schedule with too few CPU slots

After reducing Ray CPU count too aggressively, Ray reported:

```text
Pending Demands:
{'GPU': 1.0, 'CPU': 3.0} * 3 (STRICT_PACK)
```

With `RAY_NUM_CPUS=4`, the 3 GPU workers could not be scheduled.

Then with `RAY_NUM_CPUS=9`, the placement group still had too little headroom because
the TaskRunner itself occupied 1 CPU slot.

Resolution:

- Set `RAY_NUM_CPUS=12`.
- This gives Ray enough schedulable CPU slots for:
  - 3 GPU workers x 3 CPU each = 9 CPU
  - TaskRunner and driver overhead

Important distinction:

- This is Ray's schedulable CPU resource count.
- It does not mean the run is CPU-heavy by design, but Ray refuses to launch worker
  placement groups if the declared resource budget is too small.

### 5. ActorDiedError was a symptom, not the root cause

Repeated failures ended with:

```text
ray.exceptions.ActorDiedError
Worker unexpectedly exits with a connection error code 2. End of file.
```

This Ray error was generic. It did not identify the real cause.

The decisive evidence came from `dmesg`:

```text
Memory cgroup out of memory: Killed process ... (ray::TaskRunner)
```

So the TaskRunner was being killed by the kernel/SLURM memory cgroup.

This was not:

- CUDA OOM
- LoRA not loading
- Qwen model path missing
- A Python exception in our AgentLoop

It was CPU memory cgroup OOM.

### 6. Reducing Ray object store helped but did not solve the final allocation issue

Ray initially advertised a very large object store, around 186 GiB. Under this HPC
memory-cgroup setup, that contributed to huge virtual memory pressure.

Changes made:

- First reduced Ray object store to 8 GiB.
- Then reduced it to 1 GiB for the smoke path.

Observed effect:

- TaskRunner `total-vm` dropped from roughly 220 GiB to roughly 33 GiB.
- This showed the change was in the right direction.
- But the process was still killed because the SLURM job itself only had 4 GiB CPU memory.

### 7. Final root cause: the SLURM allocation had 3 GPUs but only 1 CPU and 4 GiB RAM

The decisive `scontrol show job` output:

```text
NumCPUs=1
CPUs/Task=1
ReqTRES=cpu=1,mem=4G,node=1,gres/gpu=3,gres/gpu:l40s=3
AllocTRES=cpu=1,mem=4G,node=1,gres/gpu=3,gres/gpu:l40s=3
SubmitLine=srun --partition=ice-gpu --nodes=1 --ntasks=1 --gres=gpu:l40s:3 --time=05:00:00 --pty bash
```

This allocation is not enough for verl/Ray. The run had 3 L40S GPUs but only:

- 1 CPU
- 4 GiB CPU memory

That is why Ray/verl died before meaningful GPU training began.

## Code And Parameter Changes Made

Main script:

```text
sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_l40s_3gpu.sh
```

Key changes:

- Resolve paths from the inner project root so the script can be launched from either
  the workspace root or inner project root.
- Do not pass Ray `runtime_env` by default.
- Export `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`.
- Add startup config logging:
  - project directory
  - Ray CPU count
  - Ray object store size
  - batch/rollout sizes
  - prompt/response lengths
  - torch compile / eager rollout switches
- Set conservative smoke defaults:
  - `RAY_NUM_CPUS=12`
  - `RAY_OBJECT_STORE_MEMORY=1073741824`
  - `RAY_INCLUDE_DASHBOARD=False`
  - `DATALOADER_NUM_WORKERS=0`
  - `FILTER_OVERLONG_PROMPTS_WORKERS=1`
  - `REWARD_NUM_WORKERS=1`
  - `ACTOR_USE_TORCH_COMPILE=False`
  - `ROLLOUT_ENFORCE_EAGER=True`
- Limit CPU thread fan-out:
  - `TOKENIZERS_PARALLELISM=false`
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `NUMEXPR_NUM_THREADS=1`
  - `MALLOC_ARENA_MAX=2`
  - `CUDA_MODULE_LOADING=LAZY`

Relevant commits from this debugging chain:

```text
a1ef675b Constrain Ray resources for verl smoke
9e6dc58d Advertise enough Ray CPUs for verl workers
5983ccbf Leave CPU headroom for verl placement group
00a11c9c Reduce verl smoke memory pressure
```

## Correct SLURM Allocation

Do not launch this job with only `--gres=gpu:l40s:3`. That gives GPUs but does not
request enough CPU or memory.

Use an allocation like:

```bash
srun \
  --partition=ice-gpu \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=128G \
  --gres=gpu:l40s:3 \
  --time=05:00:00 \
  --pty bash
```

If the cluster requires a different GPU resource string, keep that part from the
working command but add:

```bash
--cpus-per-task=16 --mem=128G
```

## Recommended Smoke Command

After entering a properly sized allocation:

```bash
cd /storage/ice1/0/8/hzhang961/sql_agent_training
git pull
uv run --no-sync ray stop -f

mkdir -p logs
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=0,1,2
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp/$USER}/triton_cache"
export RAY_TMPDIR="${SLURM_TMPDIR:-/tmp/$USER}/ray"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
mkdir -p "$TRITON_CACHE_DIR" "$RAY_TMPDIR"

(
  set -o pipefail
  echo "START verl grpo smoke $(date)"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  PYTHONUNBUFFERED=1 \
  VLLM_USE_V1=1 \
  TOTAL_TRAINING_STEPS=2 \
  SAVE_FREQ=-1 \
  TEST_FREQ=-1 \
  TRAIN_BATCH_SIZE=1 \
  PPO_MINI_BATCH_SIZE=1 \
  ROLLOUT_N=1 \
  MAX_RESPONSE_LENGTH=512 \
  uv run --no-sync bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_l40s_3gpu.sh

  status=$?
  echo "EXIT_CODE $status $(date)"
) > logs/verl_grpo_smoke.log 2>&1 &

tail -f logs/verl_grpo_smoke.log
```

Expected startup lines:

```text
verl RAY_NUM_CPUS=12 RAY_OBJECT_STORE_MEMORY=1073741824
verl TRAIN_BATCH_SIZE=1 PPO_MINI_BATCH_SIZE=1 ROLLOUT_N=1
verl ACTOR_USE_TORCH_COMPILE=False ROLLOUT_ENFORCE_EAGER=True
```

## Monitoring And Diagnosis Checklist

Light monitoring:

```bash
tail -f logs/verl_grpo_smoke.log
nvidia-smi
```

Ray scheduling check:

```bash
uv run --no-sync ray status
```

Do not rely on this when dashboard is disabled:

```bash
uv run --no-sync ray list actors --detail
```

It may fail because the dashboard/state API is disabled.

If Ray reports `ActorDiedError`, check kernel OOM first:

```bash
dmesg -T | grep -Ei "killed process|oom|out of memory|ray|python" | tail -120
```

Check SLURM resources:

```bash
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
scontrol show job "$SLURM_JOB_ID" | grep -Ei "mem|tres|cpu|gpu"
```

Interpretation:

- `ActorDiedError` plus `dmesg` OOM means the worker was killed by the system.
- `No available node types can fulfill resource request` means Ray schedulable
  resources are too small.
- `adapter_path=None` in disabled/default critic or reward blocks is not evidence
  that the actor policy LoRA failed to load.

## Next Steps

1. Re-request the 3x L40S allocation with adequate CPU and memory.
2. Re-run the tiny 2-step smoke.
3. Only after the smoke completes with `EXIT_CODE 0`, increase to real GRPO group
   settings such as:

```bash
TRAIN_BATCH_SIZE=4
PPO_MINI_BATCH_SIZE=4
ROLLOUT_N=4
MAX_RESPONSE_LENGTH=2048
```

4. Keep `RAY_OBJECT_STORE_MEMORY=1073741824` initially. Increase only if Ray object
   store warnings or object spilling become the bottleneck.
5. Re-enable `ACTOR_USE_TORCH_COMPILE=True` or disable `ROLLOUT_ENFORCE_EAGER` only
   after the run is stable and memory headroom is confirmed.
