# verl GRPO L40S/H100 Debug Notes, 2026-07-27

## Context

Target run:

- Model: `data/models/Qwen2.5-Coder-14B-Instruct`
- Adapter: `artifacts/checkpoints/sft_qwen25_coder_14b_lora_h100_zero2/20260725_061113/checkpoint-300`
- Training path: verl GRPO with async vLLM rollout and custom SQL AgentLoop
- Hardware requested during debugging: 3x L40S on one SLURM node
- Initial smoke target: 2 training steps, tiny rollout settings

The goal of this debugging round was not to tune GRPO quality yet. It was to make the
14B LoRA verl pipeline pass initialization and one tiny smoke run. Later in the same
debugging session, the smoke target moved to 2x H100 because 4x L40S fitting remained
fragile; those H100 findings are included because they exposed several code and
environment issues that apply to both launch paths.

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

### 8. AgentLoop `extra_fields` duplicated verl sample keys

After the H100 run reached rollout, verl failed while merging the original batch with
the generated batch:

```text
AssertionError: `uid` in tensor_dict1 and tensor_dict2 are not the same object.
```

Root cause:

- The custom `SpiderSqlAgentLoop` returned `uid` and `db_id` in `AgentLoopOutput.extra_fields`.
- The original verl batch already carried sample identifiers such as `uid`.
- `batch.union(gen_batch_output)` requires duplicated non-tensor keys to be deeply equal.
  The repeated `uid` value did not pass that identity/equality check.

Resolution:

- Kept `uid` and `db_id` inside `_sample_fields()` for internal request IDs and SQLite context.
- Removed `uid` and `db_id` from `extra_fields`.
- Added `_rollout_extra_fields()` so only rollout-owned diagnostics are returned:
  `final_sql`, `final_sql_source`, `num_execute_calls`, `num_check_calls`, and
  `num_parse_errors`.
- Added a unit test that guards against reintroducing `uid`/`db_id` into rollout
  `extra_fields`.

Relevant commit:

```text
ffab338c Avoid duplicate verl rollout fields
```

### 9. Batch-size relationships surfaced as delayed verl config/runtime errors

Several errors were caused by launch parameters that were individually reasonable but
invalid in combination.

Observed failures:

```text
AssertionError: number of items:[1] < k_partitions:[2]
ValueError: train_batch_size (1) must be >= actor.ppo_mini_batch_size (2)
ValueError: [actor_rollout_ref.rollout] Please set at least one of
  actor_rollout_ref.rollout.log_prob_micro_batch_size
  or actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
AssertionError:
  actor.use_dynamic_bsz == rollout.log_prob_use_dynamic_bsz
```

Root causes and fixes:

- With `trainer.balance_batch=True`, verl balances generated trajectories across data
  parallel ranks. On 2 H100s, `TRAIN_BATCH_SIZE=1` and `ROLLOUT_N=1` created only one
  generated item for two partitions. The H100 smoke default changed to `ROLLOUT_N=2`.
- `PPO_MINI_BATCH_SIZE` is validated against the original prompt batch size, not the
  rollout-expanded trajectory count. Therefore `TRAIN_BATCH_SIZE=1` requires
  `PPO_MINI_BATCH_SIZE=1`, even if `ROLLOUT_N=2`.
- When `LOG_PROB_USE_DYNAMIC_BSZ=False`, verl still requires a fixed
  `log_prob_micro_batch_size_per_gpu`. We defaulted this to
  `PPO_MICRO_BATCH_SIZE_PER_GPU` so smoke users only need to remember one per-GPU
  micro-batch knob.
- verl requires actor dynamic batching and rollout log-prob dynamic batching to match,
  so `actor.use_dynamic_bsz` now follows `LOG_PROB_USE_DYNAMIC_BSZ`.

Resolution:

- Added `sql_agent_training.train.verl_grpo_config`, a lightweight validator that does
  not import torch, verl, vLLM, or load the model.
- Added `DRY_RUN=1` support in the launch script. It validates batch relationships and
  prints the final `python -m verl.trainer.main_ppo ...` command without starting Ray,
  vLLM, GPU monitoring, or model loading.
- Added validation for:
  - `TRAIN_BATCH_SIZE * ROLLOUT_N >= NGPUS_PER_NODE` when `balance_batch=True`
  - `PPO_MINI_BATCH_SIZE <= TRAIN_BATCH_SIZE`
  - `PPO_MICRO_BATCH_SIZE_PER_GPU <= PPO_MINI_BATCH_SIZE`
  - fixed log-prob micro-batch presence when dynamic log-prob batching is disabled
  - `ROLLOUT_TP` divisibility by Qwen's 40 attention heads
  - `ROLLOUT_PP=1` for the installed verl vLLM rollout wrapper

Relevant commits:

```text
38a62407 Fix verl smoke batch sizing
cfef0430 Disable dynamic logprob batching by default
0c9cd9ef Align verl dynamic batch switches
6e2a9040 Fix H100 verl mini batch default
3ea404c3 Set verl logprob micro batch size
0839295a Add verl GRPO launch validation
81de01af Default logprob micro batch to actor micro batch
```

### 10. FlashAttention became a hard dependency of this verl `main_ppo` path

Initially we tried to avoid FlashAttention by setting:

```text
MODEL_USE_REMOVE_PADDING=False
MODEL_ATTN_IMPLEMENTATION=sdpa
LOG_PROB_USE_DYNAMIC_BSZ=False
```

This avoided Transformers' explicit `FlashAttention2 has been toggled on` failure, but
did not remove all FlashAttention usage from verl. Once the run reached old log-prob
calculation, it failed with:

```text
ModuleNotFoundError: No module named 'flash_attn'
```

Stack evidence:

```text
RayPPOTrainer._compute_old_log_prob
  -> left_right_2_no_padding
  -> verl.utils.attention_utils.unpad_input
  -> from flash_attn.bert_padding import ...
```

Conclusion:

- In the installed verl `main_ppo` / `RayPPOTrainer` path, `flash_attn.bert_padding`
  is needed by the padding conversion used during old/reference log-prob and actor
  update stages.
- This is independent of whether Transformers model attention itself uses SDPA.

Resolution:

- Added `PREFLIGHT=1` support: validate config and runtime dependencies, then exit
  before starting Ray/vLLM/model loading.
- Added preflight check for `flash_attn.bert_padding`.
- The check is enabled for real runs and `PREFLIGHT=1`, but disabled for `DRY_RUN=1`
  so local command-shape validation still works without GPU packages.

Relevant commit:

```text
7c6042de Preflight verl flash attention dependency
```

### 11. FlashAttention install failed until Python/CUDA/PyTorch were aligned

The first `flash-attn` installation attempt failed because the existing environment was:

```text
torch: 2.9.0+cu128
torch.version.cuda: 12.8
detected nvcc/CUDA toolkit: 13.2
```

The build error:

```text
RuntimeError:
The detected CUDA version (13.2) mismatches the version that was used to compile PyTorch (12.8).
```

The cluster had CUDA modules for `12.6.1`, `12.9.1`, and `13.0.1`, but not `12.8`.
Therefore compiling `flash-attn` against the existing `torch+cu128` environment was not
viable.

Resolution path:

- Created a fresh outer `.venv` instead of mutating the old environment in place.
- Loaded `cuda/12.6.1`.
- Installed a CUDA 12.6-aligned torch/vLLM/verl stack:

```text
python: 3.12.13
torch: 2.9.0+cu126
torch.version.cuda: 12.6
vllm: 0.12.0
```

- Removed the inner project `.venv` so `uv run` consistently uses the outer
  workspace environment.
- Updated both `.python-version` files from `3.10` to `3.12` on the server to avoid
  uv selecting/requesting Python 3.10 by mistake.
- Installed `flash-attn` into the outer environment after CUDA/PyTorch alignment.

Important lesson:

- Do not only replace torch inside an existing verl/vLLM environment. vLLM, torch,
  flash-attn, and CUDA should be treated as one binary stack.
- Manual `uv pip install flash-attn` is acceptable as an experiment, but `uv sync`
  can later remove it because it is not locked in `pyproject.toml`/`uv.lock`.

### 12. PEFT and Transformers became the next binary-stack compatibility issue

After CUDA/FlashAttention was fixed, the run progressed into model initialization and
LoRA adapter loading, then failed with:

```text
ImportError: cannot import name 'EmbeddingParallel'
from transformers.integrations.tensor_parallel
```

Stack evidence:

```text
PeftModel.from_pretrained(...)
  -> set_peft_model_state_dict(...)
  -> _maybe_shard_state_dict_for_tp(...)
  -> from transformers.integrations.tensor_parallel import EmbeddingParallel
```

Root cause:

- The installed `peft` version expects a Transformers tensor-parallel helper symbol
  named `EmbeddingParallel`.
- The installed `transformers` package does not provide that symbol.
- This is a PEFT/Transformers API compatibility mismatch surfaced while loading the
  LoRA adapter.

Resolution added so far:

- Added a preflight check that inspects whether the installed PEFT code references
  `EmbeddingParallel` and whether the installed Transformers package provides it.
- This catches the mismatch before model loading.

Likely next action:

- Prefer adjusting PEFT before upgrading Transformers, because vLLM is sensitive to
  Transformers version changes.
- First try:

```bash
uv pip install "peft==0.17.1"
```

- Then verify:

```bash
uv run --no-sync python - <<'PY'
import peft, transformers
print("peft:", peft.__version__)
print("transformers:", transformers.__version__)
try:
    from transformers.integrations.tensor_parallel import EmbeddingParallel
    print("EmbeddingParallel ok")
except Exception as exc:
    print("EmbeddingParallel failed:", repr(exc))
PY
```

Relevant commit:

```text
43f12c41 Preflight PEFT transformers compatibility
```

## Code And Parameter Changes Made

Main script:

```text
sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_l40s_4gpu.sh
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
  - `RAY_NUM_CPUS=16`
  - `RAY_OBJECT_STORE_MEMORY=1073741824`
  - `RAY_INCLUDE_DASHBOARD=False`
  - `DATALOADER_NUM_WORKERS=0`
  - `FILTER_OVERLONG_PROMPTS_WORKERS=1`
  - `REWARD_NUM_WORKERS=1`
  - `ACTOR_USE_TORCH_COMPILE=False`
  - `ROLLOUT_ENFORCE_EAGER=True`
  - `MODEL_USE_REMOVE_PADDING=False`
  - `MODEL_ATTN_IMPLEMENTATION=sdpa`
  - `DATA_TRUST_REMOTE_CODE=False`
  - `MODEL_TRUST_REMOTE_CODE=False`
  - `ROLLOUT_TP=4`
  - `ROLLOUT_PP=1`
  - `ROLLOUT_GPU_MEMORY_UTILIZATION=0.32`
  - `ROLLOUT_MAX_MODEL_LEN=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH`
  - `ROLLOUT_MAX_NUM_BATCHED_TOKENS=ROLLOUT_MAX_MODEL_LEN`
  - `ROLLOUT_MAX_NUM_SEQS=1`
  - `USE_KL_IN_REWARD=False`
  - `USE_KL_LOSS=False`
  - `ACTOR_PARAM_OFFLOAD=False`
  - `ACTOR_OPTIMIZER_OFFLOAD=False`
  - `REF_PARAM_OFFLOAD=False`
  - `LOG_PROB_USE_DYNAMIC_BSZ=False`
  - `LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=PPO_MICRO_BATCH_SIZE_PER_GPU`
- Limit CPU thread fan-out:
  - `TOKENIZERS_PARALLELISM=false`
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `NUMEXPR_NUM_THREADS=1`
  - `MALLOC_ARENA_MAX=2`
  - `CUDA_MODULE_LOADING=LAZY`
  - `VLLM_WORKER_MULTIPROC_METHOD=spawn`

Relevant commits from this debugging chain:

```text
a1ef675b Constrain Ray resources for verl smoke
9e6dc58d Advertise enough Ray CPUs for verl workers
5983ccbf Leave CPU headroom for verl placement group
00a11c9c Reduce verl smoke memory pressure
ffab338c Avoid duplicate verl rollout fields
0839295a Add verl GRPO launch validation
7c6042de Preflight verl flash attention dependency
43f12c41 Preflight PEFT transformers compatibility
```

## Correct SLURM Allocation

Do not launch this job with only `--gres=gpu:l40s:3`. That gives GPUs but does not
request enough CPU or memory.

For the later 4x L40S `ROLLOUT_TP=4` path, use an allocation like:

```bash
srun \
  --partition=ice-gpu \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=128G \
  --gres=gpu:l40s:4 \
  --time=05:00:00 \
  --pty bash
```

If the cluster requires a different GPU resource string, keep that part from the
working command but add:

```bash
--cpus-per-task=16 --mem=128G
```

For exactly 3x L40S, the current colocated verl vLLM path has no good model-parallel
shape for Qwen2.5-Coder-14B: `TP=3` does not divide 40 attention heads, and `PP=3`
is not implemented by the installed verl rollout wrapper.

## Recommended Smoke Command

After entering a properly sized allocation:

```bash
cd /home/hice1/hzhang961/scratch/sql_agent_training
git pull
uv run --no-sync ray stop -f

mkdir -p logs
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=0,1,2,3
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
  ROLLOUT_N=4 \
  ROLLOUT_TP=4 \
  ROLLOUT_PP=1 \
  ROLLOUT_GPU_MEMORY_UTILIZATION=0.32 \
  ROLLOUT_MAX_NUM_SEQS=1 \
  USE_KL_IN_REWARD=False \
  USE_KL_LOSS=False \
  ACTOR_PARAM_OFFLOAD=False \
  ACTOR_OPTIMIZER_OFFLOAD=False \
  REF_PARAM_OFFLOAD=False \
  MAX_PROMPT_LENGTH=2048 \
  MAX_RESPONSE_LENGTH=512 \
  uv run --no-sync bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_l40s_4gpu.sh

  status=$?
  echo "EXIT_CODE $status $(date)"
) > logs/verl_grpo_smoke.log 2>&1 &

tail -f logs/verl_grpo_smoke.log
```

Expected startup lines:

```text
verl RAY_NUM_CPUS=16 RAY_OBJECT_STORE_MEMORY=1073741824
verl TRAIN_BATCH_SIZE=1 PPO_MINI_BATCH_SIZE=1 ROLLOUT_N=4
verl ROLLOUT_TP=4 ROLLOUT_PP=1 ROLLOUT_GPU_MEMORY_UTILIZATION=0.32
verl USE_KL_IN_REWARD=False USE_KL_LOSS=False KL_LOSS_COEF=0.01
verl ACTOR_USE_TORCH_COMPILE=False ROLLOUT_ENFORCE_EAGER=True
verl ACTOR_PARAM_OFFLOAD=False ACTOR_OPTIMIZER_OFFLOAD=False REF_PARAM_OFFLOAD=False
```

For a 4x L40S balanced smoke, `ROLLOUT_N=1` is now known to be invalid when
`TRAIN_BATCH_SIZE=1` and `trainer.balance_batch=True`, because it creates only one
trajectory for four GPU partitions. Use either:

```bash
TRAIN_BATCH_SIZE=1
PPO_MINI_BATCH_SIZE=1
ROLLOUT_N=4
```

or:

```bash
TRAIN_BATCH_SIZE=4
PPO_MINI_BATCH_SIZE=4
ROLLOUT_N=1
```

For GRPO quality experiments, prefer multiple rollouts per prompt (`ROLLOUT_N > 1`) so
each prompt has an actual comparison group.

Before starting a GPU run, use one of the launch-script checks:

```bash
# Validate config shape and print the final verl command. Does not require flash-attn.
DRY_RUN=1 \
CUDA_VISIBLE_DEVICES=0,1 \
uv run --no-sync bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_2gpu.sh

# Validate config and runtime imports such as flash_attn/PEFT/Transformers, then exit.
PREFLIGHT=1 \
CUDA_VISIBLE_DEVICES=0,1 \
uv run --no-sync bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_h100_2gpu.sh
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

1. Re-request the 4x L40S allocation with adequate CPU and memory.
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
6. Re-enable `MODEL_USE_REMOVE_PADDING=True` or use `MODEL_ATTN_IMPLEMENTATION=flash_attention_2`
   only after installing a compatible `flash_attn` package. Without `flash_attn`, Transformers
   raises `FlashAttention2 has been toggled on`. In Hydra overrides, `_attn_implementation`
   must be appended under `override_config` with a leading `+` because the key is not present
   in verl's default config.
7. Keep `MODEL_TRUST_REMOTE_CODE=False` for Qwen2.5-Coder because it is supported by the installed
   Transformers/vLLM stack and does not require remote model code. If a future model requires custom
   code, set `MODEL_TRUST_REMOTE_CODE=True` and `DATA_TRUST_REMOTE_CODE=True` explicitly.
8. If vLLM `EngineCore` fails during startup after loading roughly half of each L40S, first check the
   worker logs for the root cause. In one failing run, the root cause was:
   `Free memory on device (16.17/44.39 GiB) on startup is less than desired GPU memory utilization (0.6, 26.64 GiB)`.
   vLLM's utilization target is a fraction of total device memory, not a fraction of currently free memory.
9. A follow-up attempt with `ROLLOUT_TP=3` failed because Qwen2.5-Coder-14B has 40 attention heads, and vLLM requires
   the head count to be divisible by the tensor parallel size:
   `Total number of attention heads (40) must be divisible by tensor parallel size (3)`.
10. A follow-up attempt with `ROLLOUT_PP=3` also failed in the installed verl path:
    `Current rollout self.name='vllm' not implemented pipeline_model_parallel_size > 1 yet.`
11. For 14B on exactly 3x L40S with this verl vLLM rollout path, there is no valid 3-way model-parallel rollout:
    `TP=3` is invalid for 40 heads, and `PP=3` is not implemented by verl's vLLM rollout wrapper. The remaining
    smoke fallback is `ROLLOUT_TP=1`, `ROLLOUT_PP=1`, PyTorch/FSDP offload, and very small rollout concurrency.
    If that still cannot fit, use more GPUs with a legal TP size such as 4, use 2 H100 with TP=2, or move rollout
    to separate vLLM GPUs instead of colocating learner and rollout on the same 3 L40S cards.
12. After acquiring 4x L40S, use `ROLLOUT_TP=4`, `ROLLOUT_PP=1`, and no FSDP offload. `TP=4` is valid for
    Qwen2.5-Coder-14B because 40 attention heads is divisible by 4.
13. If `vLLM wake_up(tags=["weights"])` OOMs while colocated with actor/reference, disable reference/KL first:
    `USE_KL_LOSS=False` and `USE_KL_IN_REWARD=False`. This removes the reference FSDP worker from the smoke path.

## 2026-07-27 H100 Smoke Result And LoRA Loading Fix

The 2x H100 no-reference smoke completed successfully:

```text
Training Progress: 100%|...| 2/2
verl RUN_WALL_TIME_SEC=298
EXIT_CODE 0 Mon Jul 27 14:12:04 EDT 2026
```

Important metrics from the two training steps:

```text
step 1: step=27.56s, gen=20.25s, old_log_prob=1.71s, update_actor=2.27s, update_weights=3.33s
step 2: step=7.27s,  gen=1.98s,  old_log_prob=0.52s, update_actor=1.63s, update_weights=3.14s
```

The smoke validated the full path:

```text
verl main_ppo -> vLLM rollout -> custom SQL AgentLoop -> SQLite reward -> old logprob -> actor update -> weight sync
```

However, both steps had reward `1.0`, advantage `0.0`, actor loss `0.0`, and gradient norm `0.0`. That is acceptable
for a smoke test, but it is not a useful learning signal. Real GRPO pilots need harder samples and/or larger
`ROLLOUT_N` so each prompt has within-group reward variance.

The smoke still showed a serious adapter-loading warning:

```text
copying from a non-meta parameter in the checkpoint to a meta parameter in the current model, which is a no-op
```

This happens while verl/FSDP initializes the actor with meta tensors and PEFT loads the existing SFT LoRA adapter.
The run can continue, but the warning means the SFT adapter checkpoint may not actually be assigned into the actor
weights. The safer path is:

1. Merge the SFT LoRA checkpoint into the 14B base model once.
2. Use that merged model as `MODEL_PATH`.
3. Set `LORA_ADAPTER_PATH=none`.
4. Let verl initialize a fresh trainable GRPO LoRA on top of the merged SFT model.

The repository now includes:

```text
scripts/merge_lora_adapter.py
```

Run the one-time merge from the outer server workspace root:

```bash
cd /home/hice1/hzhang961/scratch/sql_agent_training

uv run --no-sync python sql_agent_training/scripts/merge_lora_adapter.py \
  --base-model sql_agent_training/data/models/Qwen2.5-Coder-14B-Instruct \
  --adapter sql_agent_training/artifacts/checkpoints/sft_qwen25_coder_14b_lora_h100_zero2/20260725_061113/checkpoint-300 \
  --output-dir sql_agent_training/data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged \
  --dtype bfloat16 \
  --device-map auto
```

The H100 wrapper now defaults to:

```text
MODEL_PATH=data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged
LORA_ADAPTER_PATH=none
```

so the next H100 smoke should no longer route the old SFT LoRA adapter through the PEFT/FSDP meta-tensor load path.

## 2026-07-28 H100 GRPO Environment Stabilization

The 4x H100 GRPO run exposed a separate class of failures: the training recipe itself was standard for verl
AgentLoop GRPO, but the exact `verl + vLLM + torch + flash-attn + CUTLASS + tensordict` matrix had to be pinned
tightly. The final environment that passed import checks and `uv pip check` was:

```text
torch: 2.9.0
torch cuda: 12.8
torchvision: 0.24.0
torchaudio: 2.9.0
vllm: 0.12.0
verl: 0.9.0.dev0
verl git commit: f663282327d784068263c7c3736a4884830eea44
flash_attn: 2.8.3
numpy: 2.2.6
setuptools: 80.9.0
fsspec: 2026.2.0
tensordict: 0.10.0
nvidia-cutlass-dsl: 4.5.2
nvidia-cutlass-dsl-libs-base: 4.5.2
uv pip check: All installed packages are compatible
```

This is now recorded in `sql_agent_training/pyproject.toml` as the `verl-cu128` optional extra. The name is `cu128`
because the server environment reports `torch.version.cuda == 12.8`; the earlier `verl-cu126` label was misleading.
The `flash-attn` dependency is recorded as the official GitHub release wheel:

```text
flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

This avoids the PyPI source build path.

### Failure Chain

1. `verl==0.8.0` could run the GRPO smoke path, but its `CheckpointConfig` did not support
   `actor_rollout_ref.actor.checkpoint.save_lora_only`.
2. Upgrading only `verl` to GitHub commit `f663282327d784068263c7c3736a4884830eea44` made `save_lora_only`
   available, but the default trainer path changed to V1:

```text
TaskRunnerV1 -> import transfer_queue as tq
ModuleNotFoundError: No module named 'transfer_queue'
```

The launch script now passes:

```text
++trainer.use_v1=False
```

through `TRAINER_USE_V1=False`, keeping the run on the legacy RayPPOTrainer path that previously passed smoke.

3. `flash-attn 2.8.3` plus a newer `nvidia-cutlass-dsl` failed during vLLM import:

```text
AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'
```

The compatible stack pins:

```text
nvidia-cutlass-dsl==4.5.2
nvidia-cutlass-dsl-libs-base==4.5.2
```

and removes the incompatible leftover `nvidia-cutlass-dsl-libs-cu12==4.6.1`. It is OK for
`nvidia-cutlass-dsl-libs-cu12` to be absent in the 4.5.2 stack; `uv pip check` is the authority.

4. `tensordict` was another narrow constraint. `verl` code asserted that `DataProto.to_tensordict()` requires
`tensordict >= 0.10`, while package metadata rejected newer versions:

```text
verl requires tensordict>=0.8.0,!=0.9.0,<=0.10.0
```

Therefore the working pin is:

```text
tensordict==0.10.0
```

5. An attempted resolver repair temporarily made the environment inconsistent:

```text
torch: 2.13.0
numpy: 2.5.1
setuptools: 83.0.0
fsspec: 2026.6.0
```

This broke `vllm`, `torchaudio`, `torchvision`, `mistral-common`, `numba`, and `datasets` constraints. The repaired
pins are:

```text
torch==2.9.0
torchvision==0.24.0
torchaudio==2.9.0
numpy==2.2.6
setuptools==80.9.0
fsspec[http]==2026.2.0
```

6. A 4x H100 run with `--mem=196G` died in `checkpoint_manager.update_weights()`, but the confirmed root cause was
CPU/RAM cgroup OOM, not GPU OOM:

```text
Memory cgroup out of memory: Killed process 894711 (ray::WorkerDict)
anon-rss:34272040kB
```

The NCCL broken pipes and `ActorDiedError` were secondary effects after the Ray worker was killed. The next 4x H100
run should request substantially more host memory, e.g. `--mem=384G` or `--mem=512G`.

### Preflight Checks That Avoid Loading 14B

Before launching the full model, use lightweight import and dependency checks:

```bash
uv run --no-sync python - <<'PY'
import dataclasses
import importlib.metadata as md
from packaging.version import parse

for p in [
    "torch", "torchvision", "torchaudio", "vllm", "verl", "flash_attn",
    "numpy", "setuptools", "fsspec", "tensordict",
    "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base",
]:
    print(f"{p}: {md.version(p)}")

import torch
print("torch cuda:", torch.version.cuda, "available:", torch.cuda.is_available())

import tensordict
assert parse(tensordict.__version__) == parse("0.10.0")

from verl.trainer.config.config import CheckpointConfig
fields = [field.name for field in dataclasses.fields(CheckpointConfig)]
assert "save_lora_only" in fields, fields

import cutlass.cute as cute
assert hasattr(cute.core, "ThrMma")

import vllm.vllm_flash_attn.flash_attn_interface
from verl.protocol import DataProto
print("verl/vLLM preflight ok")
PY

uv pip check
```

Then run the script preflight:

```bash
PREFLIGHT=1 ACTOR_CHECKPOINT_SAVE_LORA_ONLY=True \
  uv run --no-sync bash sql_agent_training/scripts/run_verl_grpo_qwen25_coder_14b_l40s_4gpu.sh
```

Only after both checks pass should the full 14B run start.

### Current Working Training Shape

The active 4x H100 run uses:

```text
TRAIN_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=8
ROLLOUT_N=4
ROLLOUT_TP=4
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=2048
MAX_TURNS=3
TRAINER_USE_V1=False
ACTOR_CHECKPOINT_SAVE_LORA_ONLY=True
SAVE_FREQ=25
```

If this still fails on host memory despite `--mem=384G+`, reduce rollout pressure before changing package versions:

```text
TRAIN_BATCH_SIZE=4
PPO_MINI_BATCH_SIZE=4
ROLLOUT_MAX_NUM_SEQS=4
```

and increase total steps to preserve the approximate number of sampled trajectories.
