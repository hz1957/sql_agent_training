# Training Experiment Plan

This document separates experiments for Qwen2.5-Coder 14B LoRA training into SFT, GRPO, pipeline, and evaluation tracks. The main principle is that not every run should log every metric. Each experiment family should log the smallest metric set needed to answer its research question.

## High-Level Design

The experiment plan has four layers:

```text
Layer 1: SFT quality baseline
  Find a strong, stable supervised checkpoint for GRPO to start from.

Layer 2: SFT systems benchmark
  Measure pure training memory and tokens/sec for ZeRO/FSDP/checkpointing/bf16/FlashAttention.

Layer 3: GRPO algorithm benchmark
  Measure reward design, KL control, rollout sampling, advantage normalization, and RL stability.

Layer 4: GRPO pipeline benchmark
  Measure rollout/reference/learner/LoRA-reload bottlenecks in the Ray + vLLM setup.
```

Use SFT for systems questions because its loop is mostly forward/backward/update. Use GRPO for RL questions because its loop includes rollout, SQLite reward, reference logprobs, KL control, and LoRA weight reload.

Do not use one large mixed experiment to answer all questions. Systems settings can change throughput, memory pressure, numerical behavior, and sometimes final accuracy; reward/KL conclusions should be made with the systems stack held fixed.

## Current Baselines

SFT 14B LoRA baseline examples:

```text
configs/sft.qwen25_coder_14b_lora.h100_zero2.yaml
configs/sft.qwen25_coder_14b_lora.h100_zero3.yaml
configs/sft.qwen25_coder_14b_lora.l40s_zero2.yaml
configs/sft.qwen25_coder_14b_lora.l40s_zero3.yaml
```

GRPO Ray 14B LoRA baseline:

```text
configs/grpo.ray_14b_lora.l40s_3gpu.yaml
```

GRPO placement:

```text
GPU0: RolloutActor / vLLM / SQLite rollout-reward loop
GPU1: ReferenceLogprobWorker
GPU2: LearnerWorker / policy LoRA update
```

GRPO baseline parameters:

```yaml
rollout:
  n: 4
  max_turns: 3
  max_prompt_length: 4096
  max_response_length: 1024

training:
  max_steps: 500
  task_batch_size: 4
  update_epochs: 1
  learning_rate: 0.0000005
  kl_beta: 0.01
  clip_epsilon: 0.2
  normalize_advantages: true
  transition_reward_mode: discounted_final
  transition_reward_gamma: 0.4
```

## Global Rules

1. Change one experimental axis at a time.
2. Use smoke runs before expensive runs.
3. Keep the evaluation set fixed within one comparison table.
4. Keep the SFT starting checkpoint fixed for GRPO algorithm comparisons.
5. Keep the systems stack fixed for reward/KL comparisons.
6. Record the git commit, config path, checkpoint path, seed, GPU type, and command for every main run.

Recommended run tiers:

```text
smoke:       5-20 steps, catches OOM and broken configs.
pilot:       50-100 steps, checks learning signal and stability.
main:        300-500 steps, used for comparison.
confirm:     rerun best settings with another seed.
```

## Run Naming

Use descriptive checkpoint roots:

```yaml
output:
  checkpoint_dir: artifacts/checkpoints/<track>/<experiment_slug>
```

Suggested slugs:

```text
sft_sys_l40s_zero2_bf16_ac_on_seq2560
sft_quality_lr2e5_epoch1_lora64
grpo_reward_discounted_gamma04_kl001
grpo_kl_beta003_clip02
grpo_pipe_vllm_len1024_rolloutn4
```

Each run should be traceable from:

```text
run_config.yaml
metrics.jsonl or trainer_state.json
stdout/stderr log
eval_metrics.json
git commit
```

## Track A: SFT Quality Baseline

Primary question:

```text
Which SFT checkpoint is the best starting policy for GRPO?
```

This track should answer supervised quality questions, not systems throughput questions.

Candidate knobs:

```text
learning_rate: 1e-5, 2e-5, 5e-5
num_train_epochs: 1, 2, 3
lora_r: 32, 64, 128
lora_alpha: paired with r, e.g. alpha = 2r
weight_decay: 0.0, 0.01
warmup_ratio: 0.03, 0.05
effective_batch_size
max_prompt_length
max_response_length
```

Hold fixed:

```text
model path
dataset split
prompt template
eval subset
systems stack after it is chosen
```

Required logs:

```text
train_loss
eval_train_loss
eval_validation_loss
eval_train_token_accuracy
eval_validation_token_accuracy
learning_rate
grad_norm
epoch
global_step
```

Required final eval:

```text
Spider dev execution_accuracy
executable_rate
invalid_sql_rate
empty_sql_rate
eval sample size / seed
checkpoint path
```

Decision rule:

```text
Pick the SFT checkpoint with the best validation execution accuracy, not the lowest training loss.
If validation execution accuracy is flat, prefer the earlier/lower-KL/lower-overfit checkpoint.
```

## Track B: SFT Systems Benchmark

Primary question:

```text
What is the memory and throughput tradeoff of each training system under a pure SFT workload?
```

This is the right place to study:

```text
ZeRO-2 vs ZeRO-3
FSDP vs ZeRO-3
activation checkpointing on/off
bf16 vs fp16
FlashAttention on/off
sequence length
micro batch size
gradient accumulation
tokens/sec
peak GPU memory
```

Candidate systems matrix:

```text
deepspeed_zero2
deepspeed_zero3
fsdp_full_shard
activation_checkpointing_on
activation_checkpointing_off
bf16
fp16
flash_attention_2
default_attention
```

Important caveats:

```text
FSDP is not currently represented by the existing SFT configs.
FlashAttention may require code/config support and compatible transformers/torch/model settings.
TP/SP/CP are not simple switches in the current HF Trainer path.
```

For systems runs, use a short fixed-step benchmark before training for quality:

```text
benchmark_steps: 50-100
eval_strategy: no
save_strategy: no or save only final
fixed sequence lengths if possible
same effective batch size when comparing throughput
```

Required systems logs:

```text
gpu_type
num_gpus
distributed_backend
sharding_strategy
activation_checkpointing
dtype
attention_impl
per_device_train_batch_size
gradient_accumulation_steps
effective_global_batch_size
max_prompt_length
max_response_length
tokens_per_sec
samples_per_sec
step_time_sec_mean
step_time_sec_p50
step_time_sec_p95
peak_gpu_memory_mb_per_rank
oom
```

Success criteria:

```text
No OOM.
Stable step time after warmup.
Higher tokens/sec at comparable effective batch size.
Acceptable peak memory headroom.
```

Interpretation rule:

```text
Use SFT systems results to choose a good learner backend.
Do not claim GRPO end-to-end speedup from SFT tokens/sec alone.
```

## Track C: GRPO Reward Design

Primary question:

```text
Which reward design creates useful group-relative signal and improves held-out SQL execution accuracy?
```

Candidate reward variants:

```text
final_only
discounted_final_gamma02
discounted_final_gamma04
discounted_final_gamma07
checker_intermediate_plus_final
invalid_sql_penalty
format_penalty
```

Hold fixed:

```text
SFT starting checkpoint
systems stack
kl_beta = 0.01 initially
clip_epsilon = 0.2
learning_rate = 5e-7
rollout.n = 4
task_batch_size = 4
eval set
```

Required logs:

```text
mean_reward
reward_std
reward_min
reward_max
reward_variance_mean
zero_variance_group_ratio
num_write_transitions
num_rewrite_transitions
rewrite_ratio
invalid_sql_rate
sqlite_error_rate
empty_sql_rate
final_reward_mean
intermediate_reward_mean
write_reward_mean
rewrite_reward_mean
policy_approx_kl
clip_fraction
```

Success criteria:

```text
Lower zero_variance_group_ratio.
Higher held-out execution accuracy.
No increase in invalid SQL or empty SQL.
No KL explosion.
```

Interpretation rule:

```text
Do not rank reward designs by training mean_reward alone.
Use validation execution accuracy and rollout failure modes.
```

## Track D: GRPO KL and Policy-Update Control

Primary question:

```text
How far should the policy move away from the SFT/reference policy?
```

Suggested first sweep:

```text
kl_beta: 0.0, 0.003, 0.01, 0.03
clip_epsilon: fixed at 0.2
learning_rate: fixed at 5e-7
```

Second sweep, only after choosing a reasonable `kl_beta`:

```text
clip_epsilon: 0.1, 0.2, 0.3
learning_rate: 2e-7, 5e-7, 1e-6
```

Required logs:

```text
approx_kl
policy_approx_kl
kl_loss
kl_beta
clip_fraction
ratio_mean
ratio_min
ratio_max
loss
policy_loss
mean_reward
validation_execution_accuracy
invalid_sql_rate
```

Flag conditions:

```text
policy_approx_kl increases for many consecutive steps.
clip_fraction > 0.5 for many consecutive steps.
ratio_max spikes.
validation accuracy drops while training reward rises.
loss becomes NaN or Inf.
```

Optional future feature:

```text
adaptive KL beta with target_kl.
```

If adaptive KL is added, log:

```text
target_kl
adaptive_kl_beta
kl_beta_update_reason
```

## Track E: GRPO RL Stability

Primary question:

```text
Which RL-specific settings keep online GRPO stable under noisy rollout rewards?
```

This track is different from SFT systems stability. It focuses on RL signal quality and policy-update stability.

Candidate knobs:

```text
rollout.n: 2, 4, 8
task_batch_size: 2, 4, 8
update_epochs: 1, 2
normalize_advantages: true, false
max_grad_norm: 0.5, 1.0
max_response_length: 512, 1024
temperature: 0.6, 0.8, 1.0
top_p: 0.9, 0.95
```

Required logs:

```text
loss
policy_loss
kl_loss
grad_norm
mean_reward
mean_advantage
advantage_std
reward_variance_mean
zero_variance_group_ratio
clip_fraction
policy_approx_kl
trainable_tokens
skipped_update
loss_is_finite
invalid_sql_rate
empty_sql_rate
```

Success criteria:

```text
No NaN/Inf.
No repeated skipped updates.
No runaway KL.
Reward variance is nonzero often enough for GRPO signal.
Validation execution accuracy does not collapse.
```

## Track F: GRPO Pipeline Efficiency

Primary question:

```text
Where does end-to-end GRPO time go in the Ray + vLLM + reference + learner pipeline?
```

This track should not test FSDP vs ZeRO-3 first. It should measure the actual GRPO pipeline bottleneck.

Candidate knobs:

```text
rollout.n
task_batch_size
max_response_length
max_turns
vllm_gpu_memory_utilization
reference colocated vs separate GPU
reference dtype / quantization if supported
LoRA reload frequency
layered_summon for LoRA-to-vLLM weight sync
checkpoint frequency
include_text in rollouts.jsonl
```

Required pipeline logs:

```text
step_wall_time_sec
rollout_time_sec
reference_logprob_time_sec
learner_train_time_sec
lora_reload_time_sec
checkpoint_time_sec
prompt_tokens
response_tokens
trainable_tokens
tokens_per_sec_total
tokens_per_sec_trainable
trajectories_per_sec
gpu_memory_used_mb_per_role
gpu_utilization_pct_per_role
```

Current lightweight log fields:

```text
rollout_time_sec
generate_time_sec
tool_time_sec
reward_time_sec
prompt_tokens
response_tokens
trainable_tokens
total_tokens
tokens_per_sec_total
tokens_per_sec_trainable
trajectories_per_sec
num_turns
num_execute_calls
num_check_calls
num_parse_errors
GPU_MONITOR timestamp,index,memory_used_mb,memory_total_mb,gpu_utilization_pct
```

Notes:

```text
reference_logprob_time_sec is absent when reference/KL is disabled.
learner_train_time_sec, checkpoint_time_sec, and exact step_wall_time_sec require patching verl's trainer loop.
The lightweight fields are enough to identify rollout bottlenecks, token throughput, and GPU idle/memory pressure.
For L40S memory pressure during vLLM wake-up, try layered_summon before switching the learner to QLoRA.
```

Success criteria:

```text
Identify the bottleneck stage.
Improve trajectories/sec without lowering validation execution accuracy.
Avoid GPU idle imbalance when possible.
```

Interpretation rule:

```text
Pipeline speed is not the same as learner speed.
If rollout dominates, ZeRO/FSDP changes may not improve end-to-end throughput.
```

## Track G: Algorithm Ablation

Primary question:

```text
Which parts of the final recipe are actually necessary?
```

Run ablations only after a reasonable baseline is stable.

Candidate ablations:

```text
SFT checkpoint only, no GRPO
GRPO with kl_beta = 0
GRPO with final-only reward
GRPO with no intermediate/checker reward
GRPO with normalize_advantages = false
GRPO with rollout.n = 2 vs 4 vs 8
GRPO with update_epochs = 1 vs 2
GRPO from different SFT checkpoints
```

Required logs:

```text
global minimal GRPO logs
validation execution accuracy
invalid_sql_rate
empty_sql_rate
reward_variance_mean
zero_variance_group_ratio
```

Recommended table columns:

```text
experiment_slug
git_commit
sft_start_checkpoint
reward_design
kl_beta
rollout_n
task_batch_size
max_steps
seed
final_train_reward
dev_execution_accuracy
invalid_sql_rate
notes
```

Interpretation rule:

```text
Compare ablations only if they share the same SFT starting checkpoint, evaluation set, and systems stack.
```

## Track H: Evaluation Protocol

Primary question:

```text
Does a checkpoint improve final SQL-agent behavior on held-out Spider examples?
```

Evaluation should be separate from training. Training curves are diagnostics, not final evidence.

Recommended schedule:

```text
smoke: inspect rollouts only.
pilot: evaluate a fixed 100-200 dev subset.
main: evaluate a fixed 500-example dev subset or full dev.
best candidates: rerun full dev eval and another seed.
```

Required eval logs:

```text
checkpoint_path
eval_config
eval_split
eval_limit
eval_seed
total
execution_accuracy
executable_rate
avg_turns
write_execution_accuracy
rewrite_execution_accuracy
invalid_sql_rate
sqlite_error_rate
empty_sql_rate
```

Keep fixed:

```text
prompt template
max_new_tokens
temperature
top_p
SQLite reward implementation
eval subset and seed
```

## TP / SP / CP Positioning

Tensor parallelism, sequence parallelism, and context parallelism should not be treated as ordinary config ablations in the current code path.

Current status:

```text
Current SFT path: HF Trainer + optional DeepSpeed.
Current GRPO path: Ray + vLLM rollout + custom learner.
TP/SP/CP likely require Megatron-LM, NeMo, Colossal-AI, verl-style learner, or a substantial learner rewrite.
```

How to study them:

```text
1. First document the current SFT ZeRO/FSDP limits.
2. If long-context or larger model training becomes necessary, create a separate scaling-backend track.
3. Compare TP/SP/CP on synthetic or SFT batches before integrating with GRPO.
4. Only then consider a GRPO learner backend using those parallelisms.
```

Do not mix TP/SP/CP into the first GRPO reward/KL experiments.

## Logging Implementation Plan

Add instrumentation in stages:

1. Shared run metadata:

```text
run_id
git_commit
config_path
hostname
python_version
torch_version
cuda_version
gpu_names
command
checkpoint_dir
```

2. SFT systems logging:

```text
tokens_per_sec
samples_per_sec
step_time_sec
peak_gpu_memory_mb_per_rank
effective_global_batch_size
dtype
sharding_strategy
activation_checkpointing
attention_impl
```

3. GRPO stage timing:

```text
rollout_time_sec
reference_logprob_time_sec
learner_train_time_sec
lora_reload_time_sec
step_wall_time_sec
```

4. GRPO token and reward breakdown:

```text
prompt_tokens
response_tokens
trainable_tokens
reward component fields
invalid_sql_rate
sqlite_error_rate
empty_sql_rate
```

5. Optional systems monitor:

```text
GPU utilization and memory sampling.
Use only for systems and pipeline experiments.
```

Do not enable heavy profilers by default.

## Recommended Experiment Order

Recommended first pass:

```text
0_sft_system_smoke
  Verify the SFT stack on available GPUs.

1_sft_system_benchmark
  Compare ZeRO-2 vs ZeRO-3, activation checkpointing, bf16, sequence length, batch size.

2_sft_quality_pilot
  Choose a good SFT starting checkpoint for GRPO.

3_grpo_baseline_smoke
  Run current Ray GRPO for 5-20 steps and inspect rollouts.

4_grpo_reward_pilot
  Compare reward designs with systems fixed.

5_grpo_kl_pilot
  Sweep kl_beta and clip_epsilon after reward design is reasonable.

6_grpo_rl_stability
  Tune rollout.n, task_batch_size, update_epochs, sampling temperature.

7_grpo_pipeline_benchmark
  Add timing and find rollout/reference/learner bottleneck.

8_main_ablation
  Run final ablations and external eval.
```

## Decision Record Template

For each main experiment:

```text
experiment_slug:
  track:
  hypothesis:
  changed_variables:
  fixed_variables:
  starting_checkpoint:
  expected_failure_mode:
  metrics_to_compare:
  result:
  keep_or_drop:
  follow_up:
```
