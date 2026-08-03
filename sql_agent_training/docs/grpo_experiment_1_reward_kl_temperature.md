# GRPO Experiment 1: Reward Design, KL, and Temperature

## Goal

This experiment studies the first GRPO algorithm axis for the multi-turn Spider SQL agent:

- rollout sampling temperature
- reward design
- initial KL strength

The goal is not to maximize final accuracy in one run. The goal is to identify a reward and sampling setup that gives
GRPO useful group-relative signal while keeping the policy close enough to the SFT policy.

The current running GRPO job is treated as the baseline.

## Fixed Setup

Keep these fixed across Experiment 1 unless a run fails:

```text
SFT starting policy: Qwen2.5-Coder-14B-Instruct-SFT-Merged
train split: Spider train
validation split: Spider validation/dev
rollout_n: 4
max_turns: 3
max_prompt_length: 2048
max_response_length: 2048
learning_rate: 5e-5
clip_epsilon: 0.2
normalize_advantages: true
systems stack: current verl + vLLM + FSDP stack
checkpoint/eval cadence: fixed within each comparison group
```

Initial KL setting:

```text
kl_beta = 0.005
```

Use the same SFT starting checkpoint, data split, seed policy, and evaluation protocol for all reward comparisons.

## Phase 0: Baseline Run

Use the currently running job as the baseline reference point.

Record:

```text
temperature
kl_beta
reward design
training reward mean/std
zero-variance group ratio
nonzero-variance group ratio
policy_approx_kl
clip_fraction
invalid_sql_rate
empty_sql_rate
final validation execution accuracy
checkpoint path
stdout log path
git commit
```

Interpretation:

```text
If training reward rises but validation execution accuracy drops, treat it as reward hacking or over-update risk.
If reward variance is near zero, GRPO has little usable signal regardless of training stability.
```

## A. Sampling Calibration

Primary questions:

```text
Which temperature gives enough within-group reward variance?
Is rollout_n = 4 enough?
```

Before comparing reward designs, run rollout probes at:

```text
temperature = 0.8
temperature = 1.0
temperature = 1.2
```

Hold fixed:

```text
reward design: current baseline reward
kl_beta: 0.005 for training runs
rollout_n: 4
max_turns: 3
top_p: 1.0
top_k: -1
```

Primary selection metric:

```text
nonzero_variance_group_ratio
```

Secondary metrics:

```text
invalid_sql_rate
empty_sql_rate
average response length
average turns
mean reward
reward std
SQLite error rate
```

Decision rule:

```text
Choose the lowest temperature that produces enough nonzero reward variance without sharply increasing invalid SQL,
empty SQL, or runaway response length.
```

Expected interpretation:

- `temperature=0.8`: likely more stable SQL, but may produce too little within-group diversity.
- `temperature=1.0`: likely balanced starting point.
- `temperature=1.2`: likely more variance, but higher risk of invalid SQL and noisy credit assignment.

Do not use validation accuracy alone to choose temperature in this phase. The probe is mainly about whether GRPO has
usable group-relative signal.

If `rollout_n=4` produces too many zero-variance groups at all three temperatures, run a small additional probe:

```text
rollout_n = 8
temperature = best candidate from {0.8, 1.0, 1.2}
```

Only increase `rollout_n` if the extra variance is large enough to justify the cost.

## B. Four-Scheme Initial Screening

After choosing a temperature from Sampling Calibration, compare the four reward schemes under one fixed setting.

Fixed:

```text
gamma = 0.9
beta = 0.1
temperature: chosen value from A
rollout_n: chosen value from A, initially 4
kl_beta: 0.005
learning_rate: 5e-5
clip_epsilon: 0.2
seed: fixed
```

Compare four reward/training schemes.

### Scheme 1: Chain Sampling With Final Reward

For each task, sample four independent complete trajectories. Each trajectory contains the initial SQL generation and
up to two rewrites.

Terminal reward:

```text
R_final = 1 if the final SQL is correct, otherwise 0.
```

The final reward is propagated to earlier SQL actions:

```text
V_t = gamma ^ d_t * R_final
```

where `d_t` is the number of remaining rewrite actions before the successful final SQL.

So a successful trajectory gives:

```text
first-turn correct: 1
fixed after one rewrite: gamma
fixed after two rewrites: gamma^2
failed trajectory: 0
```

This is the basic trajectory-level GRPO baseline.

### Scheme 2: Chain Sampling With Executability Fallback Reward

Use the same independent trajectory sampling as Scheme 1, but give a fallback reward to SQL actions that execute in
SQLite while remaining semantically incorrect.

Action value:

```text
V_t =
  gamma ^ d_t, if the trajectory eventually succeeds
  beta,        if the trajectory fails but the current SQL is executable
  0,           if the trajectory fails and the current SQL is not executable
```

Constraint:

```text
0 <= beta < gamma ^ D_max
```

The executability reward is a fallback, not an additive bonus. This prevents a repaired trajectory from receiving a
larger total reward than a trajectory that was correct on the first attempt.

This tests whether executable-but-wrong SQL gives a useful intermediate signal.

### Scheme 3: Tree Sampling With Final Reward

Replace independent trajectory sampling with tree-structured sampling.

At each decision state, sample four SQL candidates from the same parent state. The parent state contains:

- task description
- database schema
- previous SQL, if any
- SQLite execution result or error
- checker feedback

A correct SQL terminates that branch. An incorrect SQL may be expanded by sampling rewrite candidates, subject to the
rollout budget and `max_turns=3`.

Terminal node value:

```text
V(n) = 1 if the current SQL is correct, otherwise 0.
```

Non-terminal node value with mean backup:

```text
V(n) = gamma * mean(V(c) for c in children(n))
```

For four actions sampled from the same parent state, compute GRPO advantages:

```text
A_i = (V_i - mean(V_1, ..., V_4)) / (std(V_1, ..., V_4) + epsilon)
```

This tests whether state-level comparison and better credit assignment beat trajectory-level reward sharing.

### Scheme 4: Tree Sampling With Executability Fallback Reward

Combine tree sampling with executable-but-wrong fallback reward.

Node value:

```text
V(n) =
  1,                                      if the current SQL is correct
  gamma * mean(V(c) for c in children),   if expanded children have future value
  beta,                                   if all descendants fail but current SQL is executable
  0,                                      if all descendants fail and current SQL is not executable
```

As in Scheme 2, executability is a fallback reward, not an additive bonus.

This tests whether executability remains useful after more accurate tree-based credit assignment.

## Reward Hyperparameters

Initial screening uses:

```text
gamma = 0.9
beta = 0.1
```

This deliberately avoids a full grid during the first pass. The purpose is to identify:

```text
best no-executability backbone: Scheme 1 or Scheme 3
whether executability reward shows potential: Scheme 2 vs Scheme 1, or Scheme 4 vs Scheme 3
whether tree sampling is worth the implementation/training cost: Scheme 3 vs Scheme 1
```

Do not sweep all `scheme x gamma x beta` combinations at full length.

## Comparisons

The four schemes form a 2 x 2 design:

| Scheme | Sampling structure | Executability fallback |
| --- | --- | --- |
| Scheme 1 | Independent chain trajectories | No |
| Scheme 2 | Independent chain trajectories | Yes |
| Scheme 3 | Tree-structured sampling | No |
| Scheme 4 | Tree-structured sampling | Yes |

Main comparisons:

```text
Scheme 2 - Scheme 1: executability fallback under chain sampling
Scheme 3 - Scheme 1: tree sampling and state-level credit assignment
Scheme 4 - Scheme 3: executability fallback under tree sampling
```

Interaction:

```text
(Scheme 4 - Scheme 3) - (Scheme 2 - Scheme 1)
```

## Metrics

Primary metric:

```text
final execution-result accuracy against gold SQL execution result
```

Reward-signal metrics:

```text
mean_reward
reward_std
reward_min
reward_max
nonzero_variance_group_ratio
zero_variance_group_ratio
advantage_std
```

SQL behavior metrics:

```text
first_turn_sql_accuracy
final_sql_accuracy
sqlite_execution_success_rate
executable_but_incorrect_rate
invalid_sql_rate
empty_sql_rate
repair_success_rate_from_executable_incorrect_states
repair_success_rate_from_execution_error_states
average_rewrite_turns
average_sql_actions
```

Policy-update metrics:

```text
policy_approx_kl
kl_loss
clip_fraction
ratio_mean
ratio_max
actor_loss
grad_norm
```

Checker metrics:

```text
checker_false_positive_rate
checker_false_negative_rate
```

The checker is used to produce feedback for rewriting, but the final reward ground truth is the SQL execution-result
comparison against the gold SQL result.

## C. Gamma Search

Run gamma search on the best no-executability backbone from B.

Candidates:

```text
gamma = 0.7
gamma = 0.9
gamma = 0.95
```

Hold fixed:

```text
executability reward: disabled
temperature: chosen value from A
rollout_n: chosen value from A
kl_beta: 0.005
learning_rate: 5e-5
clip_epsilon: 0.2
seed: fixed
```

Decision rule:

```text
Choose gamma by validation execution accuracy and repair behavior, not by training mean reward alone.
```

Interpretation:

- Lower `gamma` rewards earlier correctness more strongly and penalizes needing rewrites.
- Higher `gamma` gives more credit to SQL that can be repaired later.
- If final accuracy improves but first-turn accuracy collapses, inspect whether the policy is learning to rely too much
  on rewrites.

## D. Beta Search

Run beta search only if executability reward showed potential in B.

Fixed:

```text
backbone: best chain/tree backbone from B
gamma: best gamma from C
temperature: chosen value from A
rollout_n: chosen value from A
kl_beta: 0.005
learning_rate: 5e-5
clip_epsilon: 0.2
seed: fixed
```

Candidates:

```text
beta = 0.0
beta = 0.05
beta = 0.1
beta = 0.2
```

Constraint:

```text
0 <= beta < gamma ^ D_max
```

Decision rule:

```text
Use the smallest beta that improves reward variance, repair behavior, or validation accuracy without increasing
executable-but-incorrect final outputs.
```

If executability reward does not help in B, skip the full beta sweep and keep the no-executability reward.

## E. Optimization Search

Only after choosing the reward design, gamma, and beta policy, tune optimization parameters.

Primary knobs:

```text
kl_beta
learning_rate
clip_epsilon if needed
```

KL candidates:

Only after choosing a reasonable reward design, sweep KL:

```text
kl_beta = 0.0
kl_beta = 0.003
kl_beta = 0.005
kl_beta = 0.01
kl_beta = 0.03
```

Hold fixed:

```text
temperature: chosen value from A
reward design: chosen value from B/C/D
gamma: chosen value from C
beta: chosen value from D, if used
learning_rate: 5e-5 initially
clip_epsilon: 0.2
rollout_n: chosen value from A
```

Decision rule:

```text
Choose the largest KL freedom that improves validation execution accuracy without increasing invalid SQL, empty SQL,
or repeated KL/clip spikes.
```

After KL, tune learning rate if needed:

```text
learning_rate = 1e-5
learning_rate = 5e-5
learning_rate = 1e-4
```

Only tune `clip_epsilon` if logs show clipping is the bottleneck:

```text
clip_epsilon = 0.1
clip_epsilon = 0.2
clip_epsilon = 0.3
```

Signs that clip tuning is needed:

```text
clip_fraction stays very high
ratio_max spikes repeatedly
validation accuracy drops while reward rises
policy_approx_kl grows monotonically
```

## F. Seed Confirmation

Use at least 2-3 seeds for:

```text
final best scheme
strongest baseline
corresponding no-executability version of the best scheme, if applicable
```

The corresponding no-executability comparison is important when the final best scheme uses executability reward.

Report:

```text
mean validation execution accuracy
standard deviation across seeds
best checkpoint step distribution
invalid_sql_rate mean/std
empty_sql_rate mean/std
nonzero_variance_group_ratio mean/std
policy_approx_kl mean/std
```

## Optional Temperature Refinement

After choosing reward design and KL, optionally refine temperature:

```text
temperature = chosen temperature - 0.1
temperature = chosen temperature
temperature = chosen temperature + 0.1
```

This is a final local search, not the first temperature decision.

## Recommended Experiment Order

```text
0. Current run as baseline
A. Sampling calibration: temperature 0.8 / 1.0 / 1.2, confirm rollout_n=4
B. Four-scheme initial screening: S1 / S2 / S3 / S4 with gamma=0.9 and beta=0.1
C. Gamma search on the best no-executability backbone: 0.7 / 0.9 / 0.95
D. Beta search on the best backbone only if executability reward shows potential: 0 / 0.05 / 0.1 / 0.2
E. Optimization search: KL, learning rate, and clip epsilon if needed
F. Seed confirmation: final best, strongest baseline, and corresponding no-executability scheme
```

## Notes On Implementation Status

The current verl AgentLoop can represent a complete multi-turn trajectory as one training sample with masks applied
only to SQL write/rewrite tokens.

Independent chain sampling is the closest to the current path.

Tree sampling requires an explicit branching rollout orchestrator or AgentLoop extension. It should be treated as a
separate implementation change, not just a config switch.
