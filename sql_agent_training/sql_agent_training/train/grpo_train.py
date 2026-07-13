"""Small but complete GRPO training loop for tokenized SQL-agent transitions.

The trainer consumes the grouped policy-sample format produced by `train.grpo_rollouts` and
performs the actual RL update:

1. Compute group-relative advantages.
2. Cache old policy log-probabilities and reference log-probabilities.
3. Reuse the prepared batch for one or more clipped actor update epochs.
4. Compute clipped GRPO policy loss plus a reference KL penalty.

The `tiny` backend is intentionally included so the whole update can run on a
laptop CPU/GPU while preserving the same math used by a larger model.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sql_agent_training.agent.model_client import HuggingFaceInMemoryModelClient, ModelClient
from sql_agent_training.data.spider_dataset import SpiderExample
from sql_agent_training.train.grpo_rollouts import (
    RolloutJsonlWriter,
    build_rollout_batch_from_config,
    load_rollout_source_from_config,
)
from sql_agent_training.train.grpo_batch import GrpoBatch, GrpoGroup, summarize_grpo_batch


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency check
        raise RuntimeError("Install the train extra to run GRPO training: uv sync --extra train") from exc
    return torch


@dataclass(frozen=True)
class GrpoLossConfig:
    """Hyperparameters for one GRPO update."""

    clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    advantage_epsilon: float = 1e-6
    normalize_advantages: bool = True
    max_grad_norm: float | None = 1.0


@dataclass(frozen=True)
class GrpoTrainingBatch:
    """Padded tensors and cached log-probs for a GRPO update."""

    input_ids: Any
    attention_mask: Any
    response_mask: Any
    advantages: Any
    rewards: Any
    old_logprobs: Any
    reference_logprobs: Any
    rollout_ids: list[str]


@dataclass(frozen=True)
class GrpoTrainMetrics:
    """Scalar metrics from one optimizer step."""

    loss: float
    policy_loss: float
    kl_loss: float
    approx_kl: float
    clip_fraction: float
    mean_reward: float
    mean_advantage: float
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    policy_approx_kl: float
    trainable_tokens: int


def compute_group_advantages(
    batch: GrpoBatch,
    *,
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Compute per-sample group-relative GRPO advantages."""

    advantages: dict[str, float] = {}
    for group in batch.groups:
        rewards = [float(trajectory.reward) for trajectory in group.trajectories]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        std = variance**0.5
        for trajectory, reward in zip(group.trajectories, rewards, strict=True):
            centered = reward - mean_reward
            advantages[trajectory.rollout_id] = centered / (std + epsilon) if normalize and std > epsilon else centered
    return advantages


def _sequence_logprobs(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    torch = _require_torch()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    target_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return target_logits - torch.logsumexp(logits, dim=-1)


def _sequence_logprobs_microbatched(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    micro_batch_size: int,
) -> Any:
    torch = _require_torch()
    if micro_batch_size <= 0 or micro_batch_size >= int(input_ids.shape[0]):
        return _sequence_logprobs(model, input_ids, attention_mask)

    rows = []
    for start in range(0, int(input_ids.shape[0]), micro_batch_size):
        stop = start + micro_batch_size
        rows.append(_sequence_logprobs(model, input_ids[start:stop], attention_mask[start:stop]))
    return torch.cat(rows, dim=0)


def _masked_values(values: Any, mask: Any) -> Any:
    return values[mask.bool()]


def _max_token_id(batch: GrpoBatch) -> int:
    max_id = 0
    for trajectory in batch.trajectories:
        max_id = max(max_id, *(trajectory.prompt_ids + trajectory.response_ids))
    return max_id


def build_training_tensors(
    batch: GrpoBatch,
    *,
    advantages: dict[str, float],
    pad_token_id: int,
    device: str,
) -> dict[str, Any]:
    """Pad tokenized trajectories and align response masks to next-token log-probs."""

    torch = _require_torch()
    trajectories = batch.trajectories
    max_length = max(len(item.prompt_ids) + len(item.response_ids) for item in trajectories)
    if max_length < 2:
        raise ValueError("GRPO training requires at least two tokens per trajectory")

    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    response_mask_rows: list[list[int]] = []
    reward_rows: list[float] = []
    advantage_rows: list[float] = []
    rollout_ids: list[str] = []

    for trajectory in trajectories:
        sequence = trajectory.prompt_ids + trajectory.response_ids
        pad_length = max_length - len(sequence)
        input_rows.append(sequence + [pad_token_id] * pad_length)
        attention_rows.append([1] * len(sequence) + [0] * pad_length)

        shifted_mask = [0] * (max_length - 1)
        target_start = len(trajectory.prompt_ids) - 1
        for offset, mask_value in enumerate(trajectory.response_mask):
            shifted_mask[target_start + offset] = int(mask_value)
        response_mask_rows.append(shifted_mask)
        reward_rows.append(float(trajectory.reward))
        advantage_rows.append(float(advantages[trajectory.rollout_id]))
        rollout_ids.append(trajectory.rollout_id)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long, device=device),
        "response_mask": torch.tensor(response_mask_rows, dtype=torch.float32, device=device),
        "rewards": torch.tensor(reward_rows, dtype=torch.float32, device=device),
        "advantages": torch.tensor(advantage_rows, dtype=torch.float32, device=device),
        "rollout_ids": rollout_ids,
    }


class GrpoTrainer:
    """Run clipped GRPO updates against a causal language model."""

    def __init__(
        self,
        policy_model: Any,
        reference_model: Any,
        optimizer: Any,
        *,
        pad_token_id: int,
        loss_config: GrpoLossConfig | None = None,
        device: str = "cpu",
        logprob_micro_batch_size: int = 1,
    ) -> None:
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.pad_token_id = pad_token_id
        self.loss_config = loss_config or GrpoLossConfig()
        self.device = device
        self.logprob_micro_batch_size = logprob_micro_batch_size

    def prepare_batch(self, batch: GrpoBatch) -> GrpoTrainingBatch:
        """Build padded tensors and cache old/reference log-probabilities."""

        torch = _require_torch()
        advantages = compute_group_advantages(
            batch,
            normalize=self.loss_config.normalize_advantages,
            epsilon=self.loss_config.advantage_epsilon,
        )
        tensors = build_training_tensors(
            batch,
            advantages=advantages,
            pad_token_id=self.pad_token_id,
            device=self.device,
        )
        self.policy_model.eval()
        self.reference_model.eval()
        with torch.no_grad():
            old_logprobs = _sequence_logprobs_microbatched(
                self.policy_model,
                tensors["input_ids"],
                tensors["attention_mask"],
                micro_batch_size=self.logprob_micro_batch_size,
            ).detach()
            reference_logprobs = _sequence_logprobs_microbatched(
                self.reference_model,
                tensors["input_ids"],
                tensors["attention_mask"],
                micro_batch_size=self.logprob_micro_batch_size,
            ).detach()

        return GrpoTrainingBatch(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            response_mask=tensors["response_mask"],
            advantages=tensors["advantages"],
            rewards=tensors["rewards"],
            old_logprobs=old_logprobs,
            reference_logprobs=reference_logprobs,
            rollout_ids=tensors["rollout_ids"],
        )

    def train_prepared_batch(self, batch: GrpoTrainingBatch) -> GrpoTrainMetrics:
        """Run one optimizer step on a prepared GRPO batch."""

        torch = _require_torch()
        self.policy_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_tokens = batch.response_mask.sum().clamp_min(1.0)
        micro_batch_size = self.logprob_micro_batch_size
        batch_size = int(batch.input_ids.shape[0])

        metric_sums = {
            "policy_loss": 0.0,
            "kl": 0.0,
            "clip_fraction": 0.0,
            "ratio": 0.0,
            "policy_approx_kl": 0.0,
        }
        ratio_min: float | None = None
        ratio_max: float | None = None
        trainable_tokens = int(batch.response_mask.sum().detach().cpu().item())

        for start in range(0, batch_size, micro_batch_size if micro_batch_size > 0 else batch_size):
            stop = start + (micro_batch_size if micro_batch_size > 0 else batch_size)
            new_logprobs = _sequence_logprobs(
                self.policy_model,
                batch.input_ids[start:stop],
                batch.attention_mask[start:stop],
            )
            old_logprobs = batch.old_logprobs[start:stop]
            reference_logprobs = batch.reference_logprobs[start:stop]
            response_mask = batch.response_mask[start:stop]
            token_advantages = batch.advantages[start:stop].unsqueeze(-1)

            log_ratio = new_logprobs - old_logprobs
            ratio = torch.exp(log_ratio)
            unclipped = ratio * token_advantages
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - self.loss_config.clip_epsilon,
                1.0 + self.loss_config.clip_epsilon,
            )
            clipped = clipped_ratio * token_advantages
            policy_loss_tokens = -torch.minimum(unclipped, clipped)

            ref_delta = reference_logprobs - new_logprobs
            kl_tokens = torch.exp(ref_delta) - ref_delta - 1.0
            policy_kl_tokens = (ratio - 1.0) - log_ratio
            micro_policy_loss = (policy_loss_tokens * response_mask).sum() / total_tokens
            micro_kl_loss = self.loss_config.kl_beta * (kl_tokens * response_mask).sum() / total_tokens
            (micro_policy_loss + micro_kl_loss).backward()

            with torch.no_grad():
                response_ratios = _masked_values(ratio, response_mask)
                metric_sums["policy_loss"] += float((policy_loss_tokens * response_mask).sum().detach().cpu())
                metric_sums["kl"] += float((kl_tokens * response_mask).sum().detach().cpu())
                clip_mask = ((ratio - 1.0).abs() > self.loss_config.clip_epsilon).float()
                metric_sums["clip_fraction"] += float((clip_mask * response_mask).sum().detach().cpu())
                metric_sums["policy_approx_kl"] += float((policy_kl_tokens * response_mask).sum().detach().cpu())
                if response_ratios.numel() > 0:
                    metric_sums["ratio"] += float(response_ratios.sum().detach().cpu())
                    current_min = float(response_ratios.min().detach().cpu())
                    current_max = float(response_ratios.max().detach().cpu())
                    ratio_min = current_min if ratio_min is None else min(ratio_min, current_min)
                    ratio_max = current_max if ratio_max is None else max(ratio_max, current_max)

        if self.loss_config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.loss_config.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            denominator = float(max(trainable_tokens, 1))
            policy_loss_value = metric_sums["policy_loss"] / denominator
            approx_kl_value = metric_sums["kl"] / denominator
            kl_loss_value = self.loss_config.kl_beta * approx_kl_value
            return GrpoTrainMetrics(
                loss=policy_loss_value + kl_loss_value,
                policy_loss=policy_loss_value,
                kl_loss=kl_loss_value,
                approx_kl=approx_kl_value,
                clip_fraction=metric_sums["clip_fraction"] / denominator,
                mean_reward=float(batch.rewards.mean().detach().cpu()),
                mean_advantage=float(batch.advantages.mean().detach().cpu()),
                ratio_mean=metric_sums["ratio"] / denominator if trainable_tokens else 0.0,
                ratio_min=ratio_min if ratio_min is not None else 0.0,
                ratio_max=ratio_max if ratio_max is not None else 0.0,
                policy_approx_kl=metric_sums["policy_approx_kl"] / denominator,
                trainable_tokens=trainable_tokens,
            )


def create_tiny_causal_lm(*, vocab_size: int, hidden_size: int = 32) -> Any:
    """Create a tiny causal LM for local GRPO smoke tests."""

    torch = _require_torch()

    class TinyCausalLm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden_size)
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids: Any, attention_mask: Any | None = None) -> Any:
            del attention_mask
            return type("CausalLmOutput", (), {"logits": self.lm_head(self.embed(input_ids))})

    return TinyCausalLm()


def _device_from_config(config: dict[str, Any]) -> str:
    torch = _require_torch()
    requested = str(config.get("training", {}).get("device", "auto"))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _torch_dtype_from_config(config: dict[str, Any], device: str) -> Any | None:
    torch = _require_torch()
    if not device.startswith("cuda"):
        return None

    requested = str(config.get("model", {}).get("torch_dtype", "auto")).lower()
    if requested in {"none", "float32", "fp32"}:
        return None if requested == "none" else torch.float32
    if requested in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if requested in {"float16", "fp16", "half"}:
        return torch.float16
    if requested != "auto":
        raise ValueError(f"Unknown model.torch_dtype: {requested}")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _loss_config_from_config(config: dict[str, Any]) -> GrpoLossConfig:
    training = config.get("training", {})
    return GrpoLossConfig(
        clip_epsilon=float(training.get("clip_epsilon", 0.2)),
        kl_beta=float(training.get("kl_beta", 0.02)),
        advantage_epsilon=float(training.get("advantage_epsilon", 1e-6)),
        normalize_advantages=bool(training.get("normalize_advantages", True)),
        max_grad_norm=(float(training["max_grad_norm"]) if training.get("max_grad_norm") is not None else None),
    )


def _apply_peft_adapter_if_configured(model: Any, adapter_path: str | None, *, is_trainable: bool) -> Any:
    """Attach a PEFT adapter to a base model when an adapter path is configured."""

    if not adapter_path:
        return model

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - optional dependency check
        raise RuntimeError("Install the train extra with PEFT to run GRPO from a LoRA adapter.") from exc

    return PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)


def _build_models(config: dict[str, Any], batch: GrpoBatch | None, device: str) -> tuple[Any, Any, int, Any | None]:
    torch = _require_torch()
    model_config = config.get("model", {})
    backend = str(model_config.get("backend", "hf"))
    if backend == "tiny":
        vocab_size = int(model_config.get("vocab_size") or (_max_token_id(batch) + 1 if batch is not None else 1024))
        hidden_size = int(model_config.get("hidden_size", 32))
        policy = create_tiny_causal_lm(vocab_size=vocab_size, hidden_size=hidden_size).to(device)
        reference = copy.deepcopy(policy).to(device)
        pad_token_id = int(config.get("training", {}).get("pad_token_id", 0))
        return policy, reference, pad_token_id, None

    if backend != "hf":
        raise ValueError(f"Unknown GRPO model backend: {backend}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency check
        raise RuntimeError("Install the train extra to use Hugging Face GRPO training.") from exc

    model_path = str(model_config["path"])
    reference_path = str(model_config.get("reference_path") or model_path)
    adapter_path = str(model_config["adapter_path"]) if model_config.get("adapter_path") else None
    reference_adapter_path = (
        str(model_config["reference_adapter_path"])
        if model_config.get("reference_adapter_path")
        else adapter_path
    )
    tokenizer_config = config.get("tokenizer", {})
    tokenizer_path = str(model_config.get("tokenizer_path") or tokenizer_config.get("path") or model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    torch_dtype = _torch_dtype_from_config(config, device)
    policy_base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    policy = _apply_peft_adapter_if_configured(policy_base, adapter_path, is_trainable=True).to(device)
    reference_base = AutoModelForCausalLM.from_pretrained(
        reference_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    reference = _apply_peft_adapter_if_configured(
        reference_base,
        reference_adapter_path,
        is_trainable=False,
    ).to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    return policy, reference, int(pad_token_id), tokenizer


def _save_policy(policy_model: Any, tokenizer: Any | None, config: dict[str, Any], output_dir: str | Path) -> None:
    torch = _require_torch()
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(policy_model, "save_pretrained"):
        policy_model.save_pretrained(path)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(path)
    else:
        torch.save(policy_model.state_dict(), path / "tiny_policy.pt")
        (path / "tiny_config.json").write_text(json.dumps(config.get("model", {}), indent=2), encoding="utf-8")


def _new_checkpoint_dir(base_dir: str | Path) -> Path:
    root = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    counter = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{counter:02d}"
        counter += 1
    return candidate


def _task_batch_size(config: dict[str, Any]) -> int | None:
    training = config.get("training", {})
    rollout = config.get("rollout", {})
    value = training.get("task_batch_size", rollout.get("task_batch_size"))
    if value is None:
        return None
    size = int(value)
    return size if size > 0 else None


def _update_epochs(config: dict[str, Any]) -> int:
    epochs = int(config.get("training", {}).get("update_epochs", 1))
    if epochs <= 0:
        raise ValueError("training.update_epochs must be positive")
    return epochs


def _logprob_micro_batch_size(config: dict[str, Any]) -> int:
    size = int(config.get("training", {}).get("logprob_micro_batch_size", 1))
    if size <= 0:
        raise ValueError("training.logprob_micro_batch_size must be positive")
    return size


def _hard_example_mining_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("training", {}).get("hard_example_mining", False))


def _hard_example_candidate_batch_size(config: dict[str, Any], task_batch_size: int | None) -> int | None:
    if not _hard_example_mining_enabled(config) or task_batch_size is None:
        return task_batch_size
    training = config.get("training", {})
    multiplier = int(training.get("hard_example_candidate_multiplier", 2))
    if multiplier <= 0:
        raise ValueError("training.hard_example_candidate_multiplier must be positive")
    return task_batch_size * multiplier


def _hard_example_variance_epsilon(config: dict[str, Any]) -> float:
    return float(config.get("training", {}).get("hard_example_variance_epsilon", 1e-12))


def _group_reward_variance(group: GrpoGroup) -> float:
    rewards = [float(trajectory.reward) for trajectory in group.trajectories]
    if not rewards:
        return 0.0
    mean_reward = sum(rewards) / len(rewards)
    return sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)


def select_hard_example_batch(
    batch: GrpoBatch,
    *,
    target_groups: int | None,
    variance_epsilon: float = 1e-12,
) -> tuple[GrpoBatch, dict[str, Any]]:
    """Keep groups with non-zero reward variance for stronger GRPO signal."""

    variances = {group.uid: _group_reward_variance(group) for group in batch.groups}
    hard_groups = [group for group in batch.groups if variances[group.uid] > variance_epsilon]
    hard_groups.sort(key=lambda group: (-variances[group.uid], group.uid))
    if target_groups is not None and target_groups > 0:
        selected_groups = hard_groups[:target_groups]
    else:
        selected_groups = hard_groups

    selected_batch = GrpoBatch(groups=selected_groups)
    candidate_groups = len(batch.groups)
    return selected_batch, {
        "hard_mining_enabled": True,
        "candidate_groups": candidate_groups,
        "candidate_trajectories": batch.num_trajectories,
        "hard_groups": len(hard_groups),
        "hard_example_ratio": len(hard_groups) / candidate_groups if candidate_groups else 0.0,
        "selected_groups": len(selected_groups),
        "selected_trajectories": selected_batch.num_trajectories,
        "hard_example_variance_epsilon": variance_epsilon,
        "hard_reward_variance_per_group": {
            group.uid: variances[group.uid] for group in selected_groups
        },
    }


def _sample_step_examples(
    examples: list[SpiderExample],
    *,
    task_batch_size: int | None,
    rng: random.Random,
) -> list[SpiderExample]:
    if not examples:
        raise ValueError("training examples must be non-empty")
    if task_batch_size is None or task_batch_size >= len(examples):
        return list(examples)
    return rng.sample(examples, task_batch_size)


def _build_online_rollout_client(
    policy_model: Any,
    tokenizer: Any | None,
    config: dict[str, Any],
    device: str,
) -> ModelClient | None:
    if tokenizer is None:
        return None

    rollout = config.get("rollout", {})
    return HuggingFaceInMemoryModelClient(
        policy_model,
        tokenizer,
        device=device,
        max_new_tokens=int(rollout.get("max_response_length", 256)),
        temperature=float(rollout.get("temperature", 0.0)),
        top_p=float(rollout["top_p"]) if rollout.get("top_p") is not None else None,
        top_k=int(rollout["top_k"]) if rollout.get("top_k") is not None else None,
    )


def _write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _write_run_config(config: dict[str, Any], checkpoint_dir: Path) -> Path:
    path = checkpoint_dir / "run_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def train_grpo_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build rollouts, run GRPO updates, save checkpoint, and return metrics."""

    torch = _require_torch()
    seed = int(config.get("training", {}).get("seed", 0))
    torch.manual_seed(seed)
    rng = random.Random(seed)
    output = config.get("output", {})
    checkpoint_dir = _new_checkpoint_dir(output.get("checkpoint_dir", "artifacts/checkpoints/grpo"))
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    run_config_path = _write_run_config(config, checkpoint_dir)
    rollouts_jsonl = Path(output.get("rollouts_jsonl", Path(checkpoint_dir) / "rollouts.jsonl"))
    include_text = bool(output.get("include_text", True))
    metrics_jsonl = Path(output.get("metrics_jsonl", Path(checkpoint_dir) / "metrics.jsonl"))
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()
    device = _device_from_config(config)
    policy_model, reference_model, pad_token_id, tokenizer = _build_models(config, None, device)
    training = config.get("training", {})
    max_steps = int(training.get("max_steps", 1))
    if max_steps <= 0:
        raise ValueError("training.max_steps must be positive")
    update_epochs = _update_epochs(config)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=float(training.get("learning_rate", 1e-4)))
    trainer = GrpoTrainer(
        policy_model,
        reference_model,
        optimizer,
        pad_token_id=pad_token_id,
        loss_config=_loss_config_from_config(config),
        device=device,
        logprob_micro_batch_size=_logprob_micro_batch_size(config),
    )
    rollout_client = _build_online_rollout_client(policy_model, tokenizer, config, device)
    task_batch_size = _task_batch_size(config)
    candidate_task_batch_size = _hard_example_candidate_batch_size(config, task_batch_size)
    hard_mining_enabled = _hard_example_mining_enabled(config)
    hard_variance_epsilon = _hard_example_variance_epsilon(config)
    source = load_rollout_source_from_config(config)
    metrics_history: list[dict[str, Any]] = []
    rows_written = 0
    total_trajectories = 0
    final_batch_stats: dict[str, Any] = {}
    final_groups = 0
    optimizer_steps = 0
    save_every_steps = int(training.get("save_every_steps", 0) or 0)
    try:
        with RolloutJsonlWriter(rollouts_jsonl, include_text=include_text) as rollout_writer:
            for step_index in range(max_steps):
                step = step_index + 1
                step_examples = _sample_step_examples(
                    source.examples,
                    task_batch_size=candidate_task_batch_size,
                    rng=rng,
                )
                batch = build_rollout_batch_from_config(
                    config,
                    rollout_writer=rollout_writer,
                    model_client=rollout_client,
                    hf_tokenizer=tokenizer,
                    examples=step_examples,
                    source=source,
                )
                hard_mining_stats: dict[str, Any] = {"hard_mining_enabled": False}
                if hard_mining_enabled:
                    candidate_batch = batch
                    batch, hard_mining_stats = select_hard_example_batch(
                        candidate_batch,
                        target_groups=task_batch_size,
                        variance_epsilon=hard_variance_epsilon,
                    )
                    if not batch.groups:
                        candidate_stats = summarize_grpo_batch(candidate_batch)
                        metric_row = {
                            "step": step,
                            "update_epoch": 0,
                            "optimizer_step": optimizer_steps,
                            "groups": 0,
                            "trajectories": 0,
                            "skipped_update": True,
                            "skip_reason": "no_hard_example_groups",
                            **candidate_stats,
                            **hard_mining_stats,
                        }
                        _write_jsonl_row(metrics_jsonl, metric_row)
                        metrics_history.append(metric_row)
                        rows_written = rollout_writer.count
                        final_batch_stats = candidate_stats
                        final_groups = 0
                        continue
                batch_stats = summarize_grpo_batch(batch)
                prepared = trainer.prepare_batch(batch)
                for update_epoch in range(1, update_epochs + 1):
                    optimizer_steps += 1
                    metrics = trainer.train_prepared_batch(prepared)
                    metric_row = {
                        "step": step,
                        "update_epoch": update_epoch,
                        "optimizer_step": optimizer_steps,
                        "groups": len(batch.groups),
                        "trajectories": batch.num_trajectories,
                        "skipped_update": False,
                        **batch_stats,
                        **hard_mining_stats,
                        **metrics.__dict__,
                    }
                    _write_jsonl_row(metrics_jsonl, metric_row)
                    metrics_history.append(metric_row)
                rows_written = rollout_writer.count
                total_trajectories += batch.num_trajectories
                final_batch_stats = batch_stats
                final_groups = len(batch.groups)
                if save_every_steps > 0 and step % save_every_steps == 0 and step != max_steps:
                    _save_policy(policy_model, tokenizer, config, checkpoint_dir / f"step_{step:06d}")
    finally:
        source.close()

    _save_policy(policy_model, tokenizer, config, checkpoint_dir)

    metrics_path = Path(output.get("metrics_json", Path(checkpoint_dir) / "metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_history, indent=2), encoding="utf-8")
    final_metrics = metrics_history[-1] if metrics_history else {}
    return {
        **final_batch_stats,
        **final_metrics,
        "groups": final_groups,
        "trajectories": total_trajectories,
        "steps": max_steps,
        "update_epochs": update_epochs,
        "optimizer_steps": optimizer_steps,
        "device": device,
        "rows_written": rows_written,
        "rollouts_jsonl": str(rollouts_jsonl),
        "checkpoint_dir": str(checkpoint_dir),
        "metrics_json": str(metrics_path),
        "metrics_jsonl": str(metrics_jsonl),
        "run_config": str(run_config_path),
    }


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal complete GRPO trainer.")
    parser.add_argument("--config", default="configs/grpo.local_dryrun.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Force built-in demo rollouts.")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.dry_run:
        config["dry_run"] = True
    summary = train_grpo_from_config(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
