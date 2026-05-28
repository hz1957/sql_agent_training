"""Small but complete GRPO trainer for tokenized SQL-agent trajectories.

The trainer consumes the grouped trajectory format produced by `train.grpo` and
performs the actual RL update:

1. Compute group-relative advantages.
2. Cache old policy log-probabilities and reference log-probabilities.
3. Compute clipped GRPO policy loss plus a reference KL penalty.
4. Backpropagate and update policy weights.

The `tiny` backend is intentionally included so the whole update can run on a
laptop CPU/GPU while preserving the same math used by a larger model.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sql_agent_training.train.grpo import build_grpo_batch_from_config
from sql_agent_training.train.grpo_batch import GrpoBatch


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
    trainable_tokens: int


def compute_group_advantages(
    batch: GrpoBatch,
    *,
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Compute per-rollout group-relative GRPO advantages."""

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
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def _masked_mean(values: Any, mask: Any) -> Any:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


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
    ) -> None:
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.pad_token_id = pad_token_id
        self.loss_config = loss_config or GrpoLossConfig()
        self.device = device

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
            old_logprobs = _sequence_logprobs(
                self.policy_model,
                tensors["input_ids"],
                tensors["attention_mask"],
            ).detach()
            reference_logprobs = _sequence_logprobs(
                self.reference_model,
                tensors["input_ids"],
                tensors["attention_mask"],
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
        new_logprobs = _sequence_logprobs(self.policy_model, batch.input_ids, batch.attention_mask)
        ratio = torch.exp(new_logprobs - batch.old_logprobs)
        token_advantages = batch.advantages.unsqueeze(-1)
        unclipped = ratio * token_advantages
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.loss_config.clip_epsilon,
            1.0 + self.loss_config.clip_epsilon,
        )
        clipped = clipped_ratio * token_advantages
        policy_loss_tokens = -torch.minimum(unclipped, clipped)

        ref_delta = batch.reference_logprobs - new_logprobs
        kl_tokens = torch.exp(ref_delta) - ref_delta - 1.0
        policy_loss = _masked_mean(policy_loss_tokens, batch.response_mask)
        kl_loss = self.loss_config.kl_beta * _masked_mean(kl_tokens, batch.response_mask)
        loss = policy_loss + kl_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.loss_config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.loss_config.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            clip_mask = ((ratio - 1.0).abs() > self.loss_config.clip_epsilon).float()
            trainable_tokens = int(batch.response_mask.sum().item())
            return GrpoTrainMetrics(
                loss=float(loss.detach().cpu()),
                policy_loss=float(policy_loss.detach().cpu()),
                kl_loss=float(kl_loss.detach().cpu()),
                approx_kl=float(_masked_mean(kl_tokens, batch.response_mask).detach().cpu()),
                clip_fraction=float(_masked_mean(clip_mask, batch.response_mask).detach().cpu()),
                mean_reward=float(batch.rewards.mean().detach().cpu()),
                mean_advantage=float(batch.advantages.mean().detach().cpu()),
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


def _loss_config_from_config(config: dict[str, Any]) -> GrpoLossConfig:
    training = config.get("training", {})
    return GrpoLossConfig(
        clip_epsilon=float(training.get("clip_epsilon", 0.2)),
        kl_beta=float(training.get("kl_beta", 0.02)),
        advantage_epsilon=float(training.get("advantage_epsilon", 1e-6)),
        normalize_advantages=bool(training.get("normalize_advantages", True)),
        max_grad_norm=(
            float(training["max_grad_norm"]) if training.get("max_grad_norm") is not None else None
        ),
    )


def _build_models(config: dict[str, Any], batch: GrpoBatch, device: str) -> tuple[Any, Any, int]:
    torch = _require_torch()
    model_config = config.get("model", {})
    backend = str(model_config.get("backend", "hf"))
    if backend == "tiny":
        vocab_size = int(model_config.get("vocab_size") or (_max_token_id(batch) + 1))
        hidden_size = int(model_config.get("hidden_size", 32))
        policy = create_tiny_causal_lm(vocab_size=vocab_size, hidden_size=hidden_size).to(device)
        reference = copy.deepcopy(policy).to(device)
        pad_token_id = int(config.get("training", {}).get("pad_token_id", 0))
        return policy, reference, pad_token_id

    if backend != "hf":
        raise ValueError(f"Unknown GRPO model backend: {backend}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency check
        raise RuntimeError("Install the train extra to use Hugging Face GRPO training.") from exc

    model_path = str(model_config["path"])
    reference_path = str(model_config.get("reference_path") or model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    reference = AutoModelForCausalLM.from_pretrained(reference_path, trust_remote_code=True).to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    return policy, reference, int(pad_token_id)


def _save_policy(policy_model: Any, config: dict[str, Any], output_dir: str | Path) -> None:
    torch = _require_torch()
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(policy_model, "save_pretrained"):
        policy_model.save_pretrained(path)
    else:
        torch.save(policy_model.state_dict(), path / "tiny_policy.pt")
        (path / "tiny_config.json").write_text(json.dumps(config.get("model", {}), indent=2), encoding="utf-8")


def train_grpo_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build rollouts, run GRPO updates, save checkpoint, and return metrics."""

    torch = _require_torch()
    torch.manual_seed(int(config.get("training", {}).get("seed", 0)))
    batch = build_grpo_batch_from_config(config)
    device = _device_from_config(config)
    policy_model, reference_model, pad_token_id = _build_models(config, batch, device)
    training = config.get("training", {})
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=float(training.get("learning_rate", 1e-4)))
    trainer = GrpoTrainer(
        policy_model,
        reference_model,
        optimizer,
        pad_token_id=pad_token_id,
        loss_config=_loss_config_from_config(config),
        device=device,
    )
    prepared = trainer.prepare_batch(batch)
    metrics_history = [
        trainer.train_prepared_batch(prepared)
        for _ in range(int(training.get("max_steps", 1)))
    ]

    output = config.get("output", {})
    checkpoint_dir = output.get("checkpoint_dir", "artifacts/checkpoints/grpo")
    _save_policy(policy_model, config, checkpoint_dir)

    metrics_path = Path(output.get("metrics_json", Path(checkpoint_dir) / "metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_rows = [metrics.__dict__ for metrics in metrics_history]
    metrics_path.write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")
    final_metrics = metrics_history[-1].__dict__ if metrics_history else {}
    return {
        "groups": len(batch.groups),
        "trajectories": batch.num_trajectories,
        "steps": len(metrics_history),
        "device": device,
        "checkpoint_dir": str(checkpoint_dir),
        "metrics_json": str(metrics_path),
        **final_metrics,
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
