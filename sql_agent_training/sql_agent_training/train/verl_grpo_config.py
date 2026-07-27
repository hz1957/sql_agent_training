"""Lightweight validation for verl GRPO launch parameters."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


def _bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


@dataclass(frozen=True)
class VerlGrpoLaunchConfig:
    """Config relationships that should be checked before Ray/vLLM starts."""

    train_batch_size: int
    ppo_mini_batch_size: int
    ppo_micro_batch_size_per_gpu: int
    n_gpus_per_node: int
    rollout_n: int
    rollout_tp: int
    rollout_pp: int
    model_num_attention_heads: int
    log_prob_use_dynamic_bsz: bool
    log_prob_micro_batch_size_per_gpu: int | None
    balance_batch: bool = True

    @property
    def rollout_global_batch_size(self) -> int:
        return self.train_batch_size * self.rollout_n

    def validate(self) -> None:
        positive_fields = {
            "train_batch_size": self.train_batch_size,
            "ppo_mini_batch_size": self.ppo_mini_batch_size,
            "ppo_micro_batch_size_per_gpu": self.ppo_micro_batch_size_per_gpu,
            "n_gpus_per_node": self.n_gpus_per_node,
            "rollout_n": self.rollout_n,
            "rollout_tp": self.rollout_tp,
            "model_num_attention_heads": self.model_num_attention_heads,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.model_num_attention_heads % self.rollout_tp != 0:
            raise ValueError(
                f"rollout_tp ({self.rollout_tp}) must divide model_num_attention_heads "
                f"({self.model_num_attention_heads})."
            )

        if self.rollout_pp != 1:
            raise ValueError("Current verl vLLM rollout path expects rollout_pp=1.")

        if self.balance_batch and self.rollout_global_batch_size < self.n_gpus_per_node:
            raise ValueError(
                "train_batch_size * rollout_n must be >= n_gpus_per_node when balance_batch=True. "
                f"Got {self.train_batch_size} * {self.rollout_n} = {self.rollout_global_batch_size} "
                f"< {self.n_gpus_per_node}."
            )

        if self.ppo_mini_batch_size > self.train_batch_size:
            raise ValueError(
                f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be <= "
                f"train_batch_size ({self.train_batch_size})."
            )

        if self.ppo_micro_batch_size_per_gpu > self.ppo_mini_batch_size:
            raise ValueError(
                f"ppo_micro_batch_size_per_gpu ({self.ppo_micro_batch_size_per_gpu}) must be <= "
                f"ppo_mini_batch_size ({self.ppo_mini_batch_size})."
            )

        if not self.log_prob_use_dynamic_bsz:
            if self.log_prob_micro_batch_size_per_gpu is None:
                raise ValueError(
                    "log_prob_micro_batch_size_per_gpu is required when log_prob_use_dynamic_bsz=False."
                )
            if self.log_prob_micro_batch_size_per_gpu <= 0:
                raise ValueError(
                    f"log_prob_micro_batch_size_per_gpu must be positive, got "
                    f"{self.log_prob_micro_batch_size_per_gpu}."
                )

    def summary(self) -> str:
        return (
            "verl grpo config ok: "
            f"train_batch_size={self.train_batch_size}, "
            f"rollout_n={self.rollout_n}, "
            f"rollout_global_batch_size={self.rollout_global_batch_size}, "
            f"ppo_mini_batch_size={self.ppo_mini_batch_size}, "
            f"log_prob_use_dynamic_bsz={self.log_prob_use_dynamic_bsz}, "
            f"log_prob_micro_batch_size_per_gpu={self.log_prob_micro_batch_size_per_gpu}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--ppo-mini-batch-size", type=int, required=True)
    parser.add_argument("--ppo-micro-batch-size-per-gpu", type=int, required=True)
    parser.add_argument("--n-gpus-per-node", type=int, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--rollout-tp", type=int, required=True)
    parser.add_argument("--rollout-pp", type=int, required=True)
    parser.add_argument("--model-num-attention-heads", type=int, required=True)
    parser.add_argument("--log-prob-use-dynamic-bsz", required=True)
    parser.add_argument("--log-prob-micro-batch-size-per-gpu", type=int)
    parser.add_argument("--balance-batch", default="True")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = VerlGrpoLaunchConfig(
        train_batch_size=args.train_batch_size,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=args.ppo_micro_batch_size_per_gpu,
        n_gpus_per_node=args.n_gpus_per_node,
        rollout_n=args.rollout_n,
        rollout_tp=args.rollout_tp,
        rollout_pp=args.rollout_pp,
        model_num_attention_heads=args.model_num_attention_heads,
        log_prob_use_dynamic_bsz=_bool(args.log_prob_use_dynamic_bsz),
        log_prob_micro_batch_size_per_gpu=args.log_prob_micro_batch_size_per_gpu,
        balance_batch=_bool(args.balance_batch),
    )
    config.validate()
    print(config.summary())


if __name__ == "__main__":
    main()
