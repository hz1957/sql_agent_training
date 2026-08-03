"""Shared deterministic sampling helpers for evaluation splits."""

from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")


def select_eval_examples(
    examples: list[T],
    *,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int = 0,
) -> list[T]:
    """Select evaluation examples with CLI smoke-test limits taking precedence."""

    if limit is not None:
        return list(examples[: max(0, limit)])
    if sample_size is None:
        return list(examples)

    size = max(0, int(sample_size))
    if size >= len(examples):
        return list(examples)
    return random.Random(sample_seed).sample(list(examples), size)
