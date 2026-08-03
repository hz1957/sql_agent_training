"""Shared structures for SQL agent trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentTurn:
    """One model or tool turn in a SQL agent trajectory."""

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTrajectory:
    """A completed SQL agent trajectory."""

    uid: str
    rollout_id: str
    turns: list[AgentTurn]
    final_sql: str | None
    final_sql_source: str
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenizedTrajectory:
    """One tokenized policy sample for GRPO-style training."""

    uid: str
    rollout_id: str
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    reward: float
    prompt_text: str | None = None
    response_text: str | None = None
    group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate tensor-like sequence lengths."""

        if not self.uid:
            raise ValueError("uid must be non-empty")
        if not self.rollout_id:
            raise ValueError("rollout_id must be non-empty")
        if not self.prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if len(self.response_ids) != len(self.response_mask):
            raise ValueError("response_ids and response_mask must have the same length")
        if any(mask not in {0, 1} for mask in self.response_mask):
            raise ValueError("response_mask must contain only 0/1 values")
