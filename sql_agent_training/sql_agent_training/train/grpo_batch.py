"""GRPO transition grouping contracts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sql_agent_training.agent.trace_format import TokenizedTrajectory


@dataclass(frozen=True)
class GrpoGroup:
    """GRPO comparison group for samples from the same task."""

    uid: str
    trajectories: list[TokenizedTrajectory]


@dataclass(frozen=True)
class GrpoBatch:
    """A batch of grouped policy samples ready for GRPO advantage computation."""

    groups: list[GrpoGroup]

    @property
    def trajectories(self) -> list[TokenizedTrajectory]:
        """Flatten all trajectories in group order."""

        return [trajectory for group in self.groups for trajectory in group.trajectories]

    @property
    def num_trajectories(self) -> int:
        """Total number of trajectories."""

        return len(self.trajectories)


def build_grpo_batch(trajectories: list[TokenizedTrajectory], *, rollout_n: int, strict: bool = True) -> GrpoBatch:
    """Group tokenized policy samples by task/group id.

    Args:
        trajectories: Flat list of sampled policy transitions.
        rollout_n: Expected number of independent rollouts per group in strict mode.
        strict: If True, every group must have exactly rollout_n samples.

    Returns:
        Grouped GRPO batch.
    """

    if rollout_n <= 0:
        raise ValueError("rollout_n must be positive")
    if not trajectories:
        raise ValueError("trajectories must be non-empty")

    grouped: dict[str, list[TokenizedTrajectory]] = defaultdict(list)
    seen_rollout_ids: set[str] = set()
    for trajectory in trajectories:
        trajectory.validate()
        if trajectory.rollout_id in seen_rollout_ids:
            raise ValueError(f"duplicate rollout_id: {trajectory.rollout_id}")
        seen_rollout_ids.add(trajectory.rollout_id)
        grouped[trajectory.group_id or trajectory.uid].append(trajectory)

    groups: list[GrpoGroup] = []
    for uid in sorted(grouped):
        items = grouped[uid]
        if strict and len(items) != rollout_n:
            raise ValueError(f"uid {uid!r} has {len(items)} trajectories, expected {rollout_n}")
        groups.append(GrpoGroup(uid=uid, trajectories=items))
    return GrpoBatch(groups=groups)


def summarize_grpo_batch(batch: GrpoBatch, *, variance_epsilon: float = 1e-12) -> dict[str, Any]:
    """Return diagnostics for grouped GRPO policy samples."""

    group_sizes = [len(group.trajectories) for group in batch.groups]
    reward_variance_per_group: dict[str, float] = {}
    zero_variance_groups = 0
    for group in batch.groups:
        rewards = [float(trajectory.reward) for trajectory in group.trajectories]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        reward_variance_per_group[group.uid] = variance
        if variance <= variance_epsilon:
            zero_variance_groups += 1

    trajectories = batch.trajectories
    num_write_transitions = sum(1 for trajectory in trajectories if int(trajectory.metadata.get("turn_index", 0)) == 0)
    num_rewrite_transitions = len(trajectories) - num_write_transitions
    variance_values = list(reward_variance_per_group.values())
    return {
        "group_size_mean": sum(group_sizes) / len(group_sizes) if group_sizes else 0.0,
        "group_size_max": max(group_sizes) if group_sizes else 0,
        "num_write_transitions": num_write_transitions,
        "num_rewrite_transitions": num_rewrite_transitions,
        "rewrite_ratio": num_rewrite_transitions / len(trajectories) if trajectories else 0.0,
        "reward_variance_per_group": reward_variance_per_group,
        "reward_variance_mean": sum(variance_values) / len(variance_values) if variance_values else 0.0,
        "zero_variance_group_ratio": zero_variance_groups / len(batch.groups) if batch.groups else 0.0,
    }
