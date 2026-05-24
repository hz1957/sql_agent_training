"""GRPO rollout grouping contracts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sql_agent_training.agent.trace_format import TokenizedTrajectory


@dataclass(frozen=True)
class GrpoGroup:
    """Rollout group for one original Spider sample."""

    uid: str
    trajectories: list[TokenizedTrajectory]


@dataclass(frozen=True)
class GrpoBatch:
    """A batch of grouped trajectories ready for GRPO advantage computation."""

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
    """Group tokenized trajectories by uid.

    Args:
        trajectories: Flat list of sampled trajectories.
        rollout_n: Expected number of independent rollouts per original sample.
        strict: If True, every uid must have exactly rollout_n trajectories.

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
        grouped[trajectory.uid].append(trajectory)

    groups: list[GrpoGroup] = []
    for uid in sorted(grouped):
        items = grouped[uid]
        if strict and len(items) != rollout_n:
            raise ValueError(f"uid {uid!r} has {len(items)} trajectories, expected {rollout_n}")
        groups.append(GrpoGroup(uid=uid, trajectories=items))
    return GrpoBatch(groups=groups)


def make_response_mask(*, generated_token_count: int, tool_token_count: int = 0) -> list[int]:
    """Build a simple response mask segment."""

    if generated_token_count < 0 or tool_token_count < 0:
        raise ValueError("token counts must be non-negative")
    return [1] * generated_token_count + [0] * tool_token_count
