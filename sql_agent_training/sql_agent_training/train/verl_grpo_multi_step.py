"""Multi-step GRPO support for verl SQL-agent rollouts.

This module keeps one verl row per trajectory, writes per-SQL-turn rewards into
the token-level ``rm_scores`` tensor, and registers an advantage estimator that
normalizes rewards by ``(uid, turn_index)``.
"""

from __future__ import annotations

from collections import defaultdict
import os
from typing import Any

import numpy as np
import torch

MULTI_STEP_TURN_REWARDS_FIELD = "multi_step_turn_rewards"
ADV_ESTIMATOR_NAME = "grpo_multi_step"


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except Exception:
            pass
    return getattr(config, key, default)


def _index_value(index: Any, row: int) -> str:
    if index is None:
        return str(row)
    value = index[row]
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except Exception:
            pass
    return str(value)


def response_mask_blocks(mask_row: torch.Tensor) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` spans for contiguous trainable SQL blocks."""

    values = (mask_row.detach().cpu() > 0).tolist()
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_trainable in enumerate(values):
        if is_trainable and start is None:
            start = index
        elif not is_trainable and start is not None:
            blocks.append((start, index - 1))
            start = None
    if start is not None:
        blocks.append((start, len(values) - 1))
    return blocks


def build_multi_step_rm_scores(
    *,
    input_extra_fields: list[dict[str, Any]],
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Build token-level reward scores from AgentLoop extra fields."""

    rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
    for row, extra_fields in enumerate(input_extra_fields):
        turn_rewards = extra_fields.get(MULTI_STEP_TURN_REWARDS_FIELD) or []
        for record in turn_rewards:
            start = max(0, int(record.get("response_start", 0)))
            end = min(response_mask.size(1) - 1, int(record.get("response_end", start)))
            if end < start:
                continue
            segment = response_mask[row, start : end + 1] > 0
            if not bool(segment.any().item()):
                continue
            relative_positions = torch.nonzero(segment, as_tuple=False).flatten()
            reward_position = start + int(relative_positions[-1].item())
            rm_scores[row, reward_position] = float(record.get("reward", 0.0))
    return rm_scores


def _group_stats(
    rewards_by_group: dict[tuple[str, int], list[torch.Tensor]],
    *,
    epsilon: float,
) -> dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, bool]]:
    stats: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, bool]] = {}
    for group_key, rewards in rewards_by_group.items():
        values = torch.stack(rewards)
        if len(rewards) < 2:
            stats[group_key] = (values.mean(), torch.zeros_like(values.mean()), False)
            continue
        mean = values.mean()
        std = values.std(unbiased=False)
        stats[group_key] = (mean, std, bool((std > epsilon).item()))
    return stats


def compute_grpo_multi_step_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | None = None,
    config: Any = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute turn-level GRPO advantages and broadcast them to SQL tokens.

    Each contiguous ``response_mask == 1`` block is treated as one SQL turn. The
    reward for that turn is the sum of ``token_level_rewards`` inside the block,
    allowing sparse rewards at the final SQL token while preserving zero-reward
    turns. Groups with fewer than two samples or zero reward variance produce a
    zero advantage.
    """

    epsilon = float(_cfg_get(config, "multi_step_epsilon", epsilon))
    norm_adv_by_std_in_grpo = bool(
        _cfg_get(config, "norm_adv_by_std_in_grpo", norm_adv_by_std_in_grpo)
    )

    with torch.no_grad():
        rewards_by_group: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
        row_turns: list[list[tuple[int, int, int, torch.Tensor]]] = []

        for row in range(response_mask.size(0)):
            uid = _index_value(index, row)
            turns: list[tuple[int, int, int, torch.Tensor]] = []
            for turn_index, (start, end) in enumerate(response_mask_blocks(response_mask[row])):
                reward = token_level_rewards[row, start : end + 1].sum()
                group_key = (uid, turn_index)
                rewards_by_group[group_key].append(reward)
                turns.append((turn_index, start, end, reward))
            row_turns.append(turns)

        stats = _group_stats(rewards_by_group, epsilon=epsilon)
        advantages = torch.zeros_like(token_level_rewards, dtype=torch.float32)

        for row, turns in enumerate(row_turns):
            uid = _index_value(index, row)
            for turn_index, start, end, reward in turns:
                mean, std, usable = stats[(uid, turn_index)]
                if not usable:
                    continue
                if norm_adv_by_std_in_grpo:
                    advantage = (reward - mean) / (std + epsilon)
                else:
                    advantage = reward - mean
                advantages[row, start : end + 1] = advantage.to(advantages.dtype)

        advantages = advantages * response_mask.to(dtype=advantages.dtype)
        return advantages, advantages


try:  # pragma: no cover - exercised in the verl runtime.
    import ray
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager as _VerlAgentLoopManager
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker as _VerlAgentLoopWorker
    from verl.trainer.ppo.core_algos import register_adv_est
except ImportError:  # pragma: no cover - unit tests can import without verl.
    ray = None  # type: ignore[assignment]
    _VerlAgentLoopManager = None  # type: ignore[assignment]
    _VerlAgentLoopWorker = None  # type: ignore[assignment]

    def register_adv_est(_: str):
        def decorator(fn):
            return fn

        return decorator


register_adv_est(ADV_ESTIMATOR_NAME)(compute_grpo_multi_step_advantage)


if _VerlAgentLoopWorker is not None:

    class MultiStepRewardAgentLoopWorker(_VerlAgentLoopWorker):
        """AgentLoopWorker that replaces scalar terminal rewards with turn rewards."""

        def _postprocess(self, inputs, input_non_tensor_batch=None, **kwargs):  # type: ignore[no-untyped-def]
            output = super()._postprocess(
                inputs,
                input_non_tensor_batch=input_non_tensor_batch,
                **kwargs,
            )
            scheme = str(
                _cfg_get(
                    self.rollout_config.agent,
                    "reward_scheme",
                    os.environ.get("GRPO_REWARD_SCHEME", "outcome"),
                )
            ).lower().replace("-", "_")
            if scheme not in {"chain_final", "chain_executable"}:
                return output

            input_extra_fields = [input_item.extra_fields for input_item in inputs]
            output.batch["rm_scores"] = build_multi_step_rm_scores(
                input_extra_fields=input_extra_fields,
                response_mask=output.batch["response_mask"],
            )
            output.meta_info.setdefault("reward_extra_keys", [])
            return output


    class MultiStepRewardAgentLoopManager(_VerlAgentLoopManager):
        """Use the custom worker while keeping verl's rollout manager behavior."""

        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            super().__init__(*args, **kwargs)
            self.agent_loop_workers_class = ray.remote(MultiStepRewardAgentLoopWorker)

else:

    class MultiStepRewardAgentLoopWorker:  # pragma: no cover - dependency fallback.
        pass


    class MultiStepRewardAgentLoopManager:  # pragma: no cover - dependency fallback.
        pass
