import numpy as np
import pytest
import torch

from sql_agent_training.train.verl_grpo_multi_step import (
    build_multi_step_rm_scores,
    compute_grpo_multi_step_advantage,
    response_mask_blocks,
)
from sql_agent_training.train.verl_sql_agent_loop import _build_multi_step_turn_rewards


def test_chain_final_turn_rewards_discount_success() -> None:
    rewards = _build_multi_step_turn_rewards(
        turn_records=[
            {"turn_index": 0, "response_start": 0, "response_end": 2, "executable": True},
            {"turn_index": 1, "response_start": 7, "response_end": 10, "executable": True},
            {"turn_index": 2, "response_start": 15, "response_end": 18, "executable": True},
        ],
        final_reward=1.0,
        success_turn_index=2,
        reward_scheme="chain_final",
        gamma=0.9,
        executable_fallback_beta=0.1,
    )

    assert [item["reward"] for item in rewards] == pytest.approx([0.81, 0.9, 1.0])


def test_chain_executable_turn_rewards_use_beta_only_on_failed_trajectory() -> None:
    rewards = _build_multi_step_turn_rewards(
        turn_records=[
            {"turn_index": 0, "response_start": 0, "response_end": 2, "executable": False},
            {"turn_index": 1, "response_start": 7, "response_end": 10, "executable": True},
            {"turn_index": 2, "response_start": 15, "response_end": 18, "executable": True},
        ],
        final_reward=0.0,
        success_turn_index=None,
        reward_scheme="chain_executable",
        gamma=0.9,
        executable_fallback_beta=0.1,
    )

    assert [item["reward"] for item in rewards] == pytest.approx([0.0, 0.1, 0.1])


def test_build_multi_step_rm_scores_places_rewards_on_sql_block_end() -> None:
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
        ],
        dtype=torch.long,
    )
    inputs = [
        {
            "multi_step_turn_rewards": [
                {"response_start": 0, "response_end": 1, "reward": 0.9},
                {"response_start": 4, "response_end": 5, "reward": 1.0},
            ]
        },
        {
            "multi_step_turn_rewards": [
                {"response_start": 0, "response_end": 1, "reward": 0.1},
                {"response_start": 4, "response_end": 5, "reward": 0.0},
            ]
        },
    ]

    rm_scores = build_multi_step_rm_scores(input_extra_fields=inputs, response_mask=response_mask)

    assert torch.allclose(
        rm_scores,
        torch.tensor(
            [
                [0.0, 0.9, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_response_mask_blocks() -> None:
    assert response_mask_blocks(torch.tensor([0, 1, 1, 0, 1, 0, 1, 1])) == [(1, 2), (4, 4), (6, 7)]


def test_grpo_multi_step_advantage_normalizes_by_task_and_turn() -> None:
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    token_level_rewards = torch.tensor(
        [
            [0, 1.0, 0, 0, 0, 0],
            [0, 0.9, 0, 0, 1.0, 0],
            [0, 0.0, 0, 0, 0.1, 0],
            [0, 0.1, 0, 0, 0.0, 0],
        ],
        dtype=torch.float32,
    )

    advantages, returns = compute_grpo_multi_step_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=np.array(["task_001", "task_001", "task_001", "task_001"], dtype=object),
    )

    # Turn 0 rewards [1, 0.9, 0, 0.1], population std ~= 0.45277.
    assert advantages[0, 0].item() == pytest.approx(1.1043, abs=1e-3)
    assert advantages[1, 0].item() == pytest.approx(0.8835, abs=1e-3)
    assert advantages[2, 0].item() == pytest.approx(-1.1043, abs=1e-3)
    assert advantages[3, 0].item() == pytest.approx(-0.8835, abs=1e-3)

    # Turn 1 rewards [1, 0.1, 0], excluding the rollout that stopped at turn 0.
    assert advantages[1, 3].item() == pytest.approx(1.4084, abs=1e-3)
    assert advantages[2, 3].item() == pytest.approx(-0.5930, abs=1e-3)
    assert advantages[3, 3].item() == pytest.approx(-0.8154, abs=1e-3)
    assert torch.equal(advantages, returns)
