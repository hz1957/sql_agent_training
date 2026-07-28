import math
import random

import numpy as np
import torch

from sql_agent_training.train.verl_grpo_tree import (
    TreeNode,
    backup_tree_values,
    child_state_id,
    compute_grpo_tree_advantage,
    select_frontier_nodes,
    tree_slot_count,
)


def _node(
    node_id: str,
    *,
    parent_state_id: str = "parent",
    correct: bool = False,
    execution_ok: bool = False,
    checker_verdict: bool | None = False,
) -> TreeNode:
    return TreeNode(
        node_id=node_id,
        parent_state_id=parent_state_id,
        root_uid="task",
        turn_index=0,
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_text="SELECT 1",
        sql="SELECT 1",
        execution_ok=execution_ok,
        execution_feedback="rows=[(1,)]; row_count=1" if execution_ok else "no such table",
        checker_feedback="THE QUERY IS CORRECT." if checker_verdict else "THE QUERY IS INCORRECT.",
        checker_verdict=checker_verdict,
        correct=correct,
    )


def test_tree_slot_count_matches_bounded_frontier_design() -> None:
    assert tree_slot_count(branch_n=4, beam_size=4, max_turns=3) == 36
    assert tree_slot_count(branch_n=4, beam_size=2, max_turns=3) == 20


def test_backup_tree_values_uses_final_reward_mean_backup() -> None:
    direct = _node("direct", correct=True)
    repaired_parent = _node("needs_rewrite")
    failed = _node("failed")
    repaired_parent.children = [_node("rewrite_ok", correct=True), _node("rewrite_bad")]

    backup_tree_values([direct, repaired_parent, failed], gamma=0.9)

    assert direct.value == 1.0
    assert repaired_parent.value == 0.45
    assert failed.value == 0.0


def test_select_frontier_nodes_uses_proxy_not_gold_reward() -> None:
    bad = _node("bad", execution_ok=False, checker_verdict=False)
    executable = _node("exec", execution_ok=True, checker_verdict=False)
    checker_positive = _node("checker_positive", execution_ok=True, checker_verdict=True)

    selected = select_frontier_nodes(
        [bad, executable, checker_positive],
        beam_size=1,
        rng=random.Random(0),
        tau=0.01,
        epsilon_random=0.0,
    )

    assert [node.node_id for node in selected] == ["checker_positive"]


def test_child_state_id_depends_on_agent_visible_parent_state() -> None:
    first = child_state_id(
        root_uid="task",
        next_turn_index=1,
        sql="SELECT a FROM t",
        execution_feedback="rows=[]; row_count=0",
        checker_feedback="bad column",
    )
    second = child_state_id(
        root_uid="task",
        next_turn_index=1,
        sql="SELECT a FROM t",
        execution_feedback="rows=[]; row_count=0",
        checker_feedback="wrong aggregation",
    )

    assert first != second
    assert first.startswith("task::t1::")


def test_grpo_tree_advantage_groups_by_parent_state_and_ignores_dummy_rows() -> None:
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=torch.float32,
    )
    rewards = torch.zeros_like(response_mask)
    rewards[0, 1] = 1.0
    rewards[1, 1] = 0.9
    rewards[2, 1] = 0.0
    rewards[3, 1] = 0.1
    index = np.array(["parent"] * 4 + ["dummy"], dtype=object)

    advantages, returns = compute_grpo_tree_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
    )

    mean = 0.5
    std = math.sqrt(((1.0 - mean) ** 2 + (0.9 - mean) ** 2 + (0.0 - mean) ** 2 + (0.1 - mean) ** 2) / 4)
    expected = [(1.0 - mean) / std, (0.9 - mean) / std, (0.0 - mean) / std, (0.1 - mean) / std]
    for row, value in enumerate(expected):
        torch.testing.assert_close(advantages[row, :2], torch.full((2,), value), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(advantages[4], torch.zeros(3))
    torch.testing.assert_close(returns, advantages)
