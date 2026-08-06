import asyncio
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sql_agent_training.train.verl_grpo_tree import (
    TreeNode,
    TreeRewardAgentLoopWorker,
    TreeSqlAgentLoop,
    backup_tree_values,
    child_state_id,
    compute_grpo_tree_advantage,
    select_frontier_nodes,
    stable_root_uid_from_batch,
    stable_root_uid_from_prompt_ids,
    tree_request_seed,
    tree_slot_count,
    tree_workload_fingerprint,
)
from sql_agent_training.agent.sql_agent_loop import _irrecoverable_sqlite_error_reason
from sql_agent_training.train.verl_sql_agent_loop import _normalize_reward_scheme


def _node(
    node_id: str,
    *,
    parent_state_id: str = "parent",
    correct: bool = False,
    execution_ok: bool = False,
    checker_verdict: bool | None = False,
    prune_reason: str | None = None,
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
        prune_reason=prune_reason,
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


def test_backup_tree_values_does_not_clip_success_signal_with_executable_fallback() -> None:
    executable_leaf = _node("executable_leaf", execution_ok=True)
    failed_leaf = _node("failed_leaf", execution_ok=False)
    repaired_parent = _node("repaired_parent", execution_ok=False)
    repaired_parent.children = [_node("rewrite_exec", execution_ok=True), _node("rewrite_bad", execution_ok=False)]

    backup_tree_values(
        [executable_leaf, failed_leaf, repaired_parent],
        gamma=0.9,
        executable_fallback_beta=0.1,
    )

    assert executable_leaf.value == 0.0
    assert failed_leaf.value == 0.0
    assert math.isclose(repaired_parent.value, 0.0)


def test_tree_reward_scheme_aliases() -> None:
    assert _normalize_reward_scheme("s3") == "tree_final"
    assert _normalize_reward_scheme("tree-final") == "tree_final"
    assert _normalize_reward_scheme("s4") == "tree_executable"
    assert _normalize_reward_scheme("tree-executable") == "tree_executable"


def test_tree_worker_init_kwargs_accepts_new_verl_llm_client() -> None:
    if not hasattr(TreeRewardAgentLoopWorker, "_tree_loop_init_kwargs"):
        pytest.skip("verl AgentLoopWorker is not installed")

    worker = object.__new__(TreeRewardAgentLoopWorker)
    worker.config = SimpleNamespace(data={})
    worker.llm_client = object()
    worker.tokenizer = object()
    worker.processor = None
    worker.dataset_cls = object

    kwargs = worker._tree_loop_init_kwargs()

    assert kwargs["server_manager"] is worker.llm_client
    assert kwargs["tokenizer"] is worker.tokenizer


def test_tree_worker_root_kwargs_filters_run_tree_reserved_fields() -> None:
    if not hasattr(TreeRewardAgentLoopWorker, "_tree_root_kwargs"):
        pytest.skip("verl AgentLoopWorker is not installed")

    worker = object.__new__(TreeRewardAgentLoopWorker)
    batch = SimpleNamespace(
        non_tensor_batch={
            "raw_prompt": np.array(["prompt"], dtype=object),
            "extra_info": np.array([{"uid": "task"}], dtype=object),
            "priority": np.array([7], dtype=object),
            "root_uid": np.array(["reserved"], dtype=object),
            "slot_count": np.array([20], dtype=object),
        }
    )

    kwargs = worker._tree_root_kwargs(batch, 0)

    assert kwargs == {"raw_prompt": "prompt", "extra_info": {"uid": "task"}}


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


def test_irrecoverable_sqlite_error_reason_is_conservative() -> None:
    schema_prompt = "Database: music\n- Singer(Name)\n- Album(Title)"

    assert _irrecoverable_sqlite_error_reason("no such table: Phantom", schema_prompt) == "missing_table:Phantom"
    assert _irrecoverable_sqlite_error_reason("Table 'Phantom' doesn't exist", schema_prompt) == (
        "missing_table:Phantom"
    )
    assert _irrecoverable_sqlite_error_reason("interrupted", schema_prompt) == "timeout_or_interrupted"
    assert _irrecoverable_sqlite_error_reason("out of memory", schema_prompt) == "memory_limit"
    assert _irrecoverable_sqlite_error_reason("no such table: Singers", schema_prompt) is None


def test_parent_state_prunes_checker_terminal_and_severe_error_not_gold_reward() -> None:
    async def run_case() -> None:
        loop = object.__new__(TreeSqlAgentLoop)
        loop.max_turns = 3
        loop.prune_on_terminal_proxy = True

        async def encode(_: str, *, remove_system_prompt: bool) -> list[int]:
            del remove_system_prompt
            return [101, 102]

        loop._encode_user_prompt = encode
        fields = {"question": "List names.", "schema_prompt": "Database: music\n- Singer(Name)"}

        gold_correct_checker_rejected = _node(
            "gold_correct_checker_rejected",
            correct=True,
            execution_ok=True,
            checker_verdict=False,
        )
        assert (
            await loop._parent_state_for_node(
                fields=fields,
                root_uid="task",
                node=gold_correct_checker_rejected,
            )
            is not None
        )

        checker_terminal = _node("checker_terminal", correct=False, execution_ok=True, checker_verdict=True)
        assert (
            await loop._parent_state_for_node(fields=fields, root_uid="task", node=checker_terminal)
        ) is None

        severe_error = _node(
            "severe_error",
            correct=False,
            execution_ok=False,
            checker_verdict=False,
            prune_reason="missing_table:Phantom",
        )
        assert await loop._parent_state_for_node(fields=fields, root_uid="task", node=severe_error) is None

    asyncio.run(run_case())


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


def test_tree_request_seed_is_stable_and_request_local() -> None:
    first = tree_request_seed(
        root_uid="task",
        parent_state_id="task::root",
        child_index=0,
        request_kind="candidate",
    )
    repeated = tree_request_seed(
        root_uid="task",
        parent_state_id="task::root",
        child_index=0,
        request_kind="candidate",
    )
    sibling = tree_request_seed(
        root_uid="task",
        parent_state_id="task::root",
        child_index=1,
        request_kind="candidate",
    )
    checker = tree_request_seed(
        root_uid="task",
        parent_state_id="task::root",
        child_index=0,
        request_kind="checker",
    )

    assert first == repeated
    assert len({first, sibling, checker}) == 3
    assert 0 <= first < 2**32


def test_stable_root_uid_uses_dataset_uid_when_available() -> None:
    batch = SimpleNamespace(
        batch={"input_ids": torch.tensor([[1, 2, 0]])},
        non_tensor_batch={"uid": np.array(["task-7"], dtype=object)},
    )

    assert stable_root_uid_from_batch(batch, 0) == "task-7"


def test_stable_root_uid_falls_back_to_prompt_tokens() -> None:
    first = stable_root_uid_from_prompt_ids(torch.tensor([0, 11, 12, 0]), torch.tensor([0, 1, 1, 0]))
    repeated = stable_root_uid_from_prompt_ids(np.array([11, 12]))
    changed = stable_root_uid_from_prompt_ids([11, 13])
    batch = SimpleNamespace(
        batch={
            "input_ids": torch.tensor([[0, 11, 12, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 0]]),
        },
        non_tensor_batch={},
    )

    assert first == repeated
    assert stable_root_uid_from_batch(batch, 0) == first
    assert first.startswith("prompt::")
    assert first != changed


def test_tree_workload_fingerprint_is_order_independent_and_token_exact() -> None:
    records = [
        {"request_key": "candidate-1", "prompt_ids": [1, 2], "response_ids": [3]},
        {"request_key": "checker-1", "prompt_ids": [1, 2, 4], "response_ids": [5, 6]},
    ]

    first = tree_workload_fingerprint(records)
    reordered = tree_workload_fingerprint(list(reversed(records)))
    changed = tree_workload_fingerprint(
        [
            records[0],
            {"request_key": "checker-1", "prompt_ids": [1, 2, 9], "response_ids": [5, 6]},
        ]
    )

    assert first == reordered
    assert first["requests"] == 2
    assert first["prompt_tokens"] == 5
    assert first["response_tokens"] == 3
    assert first["prompt_sha256"] != changed["prompt_sha256"]
    assert first["response_sha256"] == changed["response_sha256"]
    assert first["request_sha256"] != changed["request_sha256"]


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


def test_tree_advantage_uses_correctness_before_executable_fallback() -> None:
    response_mask = torch.tensor(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=torch.float32,
    )
    rewards = torch.zeros_like(response_mask)
    rewards[0, 1] = 0.05
    rewards[1, 1] = 0.10
    index = np.array(["parent", "parent"], dtype=object)
    executable = np.array([True, False], dtype=object)

    advantages, _ = compute_grpo_tree_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        executable=executable,
        executable_fallback_weight=0.1,
    )

    assert advantages[0, 0] < 0
    assert advantages[1, 0] > 0
    torch.testing.assert_close(advantages[0], torch.full((2,), -1.0), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(advantages[1], torch.full((2,), 1.0), rtol=1e-4, atol=1e-4)


def test_tree_advantage_scales_executable_fallback_after_normalization() -> None:
    response_mask = torch.tensor(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=torch.float32,
    )
    rewards = torch.zeros_like(response_mask)
    index = np.array(["parent", "parent"], dtype=object)
    executable = np.array([True, False], dtype=object)

    advantages, _ = compute_grpo_tree_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        executable=executable,
        executable_fallback_weight=0.1,
    )

    torch.testing.assert_close(advantages[0], torch.full((2,), 0.1), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(advantages[1], torch.full((2,), -0.1), rtol=1e-4, atol=1e-4)


def test_tree_advantage_skips_all_failed_groups_without_executability_contrast() -> None:
    response_mask = torch.tensor(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=torch.float32,
    )
    rewards = torch.zeros_like(response_mask)
    index = np.array(["parent", "parent"], dtype=object)
    executable = np.array([True, True], dtype=object)

    advantages, _ = compute_grpo_tree_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
        executable=executable,
        executable_fallback_weight=0.1,
    )

    torch.testing.assert_close(advantages, torch.zeros_like(response_mask))
