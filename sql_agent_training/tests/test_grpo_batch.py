import pytest

from sql_agent_training.train.grpo_batch import TokenizedTrajectory, build_grpo_batch, summarize_grpo_batch


def _trajectory(uid: str, rollout: int) -> TokenizedTrajectory:
    return TokenizedTrajectory(
        uid=uid,
        rollout_id=f"{uid}:{rollout}",
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        response_mask=[1, 1, 0],
        reward=float(rollout),
    )


def _transition(uid: str, rollout: int, group_id: str, *, turn_index: int | None = None) -> TokenizedTrajectory:
    return TokenizedTrajectory(
        uid=uid,
        rollout_id=f"{uid}:{rollout}",
        prompt_ids=[1, 2],
        response_ids=[3],
        response_mask=[1],
        reward=float(rollout),
        group_id=group_id,
        metadata={"turn_index": rollout if turn_index is None else turn_index},
    )


def test_build_grpo_batch_groups_by_uid() -> None:
    batch = build_grpo_batch(
        [_trajectory("a", 0), _trajectory("b", 0), _trajectory("a", 1), _trajectory("b", 1)],
        rollout_n=2,
    )

    assert [group.uid for group in batch.groups] == ["a", "b"]
    assert batch.num_trajectories == 4
    assert [trajectory.rollout_id for trajectory in batch.groups[0].trajectories] == ["a:0", "a:1"]


def test_build_grpo_batch_can_group_by_explicit_group_id() -> None:
    batch = build_grpo_batch(
        [
            _transition("a", 0, "task-a"),
            _transition("a", 1, "task-a"),
            _transition("a", 2, "task-b"),
        ],
        rollout_n=2,
        strict=False,
    )

    assert [group.uid for group in batch.groups] == ["task-a", "task-b"]
    assert [trajectory.rollout_id for trajectory in batch.groups[0].trajectories] == ["a:0", "a:1"]
    assert [trajectory.rollout_id for trajectory in batch.groups[1].trajectories] == ["a:2"]


def test_summarize_grpo_batch_reports_transition_diagnostics() -> None:
    batch = build_grpo_batch(
        [
            _transition("a", 0, "task-a"),
            _transition("a", 1, "task-a"),
            _transition("b", 0, "task-b"),
            _transition("b", 2, "task-b-alt", turn_index=0),
        ],
        rollout_n=2,
        strict=False,
    )

    summary = summarize_grpo_batch(batch)

    assert summary["group_size_mean"] == pytest.approx(4 / 3)
    assert summary["group_size_max"] == 2
    assert summary["num_write_transitions"] == 3
    assert summary["num_rewrite_transitions"] == 1
    assert summary["rewrite_ratio"] == pytest.approx(0.25)
    assert summary["reward_variance_per_group"]["task-a"] == pytest.approx(0.25)
    assert summary["zero_variance_group_ratio"] == pytest.approx(2 / 3)


def test_build_grpo_batch_rejects_incomplete_group() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        build_grpo_batch([_trajectory("a", 0)], rollout_n=2)


def test_build_grpo_batch_rejects_duplicate_rollout_id() -> None:
    item = _trajectory("a", 0)
    with pytest.raises(ValueError, match="duplicate rollout_id"):
        build_grpo_batch([item, item], rollout_n=2)


def test_tokenized_trajectory_rejects_bad_mask_length() -> None:
    bad = TokenizedTrajectory(
        uid="a",
        rollout_id="a:0",
        prompt_ids=[1],
        response_ids=[2, 3],
        response_mask=[1],
        reward=0.0,
    )
    with pytest.raises(ValueError, match="same length"):
        bad.validate()
