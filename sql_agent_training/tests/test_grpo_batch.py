import pytest

from sql_agent_training.train.grpo_batch import TokenizedTrajectory, build_grpo_batch


def _trajectory(uid: str, rollout: int) -> TokenizedTrajectory:
    return TokenizedTrajectory(
        uid=uid,
        rollout_id=f"{uid}:{rollout}",
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        response_mask=[1, 1, 0],
        reward=float(rollout),
    )


def test_build_grpo_batch_groups_by_uid() -> None:
    batch = build_grpo_batch(
        [_trajectory("a", 0), _trajectory("b", 0), _trajectory("a", 1), _trajectory("b", 1)],
        rollout_n=2,
    )

    assert [group.uid for group in batch.groups] == ["a", "b"]
    assert batch.num_trajectories == 4
    assert [trajectory.rollout_id for trajectory in batch.groups[0].trajectories] == ["a:0", "a:1"]


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
