from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sql_agent_training.agent.trace_format import TokenizedTrajectory
from sql_agent_training.train.grpo_batch import build_grpo_batch
from sql_agent_training.train.grpo_trainer import (
    GrpoLossConfig,
    GrpoTrainer,
    build_training_tensors,
    compute_group_advantages,
    create_tiny_causal_lm,
    train_grpo_from_config,
)


def _trajectory(uid: str, rollout: int, response_ids: list[int], reward: float) -> TokenizedTrajectory:
    return TokenizedTrajectory(
        uid=uid,
        rollout_id=f"{uid}:{rollout}",
        prompt_ids=[1, 2],
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        reward=reward,
    )


def test_compute_group_advantages_normalizes_within_uid() -> None:
    batch = build_grpo_batch(
        [
            _trajectory("a", 0, [3], 0.0),
            _trajectory("a", 1, [4], 1.0),
            _trajectory("b", 0, [5], 0.5),
            _trajectory("b", 1, [6], 0.5),
        ],
        rollout_n=2,
    )

    advantages = compute_group_advantages(batch)

    assert advantages["a:0"] == pytest.approx(-1.0, abs=1e-5)
    assert advantages["a:1"] == pytest.approx(1.0, abs=1e-5)
    assert advantages["b:0"] == 0.0
    assert advantages["b:1"] == 0.0


def test_build_training_tensors_aligns_response_mask_to_shifted_labels() -> None:
    batch = build_grpo_batch(
        [_trajectory("a", 0, [3, 4], 0.0), _trajectory("a", 1, [5], 1.0)],
        rollout_n=2,
    )
    tensors = build_training_tensors(
        batch,
        advantages={"a:0": -1.0, "a:1": 1.0},
        pad_token_id=0,
        device="cpu",
    )

    assert tensors["input_ids"].tolist() == [[1, 2, 3, 4], [1, 2, 5, 0]]
    assert tensors["response_mask"].tolist() == [[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]
    assert tensors["advantages"].tolist() == [-1.0, 1.0]


def test_grpo_trainer_updates_tiny_policy_weights() -> None:
    batch = build_grpo_batch(
        [_trajectory("a", 0, [3, 4], 0.0), _trajectory("a", 1, [5, 6], 1.0)],
        rollout_n=2,
    )
    torch.manual_seed(0)
    policy = create_tiny_causal_lm(vocab_size=8, hidden_size=8)
    reference = create_tiny_causal_lm(vocab_size=8, hidden_size=8)
    reference.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    trainer = GrpoTrainer(
        policy,
        reference,
        optimizer,
        pad_token_id=0,
        loss_config=GrpoLossConfig(kl_beta=0.0),
    )
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}

    prepared = trainer.prepare_batch(batch)
    metrics = trainer.train_prepared_batch(prepared)

    assert metrics.trainable_tokens == 4
    assert metrics.mean_reward == pytest.approx(0.5)
    assert any(not torch.equal(before[name], parameter) for name, parameter in policy.named_parameters())


def test_train_grpo_from_config_runs_tiny_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    metrics_json = checkpoint_dir / "metrics.json"

    summary = train_grpo_from_config(
        {
            "dry_run": True,
            "model": {"backend": "tiny", "hidden_size": 8},
            "tokenizer": {"kind": "whitespace"},
            "rollout": {
                "n": 2,
                "max_turns": 1,
                "scripted_responses": ["SELECT COUNT(*) FROM Singer", "SELECT Name FROM Singer"],
            },
            "training": {"seed": 0, "device": "cpu", "max_steps": 1, "learning_rate": 0.01},
            "output": {"checkpoint_dir": str(checkpoint_dir), "metrics_json": str(metrics_json)},
        }
    )

    assert summary["steps"] == 1
    assert summary["trajectories"] == 2
    assert summary["mean_reward"] == pytest.approx(0.5)
    assert (checkpoint_dir / "tiny_policy.pt").exists()
    assert metrics_json.exists()
