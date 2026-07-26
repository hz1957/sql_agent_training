"""Tests for the Ray GRPO pipeline.

The ``dryrun`` tests use tiny scripted models and do not require real GPUs,
vLLM, or Ray workers. They exercise the core data-flow logic by calling worker
class methods directly (without Ray remote dispatch).

The ``ray`` marker tests require an actual Ray cluster with two GPUs.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sql_agent_training.agent.trace_format import TokenizedTrajectory
from sql_agent_training.train.grpo_batch import GrpoBatch, GrpoGroup, build_grpo_batch
from sql_agent_training.train.grpo_train import (
    GrpoLossConfig,
    GrpoTrainer,
    GrpoTrainingBatch,
    create_tiny_causal_lm,
)
from sql_agent_training.train.grpo_ray_pipeline import LearnerStepResult, ReferenceLogprobWorker, _role_num_gpus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trajectory(uid: str, rollout: int, response_ids: list[int], reward: float) -> TokenizedTrajectory:
    return TokenizedTrajectory(
        uid=uid,
        rollout_id=f"{uid}:{rollout}",
        prompt_ids=[1, 2],
        response_ids=response_ids,
        response_mask=[1] * len(response_ids),
        reward=reward,
        group_id=uid,
    )


def _tiny_batch() -> GrpoBatch:
    return build_grpo_batch(
        [
            _trajectory("a", 0, [3, 4], 0.0),
            _trajectory("a", 1, [5, 6], 1.0),
        ],
        rollout_n=2,
    )


def _make_tiny_trainer(device: str = "cpu") -> tuple[Any, Any, GrpoTrainer]:
    torch.manual_seed(42)
    policy = create_tiny_causal_lm(vocab_size=16, hidden_size=8)
    reference = create_tiny_causal_lm(vocab_size=16, hidden_size=8)
    reference.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    trainer = GrpoTrainer(
        policy,
        reference,
        optimizer,
        pad_token_id=0,
        loss_config=GrpoLossConfig(kl_beta=0.0),
        device=device,
    )
    return policy, reference, trainer


# ---------------------------------------------------------------------------
# Unit tests: prepare_batch_with_ref_logprobs (new method)
# ---------------------------------------------------------------------------


def test_prepare_batch_with_ref_logprobs_returns_correct_shape() -> None:
    """prepare_batch_with_ref_logprobs should produce the same shape as prepare_batch."""
    batch = _tiny_batch()
    _, reference, trainer = _make_tiny_trainer()

    # Normal prepare_batch
    normal = trainer.prepare_batch(batch)

    # Prepare with external ref logprobs (taken from a normal prepare to match shapes)
    prepared_with_ref = trainer.prepare_batch_with_ref_logprobs(batch, normal.reference_logprobs)

    assert prepared_with_ref.input_ids.shape == normal.input_ids.shape
    assert prepared_with_ref.response_mask.shape == normal.response_mask.shape
    assert prepared_with_ref.reference_logprobs.shape == normal.reference_logprobs.shape


def test_prepare_batch_with_ref_logprobs_uses_supplied_reference() -> None:
    """The supplied reference logprobs tensor should be passed through unchanged."""
    batch = _tiny_batch()
    _, _, trainer = _make_tiny_trainer()

    fake_ref = torch.full((2, 3), fill_value=-1.23)  # seq_len = 4, shifted = 3
    prepared = trainer.prepare_batch_with_ref_logprobs(batch, fake_ref)

    assert torch.allclose(prepared.reference_logprobs, fake_ref)


def test_prepare_batch_with_ref_logprobs_produces_trainable_batch() -> None:
    """A batch prepared with external ref logprobs should train without error."""
    batch = _tiny_batch()
    _, _, trainer = _make_tiny_trainer()

    normal = trainer.prepare_batch(batch)
    prepared = trainer.prepare_batch_with_ref_logprobs(batch, normal.reference_logprobs)
    metrics = trainer.train_prepared_batch(prepared)

    assert metrics.trainable_tokens > 0
    assert isinstance(metrics.loss, float)


# ---------------------------------------------------------------------------
# Unit tests: LearnerStepResult dataclass
# ---------------------------------------------------------------------------


def test_learner_step_result_stores_all_fields() -> None:
    result = LearnerStepResult(
        metrics={"loss": 0.5},
        lora_state_dict={"weight": torch.zeros(2)},
        optimizer_step=3,
        batch_stats={"groups": 2},
    )

    assert result.optimizer_step == 3
    assert result.metrics["loss"] == 0.5
    assert "weight" in result.lora_state_dict
    assert result.batch_stats["groups"] == 2


def test_learner_step_result_batch_stats_defaults_to_empty() -> None:
    result = LearnerStepResult(
        metrics={},
        lora_state_dict={},
        optimizer_step=0,
    )

    assert result.batch_stats == {}


def test_role_num_gpus_defaults_to_two_gpu_colocated_layout() -> None:
    config: dict[str, Any] = {"ray": {"reference_colocate_with_rollout": True}}

    assert _role_num_gpus(config, "rollout") == 0.5
    assert _role_num_gpus(config, "reference") == 0.5
    assert _role_num_gpus(config, "learner") == 1.0


def test_role_num_gpus_supports_three_gpu_separate_layout() -> None:
    config: dict[str, Any] = {
        "ray": {
            "reference_colocate_with_rollout": False,
            "rollout_num_gpus": 1,
            "reference_num_gpus": 1,
            "learner_num_gpus": 1,
        }
    }

    assert _role_num_gpus(config, "rollout") == 1.0
    assert _role_num_gpus(config, "reference") == 1.0
    assert _role_num_gpus(config, "learner") == 1.0


def _stub_reference_worker_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    tiny_reference = create_tiny_causal_lm(vocab_size=16, hidden_size=8)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            return tiny_reference

    fake_transformers = types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


@pytest.mark.dryrun
def test_reference_logprob_worker_computes_logprobs_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_reference_worker_deps(monkeypatch)

    worker = ReferenceLogprobWorker(
        {
            "model": {"path": "fake/path", "torch_dtype": "none"},
            "ray": {"reference_device": "cpu"},
        }
    )
    batch = _tiny_batch()

    ref_logprobs = worker.compute_ref_logprobs(batch)

    assert set(ref_logprobs) == {trajectory.rollout_id for trajectory in batch.trajectories}
    for trajectory in batch.trajectories:
        assert len(ref_logprobs[trajectory.rollout_id]) == len(trajectory.prompt_ids) + len(trajectory.response_ids) - 1


# ---------------------------------------------------------------------------
# Dryrun integration: worker logic (without Ray remote, no real GPU)
# ---------------------------------------------------------------------------


def _stub_learner_worker_deps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Patch heavy imports so LearnerWorker can be instantiated on CPU with tiny model."""
    import sql_agent_training.train.grpo_ray_pipeline as pipeline_mod

    tiny_policy = create_tiny_causal_lm(vocab_size=16, hidden_size=8)
    tiny_tokenizer = types.SimpleNamespace(
        pad_token_id=0,
        eos_token_id=1,
        pad_token=None,
    )

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            return tiny_tokenizer

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            return tiny_policy

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return {"policy": tiny_policy, "tokenizer": tiny_tokenizer}


@pytest.mark.dryrun
def test_learner_worker_train_step_dryrun(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """LearnerWorker.train_step should complete end-to-end on CPU with tiny model."""
    stubs = _stub_learner_worker_deps(monkeypatch, tmp_path)

    from sql_agent_training.train.grpo_ray_pipeline import LearnerWorker

    config: dict[str, Any] = {
        "model": {"path": "fake/path", "torch_dtype": "none"},
        "ray": {"learner_device": "cpu"},
        "training": {
            "seed": 0,
            "learning_rate": 0.01,
            "clip_epsilon": 0.2,
            "kl_beta": 0.0,
            "max_grad_norm": 1.0,
            "logprob_micro_batch_size": 1,
            "update_epochs": 1,
        },
        "tokenizer": {"kind": "whitespace"},
    }

    worker = LearnerWorker(config)
    batch = _tiny_batch()

    # Build fake ref_logprobs aligned to the batch shapes
    normal_prepared = worker._trainer.prepare_batch(batch)
    seq_len_minus_1 = normal_prepared.input_ids.shape[1] - 1
    num_traj = normal_prepared.input_ids.shape[0]
    ref_logprobs = {
        traj.rollout_id: [-0.5] * (len(traj.prompt_ids) + len(traj.response_ids) - 1) for traj in batch.trajectories
    }

    result = worker.train_step(batch, ref_logprobs)

    assert isinstance(result, LearnerStepResult)
    assert result.optimizer_step == 1
    assert result.metrics["trainable_tokens"] > 0
    assert isinstance(result.lora_state_dict, dict)


# ---------------------------------------------------------------------------
# Integration: grpo_ray_pipeline import and train_grpo_ray signature
# ---------------------------------------------------------------------------


def test_train_grpo_ray_is_importable() -> None:
    """train_grpo_ray should be importable without starting Ray."""
    from sql_agent_training.train.grpo_ray_pipeline import train_grpo_ray  # noqa: F401

    assert callable(train_grpo_ray)


def test_grpo_train_main_accepts_ray_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--ray flag should route to train_grpo_ray instead of train_grpo_from_config."""
    import sql_agent_training.train.grpo_train as grpo_train_mod

    called_with: list[dict[str, Any]] = []

    def fake_train_grpo_ray(config: dict[str, Any]) -> dict[str, Any]:
        called_with.append(config)
        return {"rank": 0, "steps": 1}

    monkeypatch.setattr(
        "sql_agent_training.train.grpo_ray_pipeline.train_grpo_ray",
        fake_train_grpo_ray,
        raising=False,
    )
    # Patch the import inside grpo_train.main to pick up our fake
    import importlib

    fake_pipeline = types.SimpleNamespace(train_grpo_ray=fake_train_grpo_ray)
    monkeypatch.setitem(sys.modules, "sql_agent_training.train.grpo_ray_pipeline", fake_pipeline)

    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("dry_run: true\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["grpo_train", "--config", str(config_file), "--ray"])

    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        grpo_train_mod.main()

    assert len(called_with) == 1
    assert called_with[0].get("dry_run") is True
