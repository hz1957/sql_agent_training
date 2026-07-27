import pytest

from sql_agent_training.train.verl_grpo_config import VerlGrpoLaunchConfig, validate_runtime_dependencies


def _config(**overrides) -> VerlGrpoLaunchConfig:
    values = {
        "train_batch_size": 1,
        "ppo_mini_batch_size": 1,
        "ppo_micro_batch_size_per_gpu": 1,
        "n_gpus_per_node": 2,
        "rollout_n": 2,
        "rollout_tp": 2,
        "rollout_pp": 1,
        "model_num_attention_heads": 40,
        "log_prob_use_dynamic_bsz": False,
        "log_prob_micro_batch_size_per_gpu": 1,
    }
    values.update(overrides)
    return VerlGrpoLaunchConfig(**values)


def test_h100_smoke_batch_config_validates() -> None:
    _config().validate()


def test_rejects_ppo_mini_batch_larger_than_train_batch() -> None:
    with pytest.raises(ValueError, match="ppo_mini_batch_size"):
        _config(ppo_mini_batch_size=2).validate()


def test_rejects_missing_logprob_micro_batch_when_not_dynamic() -> None:
    with pytest.raises(ValueError, match="log_prob_micro_batch_size_per_gpu"):
        _config(log_prob_micro_batch_size_per_gpu=None).validate()


def test_rejects_rollout_batch_smaller_than_gpu_count() -> None:
    with pytest.raises(ValueError, match="train_batch_size \\* rollout_n"):
        _config(rollout_n=1).validate()


def test_dynamic_logprob_does_not_require_fixed_micro_batch() -> None:
    _config(log_prob_use_dynamic_bsz=True, log_prob_micro_batch_size_per_gpu=None).validate()


def test_rejects_tensor_parallel_size_that_does_not_divide_attention_heads() -> None:
    with pytest.raises(ValueError, match="rollout_tp"):
        _config(rollout_tp=3).validate()


def test_flash_attn_runtime_check_can_be_disabled() -> None:
    validate_runtime_dependencies(require_flash_attn=False)


def test_flash_attn_runtime_check_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.import_module", fail_import)
    with pytest.raises(RuntimeError, match="flash_attn.bert_padding"):
        validate_runtime_dependencies(require_flash_attn=True)
