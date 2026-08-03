from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sql_agent_training.train.verl_layered_summon import (
    LayeredSummonActorRolloutRefWorker,
    _qualify_fsdp_param_name,
    _wrap_update_weights,
    layered_summon_lora_params_fixed,
    patch_verl_layered_summon,
)


class _FakeTensor:
    def __init__(self, elements: int) -> None:
        self.elements = elements

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self


def test_qualify_fsdp_param_name_uses_lora_unit_path() -> None:
    assert (
        _qualify_fsdp_param_name(
            "_fsdp_wrapped_module.base_model.model.layers.0.q_proj.lora_A.default",
            "_fsdp_wrapped_module.weight",
        )
        == "base_model.model.layers.0.q_proj.lora_A.default.weight"
    )


def test_fixed_layered_summon_collects_lora_leaf_units(monkeypatch: Any) -> None:
    import peft.utils.save_and_load as peft_save
    import verl.utils.fsdp_utils as fsdp_utils

    class FakeLeaf:
        def __init__(self, tensor: _FakeTensor) -> None:
            self.tensor = tensor
            self._is_root = False

        def named_modules(self) -> list[tuple[str, Any]]:
            return [("", self)]

        def named_parameters(self) -> list[tuple[str, _FakeTensor]]:
            return [("_fsdp_wrapped_module.weight", self.tensor)]

    class FakeRoot:
        def __init__(self) -> None:
            self._fsdp_wrapped_module = type("FakePeftModel", (), {})()
            self.leaf_a = FakeLeaf(_FakeTensor(8))
            self.leaf_b = FakeLeaf(_FakeTensor(12))

        def named_modules(self) -> list[tuple[str, Any]]:
            prefix = "_fsdp_wrapped_module.base_model.model.layers.0.q_proj"
            return [
                ("", self),
                (f"{prefix}.lora_A.default", self.leaf_a),
                (f"{prefix}.lora_B.default", self.leaf_b),
            ]

    class FakeSummon:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: Any) -> None:
            return None

    root = FakeRoot()
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda _module: 1)
    monkeypatch.setattr(fsdp_utils.FSDP, "summon_full_params", lambda *_args, **_kwargs: FakeSummon())
    monkeypatch.setattr(
        fsdp_utils,
        "get_torch_device",
        lambda: type("FakeDevice", (), {"empty_cache": staticmethod(lambda: None)})(),
    )

    def fake_get_peft_state(_model: Any, *, state_dict: dict[str, Any]) -> dict[str, Any]:
        return {
            name.replace(".default", ""): tensor
            for name, tensor in state_dict.items()
            if "lora_" in name and ".default" in name
        }

    monkeypatch.setattr(peft_save, "get_peft_model_state_dict", fake_get_peft_state)

    params = layered_summon_lora_params_fixed(root)

    assert list(params) == [
        "base_model.model.layers.0.q_proj.lora_A.weight",
        "base_model.model.layers.0.q_proj.lora_B.weight",
    ]
    assert root.leaf_a._is_root is False
    assert root.leaf_b._is_root is False


def test_fixed_layered_summon_rejects_empty_collection() -> None:
    class FakeRoot:
        _fsdp_wrapped_module = type("FakePeftModel", (), {})()

        def named_modules(self) -> list[tuple[str, Any]]:
            return [("", self)]

    with pytest.raises(RuntimeError, match="refusing verl's full-summon fallback"):
        layered_summon_lora_params_fixed(FakeRoot())


def test_update_weights_wrapper_installs_and_restores_collector(monkeypatch: Any) -> None:
    import verl.utils.fsdp_utils as fsdp_utils

    original_collector = fsdp_utils.layered_summon_lora_params
    events: list[str] = []
    monkeypatch.setattr(
        fsdp_utils,
        "get_torch_device",
        lambda: type("FakeDevice", (), {"empty_cache": staticmethod(lambda: events.append("empty_cache"))})(),
    )

    class FakeWorker:
        async def update_weights(self) -> str:
            assert fsdp_utils.layered_summon_lora_params is layered_summon_lora_params_fixed
            events.append("update_weights")
            return "ok"

    assert _wrap_update_weights(FakeWorker)
    assert asyncio.run(FakeWorker().update_weights()) == "ok"
    assert events == ["empty_cache", "update_weights"]
    assert fsdp_utils.layered_summon_lora_params is original_collector


def test_patch_selects_importable_worker_subclass(monkeypatch: Any) -> None:
    import verl.workers.engine_workers as engine_workers

    original = engine_workers.ActorRolloutRefWorker
    monkeypatch.setenv("SQL_AGENT_FIX_LAYERED_SUMMON", "1")
    try:
        engine_workers.ActorRolloutRefWorker = original
        assert patch_verl_layered_summon()
        assert engine_workers.ActorRolloutRefWorker is LayeredSummonActorRolloutRefWorker
        assert LayeredSummonActorRolloutRefWorker.__module__ == "sql_agent_training.train.verl_layered_summon"
        assert getattr(
            LayeredSummonActorRolloutRefWorker.update_weights,
            "_sql_agent_layered_summon_fixed",
            False,
        )
    finally:
        engine_workers.ActorRolloutRefWorker = original
