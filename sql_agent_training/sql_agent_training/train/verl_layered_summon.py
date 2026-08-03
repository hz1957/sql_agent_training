"""Compatibility patch for verl's FSDP layered LoRA summon."""

from __future__ import annotations

import functools
import inspect
import os
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any

from verl.workers.engine_workers import ActorRolloutRefWorker as _VerlActorRolloutRefWorker

PATCH_ENV = "SQL_AGENT_FIX_LAYERED_SUMMON"
LOG_PREFIX = "LAYERED_SUMMON_PATCH"
_COLLECTION_LOGGED = False


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _clean_fsdp_name(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "").strip(".")


def _qualify_fsdp_param_name(unit_name: str, param_name: str) -> str:
    """Rebuild a PEFT parameter name from an FSDP unit and its local name."""

    clean_unit_name = _clean_fsdp_name(unit_name)
    clean_param_name = _clean_fsdp_name(param_name)
    if not clean_unit_name:
        return clean_param_name
    if clean_param_name == clean_unit_name or clean_param_name.startswith(f"{clean_unit_name}."):
        return clean_param_name
    return f"{clean_unit_name}.{clean_param_name}"


def layered_summon_lora_params_fixed(fsdp_module: Any) -> OrderedDict[str, Any]:
    """Collect LoRA tensors one FSDP unit at a time with qualified PEFT keys."""

    global _COLLECTION_LOGGED

    from peft.utils.save_and_load import get_peft_model_state_dict
    from verl.utils import fsdp_utils

    lora_params: OrderedDict[str, Any] = OrderedDict()
    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)

    for unit_name, submodule in fsdp_module.named_modules():
        if unit_name == "" or fsdp_utils.fsdp_version(submodule) == 0:
            continue

        clean_unit_name = _clean_fsdp_name(unit_name)
        if clean_unit_name.endswith(".model") or clean_unit_name.endswith(".layers"):
            continue

        nested_fsdp_names = {
            name for name, module in submodule.named_modules() if name != "" and fsdp_utils.fsdp_version(module) > 0
        }

        def local_params() -> list[tuple[str, Any]]:
            return [
                (name, param)
                for name, param in submodule.named_parameters()
                if not any(name.startswith(f"{nested_name}.") for nested_name in nested_fsdp_names)
            ]

        if not any("lora_" in _qualify_fsdp_param_name(unit_name, name) for name, _ in local_params()):
            continue

        is_fsdp1 = fsdp_utils.fsdp_version(submodule) == 1
        previous_is_root = getattr(submodule, "_is_root", None)
        if is_fsdp1:
            submodule._is_root = True
        summon_context = fsdp_utils.FSDP.summon_full_params(submodule, writeback=False) if is_fsdp1 else nullcontext()

        try:
            with summon_context:
                qualified_state = {_qualify_fsdp_param_name(unit_name, name): param for name, param in local_params()}
                unit_lora_params = get_peft_model_state_dict(peft_model, state_dict=qualified_state)
                for name, param in unit_lora_params.items():
                    lora_params[name] = (
                        param.full_tensor().detach().cpu() if hasattr(param, "full_tensor") else param.detach().cpu()
                    )
        finally:
            if is_fsdp1:
                submodule._is_root = previous_is_root
        fsdp_utils.get_torch_device().empty_cache()

    if not lora_params:
        raise RuntimeError(
            "Layered LoRA summon collected no tensors; refusing verl's full-summon fallback. "
            f"Disable {PATCH_ENV} explicitly to use the unpatched behavior."
        )

    if not _COLLECTION_LOGGED:
        print(f"{LOG_PREFIX}_RESULT tensors={len(lora_params)}", flush=True)
        _COLLECTION_LOGGED = True
    return lora_params


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _wrap_update_weights(worker_cls: type[Any]) -> bool:
    original = getattr(worker_cls, "update_weights", None)
    if original is None or getattr(original, "_sql_agent_layered_summon_fixed", False):
        return False

    @functools.wraps(original)
    async def update_weights_with_layered_fix(self: Any, *args: Any, **kwargs: Any) -> Any:
        import verl.utils.fsdp_utils as fsdp_utils

        original_collector = fsdp_utils.layered_summon_lora_params
        fsdp_utils.layered_summon_lora_params = layered_summon_lora_params_fixed
        try:
            # Release cached actor blocks before vLLM remaps its sleeping weight buffers.
            fsdp_utils.get_torch_device().empty_cache()
            return await _await_if_needed(original(self, *args, **kwargs))
        finally:
            fsdp_utils.layered_summon_lora_params = original_collector

    update_weights_with_layered_fix._sql_agent_layered_summon_fixed = True  # type: ignore[attr-defined]
    setattr(worker_cls, "update_weights", update_weights_with_layered_fix)
    return True


class LayeredSummonActorRolloutRefWorker(_VerlActorRolloutRefWorker):
    """Importable worker subclass whose compatibility patch survives Ray spawning."""


_wrap_update_weights(LayeredSummonActorRolloutRefWorker)


def patch_verl_layered_summon() -> bool:
    """Select the patched worker class when the compatibility flag is enabled."""

    if not _env_flag(PATCH_ENV):
        return False

    import verl.workers.engine_workers as engine_workers

    if engine_workers.ActorRolloutRefWorker is LayeredSummonActorRolloutRefWorker:
        return False
    engine_workers.ActorRolloutRefWorker = LayeredSummonActorRolloutRefWorker
    print(
        f"{LOG_PREFIX}_ENABLED "
        f"worker={LayeredSummonActorRolloutRefWorker.__module__}.{LayeredSummonActorRolloutRefWorker.__name__}",
        flush=True,
    )
    return True
