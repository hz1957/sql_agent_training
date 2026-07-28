"""verl PPO entrypoint that registers SQL-agent multi-step GRPO extensions."""

from __future__ import annotations

import importlib
from typing import Any

from verl.trainer import main_ppo as verl_main_ppo


def _import_multi_step_extensions() -> None:
    import sql_agent_training.train.verl_grpo_multi_step  # noqa: F401


def _wrap_task_runner(base_cls: type[Any], *, module_name: str, attr_name: str) -> type[Any]:
    class MultiStepTaskRunner(base_cls):  # type: ignore[misc, valid-type]
        """Import project extensions inside the Ray TaskRunner process."""

        _sql_agent_multi_step_patched = True

        def run(self, config):  # type: ignore[no-untyped-def]
            _import_multi_step_extensions()
            return super().run(config)

    MultiStepTaskRunner.__name__ = attr_name
    MultiStepTaskRunner.__qualname__ = attr_name
    MultiStepTaskRunner.__module__ = module_name
    return MultiStepTaskRunner


def _patch_verl_task_runners() -> list[str]:
    """Patch whichever verl TaskRunner class exists in the installed version."""

    _import_multi_step_extensions()
    patched: list[str] = []
    for module_name in ("verl.trainer.main_ppo", "verl.trainer.main_ppo_v0"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for attr_name in ("TaskRunner", "TaskRunnerV1"):
            base_cls = getattr(module, attr_name, None)
            if base_cls is None or getattr(base_cls, "_sql_agent_multi_step_patched", False):
                continue
            setattr(module, attr_name, _wrap_task_runner(base_cls, module_name=module_name, attr_name=attr_name))
            patched.append(f"{module_name}.{attr_name}")
    if not patched:
        raise RuntimeError("Could not find a verl TaskRunner/TaskRunnerV1 class to patch.")
    return patched


def main() -> None:
    """Run verl's Hydra entrypoint after swapping in the custom TaskRunner."""

    _patch_verl_task_runners()
    verl_main_ppo.main()


if __name__ == "__main__":
    main()
