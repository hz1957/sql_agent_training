"""verl PPO entrypoint that registers SQL-agent multi-step GRPO extensions."""

from __future__ import annotations

import importlib

from verl.trainer import main_ppo as verl_main_ppo


def _import_multi_step_extensions() -> None:
    import sql_agent_training.train.verl_grpo_multi_step  # noqa: F401
    from sql_agent_training.train.verl_grpo_tree import patch_verl_compute_advantage

    patch_verl_compute_advantage()


def _patch_task_runner_run(task_runner_cls) -> bool:  # type: ignore[no-untyped-def]
    """Patch a regular class or Ray ActorClass without subclassing it."""

    metadata = getattr(task_runner_cls, "__ray_metadata__", None)
    target_cls = getattr(metadata, "modified_class", task_runner_cls)
    original_run = getattr(target_cls, "run", None)
    if original_run is None or getattr(original_run, "_sql_agent_multi_step_patched", False):
        return False

    def run_with_multi_step_extensions(self, config, *args, **kwargs):  # type: ignore[no-untyped-def]
        _import_multi_step_extensions()
        return original_run(self, config, *args, **kwargs)

    run_with_multi_step_extensions._sql_agent_multi_step_patched = True  # type: ignore[attr-defined]
    setattr(target_cls, "run", run_with_multi_step_extensions)
    return True


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
            if base_cls is None:
                continue
            if _patch_task_runner_run(base_cls):
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
