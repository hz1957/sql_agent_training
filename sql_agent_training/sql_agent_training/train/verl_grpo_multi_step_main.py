"""verl PPO entrypoint that registers SQL-agent multi-step GRPO extensions."""

from __future__ import annotations

from verl.trainer import main_ppo as verl_main_ppo


class MultiStepTaskRunner(verl_main_ppo.TaskRunner):
    """Import project extensions inside the Ray TaskRunner process."""

    def run(self, config):  # type: ignore[no-untyped-def]
        import sql_agent_training.train.verl_grpo_multi_step  # noqa: F401

        return super().run(config)


def main() -> None:
    """Run verl's Hydra entrypoint after swapping in the custom TaskRunner."""

    import sql_agent_training.train.verl_grpo_multi_step  # noqa: F401

    verl_main_ppo.TaskRunner = MultiStepTaskRunner
    verl_main_ppo.main()


if __name__ == "__main__":
    main()
