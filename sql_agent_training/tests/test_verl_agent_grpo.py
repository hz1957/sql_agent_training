from sql_agent_training.train.verl_agent_grpo import build_verl_main_ppo_command, command_to_shell


def _config() -> dict:
    return {
        "model": {"path": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "sft_checkpoint": None},
        "data": {
            "data_dir": "data/spider",
            "train_file": "train_spider.json",
            "validation_file": "dev.json",
        },
        "rollout": {
            "n": 2,
            "temperature": 0.7,
            "max_prompt_length": 1024,
            "max_response_length": 256,
        },
        "verl": {
            "default_agent_loop": "sql_agent",
            "agent_loop_config_path": "configs/verl_sql_agent_loop.yaml",
            "train_parquet": "artifacts/verl/train.parquet",
            "val_parquet": "artifacts/verl/val.parquet",
            "train_batch_size": 1,
            "val_batch_size": 1,
            "total_training_steps": 1,
            "n_gpus_per_node": 1,
            "rollout_name": "vllm",
            "agent_num_workers": 1,
        },
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "output": {"checkpoint_dir": "artifacts/checkpoints/grpo"},
    }


def test_build_verl_main_ppo_command_contains_agent_grpo_overrides() -> None:
    command = build_verl_main_ppo_command(_config())

    assert command[1:3] == ["-m", "verl.trainer.main_ppo"]
    assert "algorithm.adv_estimator=grpo" in command
    assert "actor_rollout_ref.rollout.agent.default_agent_loop=sql_agent" in command
    assert "actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/verl_sql_agent_loop.yaml" in command
    assert "actor_rollout_ref.rollout.n=2" in command
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in command
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=1" in command
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" in command
    assert "data.train_files=artifacts/verl/train.parquet" in command
    assert "reward.reward_model.enable=false" in command


def test_command_to_shell_quotes_arguments() -> None:
    rendered = command_to_shell(["python", "-m", "verl.trainer.main_ppo", "trainer.logger=[console]"])

    assert "verl.trainer.main_ppo" in rendered
    assert "trainer.logger" in rendered
