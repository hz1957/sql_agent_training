"""VERL Agent GRPO entrypoint for Spider SQL agent."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from sql_agent_training.agent.sql_agent_loop import SqlAgentInput, SqlAgentLoop
from sql_agent_training.agent.tokenization import load_tokenizer, trajectory_to_tokenized
from sql_agent_training.train.grpo_batch import build_grpo_batch
from scripts.prepare_verl_spider import write_verl_spider_parquet


def _run_dry_validation(rollout_n: int, tokenizer_kind: str, model_path: str) -> None:
    sample = SqlAgentInput(
        uid="dry-sample-0",
        rollout_id="dry-sample-0:template",
        question="How many singers are there?",
        db_id="dry",
        schema_prompt="Database: dry\n- Singer(Name)",
        gold_sql=None,
    )
    tokenizer = load_tokenizer(tokenizer_kind, model_path if tokenizer_kind == "hf" else None)
    tokenized = []
    for index in range(rollout_n):
        rollout_sample = SqlAgentInput(
            uid=sample.uid,
            rollout_id=f"dry-sample-0:{index}",
            question=sample.question,
            db_id=sample.db_id,
            schema_prompt=sample.schema_prompt,
            gold_sql=sample.gold_sql,
        )
        trajectory = SqlAgentLoop(max_turns=1).empty_trajectory(rollout_sample, reason="dry_run")
        tokenized.append(trajectory_to_tokenized(trajectory, tokenizer))

    batch = build_grpo_batch(tokenized, rollout_n=rollout_n)
    print(f"dry_run/groups: {len(batch.groups)}")
    print(f"dry_run/trajectories: {batch.num_trajectories}")


def _as_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _model_path(config: dict) -> str:
    return str(config["model"].get("sft_checkpoint") or config["model"]["path"])


def _prepare_verl_parquets(config: dict) -> dict[str, int]:
    data_dir = Path(config["data"]["data_dir"])
    verl_config = config["verl"]
    train_count = write_verl_spider_parquet(
        data_dir=data_dir,
        split_file=config["data"]["train_file"],
        output_path=Path(verl_config["train_parquet"]),
        limit=int(verl_config["train_limit"]) if verl_config.get("train_limit") is not None else None,
        agent_name=str(verl_config["default_agent_loop"]),
    )
    val_count = write_verl_spider_parquet(
        data_dir=data_dir,
        split_file=config["data"]["validation_file"],
        output_path=Path(verl_config["val_parquet"]),
        limit=int(verl_config["val_limit"]) if verl_config.get("val_limit") is not None else None,
        agent_name=str(verl_config["default_agent_loop"]),
    )
    return {"train_rows": train_count, "val_rows": val_count}


def build_verl_main_ppo_command(config: dict) -> list[str]:
    """Build the VERL one-batch Agent GRPO command as argv."""

    rollout = config["rollout"]
    verl_config = config["verl"]
    algorithm = config["algorithm"]
    output = config["output"]
    model_path = _model_path(config)
    overrides = [
        f"algorithm.adv_estimator={algorithm.get('adv_estimator', 'grpo')}",
        f"algorithm.use_kl_in_reward={_as_bool(algorithm.get('use_kl_in_reward', False))}",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.trust_remote_code=true",
        f"actor_rollout_ref.rollout.name={verl_config.get('rollout_name', 'vllm')}",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.n={int(rollout['n'])}",
        f"actor_rollout_ref.rollout.temperature={float(rollout.get('temperature', 0.7))}",
        f"actor_rollout_ref.rollout.prompt_length={int(rollout['max_prompt_length'])}",
        f"actor_rollout_ref.rollout.response_length={int(rollout['max_response_length'])}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={int(verl_config.get('tensor_model_parallel_size', 1))}",
        f"actor_rollout_ref.rollout.agent.default_agent_loop={verl_config['default_agent_loop']}",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={verl_config['agent_loop_config_path']}",
        f"actor_rollout_ref.rollout.agent.num_workers={int(verl_config.get('agent_num_workers', 1))}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={int(verl_config.get('ppo_mini_batch_size', 1))}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={int(verl_config.get('ppo_micro_batch_size_per_gpu', 1))}",
        f"data.train_files={verl_config['train_parquet']}",
        f"data.val_files={verl_config['val_parquet']}",
        f"data.train_batch_size={int(verl_config.get('train_batch_size', 1))}",
        f"data.val_batch_size={int(verl_config.get('val_batch_size', 1))}",
        f"data.max_prompt_length={int(rollout['max_prompt_length'])}",
        f"data.max_response_length={int(rollout['max_response_length'])}",
        "data.return_raw_chat=true",
        "data.filter_overlong_prompts=false",
        "data.shuffle=false",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={int(verl_config.get('total_training_steps', 1))}",
        f"trainer.n_gpus_per_node={int(verl_config.get('n_gpus_per_node', 1))}",
        "trainer.val_before_train=false",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        "trainer.logger=[console]",
        f"trainer.default_local_dir={output['checkpoint_dir']}",
        "reward.reward_model.enable=false",
    ]
    return [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]


def command_to_shell(command: list[str]) -> str:
    """Render argv as a shell-readable command."""

    return " ".join(shlex.quote(part) for part in command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VERL Agent GRPO for Spider SQL agent.")
    parser.add_argument("--config", default="configs/agent_grpo.local_dryrun.yaml")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare VERL parquet and stop.")
    parser.add_argument("--print-command", action="store_true", help="Print the VERL main_ppo command and stop.")
    parser.add_argument("--run-verl", action="store_true", help="Actually run VERL main_ppo. Intended for AutoDL.")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config.get("dry_run", False):
        print("VERL Agent GRPO dry run config loaded.")
        print(f"model: {config['model']['path']}")
        print(f"rollout.n: {config['rollout']['n']}")
        tokenizer_kind = config.get("tokenizer", {}).get("kind", "whitespace")
        _run_dry_validation(int(config["rollout"]["n"]), tokenizer_kind, config["model"]["path"])
        return

    counts = _prepare_verl_parquets(config)
    print(f"prepared train_rows={counts['train_rows']} val_rows={counts['val_rows']}")
    command = build_verl_main_ppo_command(config)
    print(command_to_shell(command))

    if args.prepare_only or args.print_command or not args.run_verl:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
