"""Pipeline entrypoint for SFT and Agent GRPO stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


VALID_STAGES = {"sft", "grpo"}


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def _parse_stages(value: str | None, config: dict[str, Any]) -> list[str]:
    if value:
        stages = [stage.strip() for stage in value.split(",") if stage.strip()]
    else:
        stages = list(config.get("stages") or [])
    if not stages:
        raise ValueError("At least one stage is required.")
    unknown = [stage for stage in stages if stage not in VALID_STAGES]
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Valid stages: {sorted(VALID_STAGES)}")
    return stages


def build_stage_commands(config: dict[str, Any], stages: list[str]) -> list[list[str]]:
    """Build subprocess commands for selected pipeline stages."""

    commands: list[list[str]] = []
    for stage in stages:
        if stage == "sft":
            sft_config = str(config.get("sft_config", "configs/sft.yaml"))
            sft_command = [sys.executable, "-m", "sql_agent_training.train.sft", "--config", sft_config]
            if bool(config.get("sft_dry_run", False)):
                sft_command.append("--dry-run")
            commands.append(sft_command)
        elif stage == "grpo":
            grpo_config = str(config.get("grpo_config", "configs/agent_grpo.yaml"))
            grpo_command = [sys.executable, "-m", "sql_agent_training.train.verl_agent_grpo", "--config", grpo_config]
            if bool(config.get("grpo_run_verl", False)):
                grpo_command.append("--run-verl")
            elif bool(config.get("grpo_prepare_only", False)):
                grpo_command.append("--prepare-only")
            commands.append(grpo_command)

    return commands


def run_commands(commands: list[list[str]]) -> None:
    """Run pipeline commands in order."""

    for command in commands:
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SQL agent training pipeline.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--stages", default=None, help="Comma-separated stages: sft, grpo, or sft,grpo.")
    parser.add_argument("--print-commands", action="store_true", help="Print stage commands without running them.")
    args = parser.parse_args()

    config = _load_config(args.config)
    stages = _parse_stages(args.stages, config)
    commands = build_stage_commands(config, stages)

    print(f"Pipeline config: {args.config}")
    print(f"Requested stages: {stages}")
    for command in commands:
        print("+ " + " ".join(command))

    if args.print_commands:
        return
    run_commands(commands)


if __name__ == "__main__":
    main()
