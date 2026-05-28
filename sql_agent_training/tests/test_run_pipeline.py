import sys

import pytest

from sql_agent_training.train.run_pipeline import _parse_stages, build_stage_commands


def test_parse_stages_uses_config_default() -> None:
    assert _parse_stages(None, {"stages": ["sft", "grpo"]}) == ["sft", "grpo"]


def test_parse_stages_accepts_single_stage_override() -> None:
    assert _parse_stages("grpo", {"stages": ["sft", "grpo"]}) == ["grpo"]


def test_parse_stages_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown stages"):
        _parse_stages("sft,bad", {})


def test_build_stage_commands_supports_sft_only() -> None:
    commands = build_stage_commands(
        {
            "sft_config": "configs/sft.local_dryrun.yaml",
            "sft_dry_run": True,
        },
        ["sft"],
    )

    assert commands == [
        [
            sys.executable,
            "-m",
            "sql_agent_training.train.sft",
            "--config",
            "configs/sft.local_dryrun.yaml",
            "--dry-run",
        ]
    ]


def test_build_stage_commands_supports_grpo_only() -> None:
    commands = build_stage_commands(
        {
            "grpo_config": "configs/grpo.yaml",
        },
        ["grpo"],
    )

    assert commands == [
        [
            sys.executable,
            "-m",
            "sql_agent_training.train.grpo_trainer",
            "--config",
            "configs/grpo.yaml",
        ]
    ]


def test_build_stage_commands_supports_sft_then_grpo_dry_run() -> None:
    commands = build_stage_commands(
        {
            "sft_config": "configs/sft.local_dryrun.yaml",
            "sft_dry_run": True,
            "grpo_config": "configs/grpo.local_dryrun.yaml",
            "grpo_dry_run": True,
        },
        ["sft", "grpo"],
    )

    assert commands == [
        [
            sys.executable,
            "-m",
            "sql_agent_training.train.sft",
            "--config",
            "configs/sft.local_dryrun.yaml",
            "--dry-run",
        ],
        [
            sys.executable,
            "-m",
            "sql_agent_training.train.grpo_trainer",
            "--config",
            "configs/grpo.local_dryrun.yaml",
            "--dry-run",
        ],
    ]
