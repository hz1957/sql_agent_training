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


def test_build_stage_commands_supports_grpo_only_prepare() -> None:
    commands = build_stage_commands(
        {
            "grpo_config": "configs/agent_grpo.yaml",
            "grpo_prepare_only": True,
            "grpo_run_verl": False,
        },
        ["grpo"],
    )

    assert commands == [
        [
            sys.executable,
            "-m",
            "sql_agent_training.train.verl_agent_grpo",
            "--config",
            "configs/agent_grpo.yaml",
            "--prepare-only",
        ]
    ]


def test_build_stage_commands_supports_sft_then_grpo_autodl() -> None:
    commands = build_stage_commands(
        {
            "sft_config": "configs/sft.autodl.yaml",
            "grpo_config": "configs/agent_grpo.autodl.yaml",
            "grpo_run_verl": True,
        },
        ["sft", "grpo"],
    )

    assert commands == [
        [
            sys.executable,
            "-m",
            "sql_agent_training.train.sft",
            "--config",
            "configs/sft.autodl.yaml",
        ],
        [
            sys.executable,
            "-m",
            "sql_agent_training.train.verl_agent_grpo",
            "--config",
            "configs/agent_grpo.autodl.yaml",
            "--run-verl",
        ],
    ]
