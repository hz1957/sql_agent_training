import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from sql_agent_training.train.sft import (
    _build_train_metric_dataset,
    _compute_token_accuracy,
    _deepspeed_config,
    _eval_strategy,
    _lora_config_kwargs,
    _new_checkpoint_dir,
    _normalize_save_strategy,
    _optional_positive_int,
    _trainer_output_dir,
)
from sql_agent_training.train.sft_dataset import IGNORE_INDEX, SftTorchDataset, TokenizedSftRecord


def test_sft_dry_run_writes_jsonl(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "spider"
    data_dir.mkdir(parents=True)
    (data_dir / "tables.json").write_text(
        json.dumps(
            [
                {
                    "db_id": "music",
                    "table_names_original": ["Singer"],
                    "column_names_original": [[-1, "*"], [0, "Name"]],
                }
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "train_spider.json").write_text(
        json.dumps([{"db_id": "music", "question": "List names.", "query": "SELECT Name FROM Singer"}]),
        encoding="utf-8",
    )
    output_path = tmp_path / "artifacts" / "sft.jsonl"
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"data_dir": str(data_dir), "train_file": "train_spider.json"},
                "output": {"sft_jsonl": str(output_path)},
                "model": {"path": "dummy"},
                "tokenizer": {"kind": "whitespace"},
                "training": {"max_prompt_length": 128, "max_response_length": 32},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sql_agent_training.train.sft",
            "--config",
            str(config_path),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote 1 SFT records" in completed.stdout
    row = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert row["completion"] == "SELECT Name FROM Singer"
    assert "SELECT Name FROM Singer" not in row["prompt"]


def test_sft_new_checkpoint_dir_uses_timestamped_run_folder(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "sft_model"

    checkpoint_dir = _new_checkpoint_dir(checkpoint_root)

    assert checkpoint_dir.parent == checkpoint_root
    assert checkpoint_dir.name.startswith("20")
    assert len(checkpoint_dir.name) == len("20260711_061234")


def test_sft_trainer_output_dir_defaults_to_final_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "sft_model" / "20260711_061234"
    config = {"output": {"checkpoint_dir": str(tmp_path / "sft_model")}}

    assert _trainer_output_dir(config, checkpoint_dir) == checkpoint_dir


def test_sft_trainer_output_dir_can_be_overridden(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "sft_model"
    trainer_dir = tmp_path / "trainer_state"
    config = {"output": {"checkpoint_dir": str(checkpoint_dir), "trainer_output_dir": str(trainer_dir)}}

    assert _trainer_output_dir(config, checkpoint_dir) == trainer_dir


def test_sft_save_strategy_normalizes_yaml_booleans() -> None:
    assert _normalize_save_strategy(False) == "no"
    assert _normalize_save_strategy(None) == "no"
    assert _normalize_save_strategy("no") == "no"
    assert _normalize_save_strategy("steps") == "steps"
    assert _normalize_save_strategy("false") == "no"


def test_sft_save_strategy_rejects_invalid_values() -> None:
    try:
        _normalize_save_strategy(True)
    except ValueError as exc:
        assert "training.save_strategy" in str(exc)
    else:
        raise AssertionError("Expected invalid save_strategy to raise ValueError")


def test_deepspeed_config_accepts_disabled_path_and_inline_config() -> None:
    assert _deepspeed_config({}) is None
    assert _deepspeed_config({"deepspeed": False}) is None
    assert _deepspeed_config({"deepspeed": ""}) is None
    assert _deepspeed_config({"deepspeed": "configs/ds_zero3.json"}) == "configs/ds_zero3.json"
    assert _deepspeed_config({"deepspeed": {"zero_optimization": {"stage": 3}}}) == {
        "zero_optimization": {"stage": 3}
    }


def test_deepspeed_config_rejects_invalid_values() -> None:
    try:
        _deepspeed_config({"deepspeed": 3})
    except ValueError as exc:
        assert "training.deepspeed" in str(exc)
    else:
        raise AssertionError("Expected invalid deepspeed config to raise ValueError")


def test_eval_strategy_normalizes_values() -> None:
    assert _eval_strategy({}) == "no"
    assert _eval_strategy({"eval_strategy": "steps"}) == "steps"
    assert _eval_strategy({"eval_strategy": "epoch"}) == "epoch"
    assert _eval_strategy({"eval_strategy": "false"}) == "no"


def test_eval_strategy_rejects_invalid_values() -> None:
    try:
        _eval_strategy({"eval_strategy": "sometimes"})
    except ValueError as exc:
        assert "training.eval_strategy" in str(exc)
    else:
        raise AssertionError("Expected invalid eval_strategy to raise ValueError")


def test_optional_positive_int_accepts_missing_and_positive_values() -> None:
    assert _optional_positive_int({}, "eval_steps") is None
    assert _optional_positive_int({"eval_steps": "100"}, "eval_steps") == 100


def test_optional_positive_int_rejects_non_positive_values() -> None:
    try:
        _optional_positive_int({"eval_steps": 0}, "eval_steps")
    except ValueError as exc:
        assert "training.eval_steps" in str(exc)
    else:
        raise AssertionError("Expected non-positive optional integer to raise ValueError")


def test_compute_token_accuracy_uses_next_token_shift_and_ignores_prompt_labels() -> None:
    predictions = np.array([[9, 5, 9, 6, 7]])
    labels = np.array([[IGNORE_INDEX, IGNORE_INDEX, 5, 8, IGNORE_INDEX]])

    metrics = _compute_token_accuracy((predictions, labels))

    assert metrics["token_accuracy"] == 0.5


def test_build_train_metric_dataset_uses_configured_sample_size() -> None:
    dataset = SftTorchDataset(
        [
            TokenizedSftRecord(
                uid=str(index),
                db_id="music",
                input_ids=[1, 2, index],
                attention_mask=[1, 1, 1],
                labels=[IGNORE_INDEX, 2, index],
            )
            for index in range(5)
        ]
    )

    metric_dataset = _build_train_metric_dataset({"eval": {"train_sample_size": 2, "sample_seed": 0}}, dataset)

    assert len(metric_dataset) == 2


def test_lora_config_kwargs_uses_qwen_defaults() -> None:
    kwargs = _lora_config_kwargs({"lora": {"enabled": True}})

    assert kwargs == {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }


def test_lora_config_kwargs_accepts_overrides() -> None:
    kwargs = _lora_config_kwargs(
        {
            "lora": {
                "r": 32,
                "alpha": 64,
                "dropout": 0.1,
                "bias": "lora_only",
                "target_modules": ["q_proj", "v_proj"],
            }
        }
    )

    assert kwargs == {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "bias": "lora_only",
        "target_modules": ["q_proj", "v_proj"],
    }
