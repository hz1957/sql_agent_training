import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sql_agent_training.train.sft import _assert_checkpoint_complete


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


def test_checkpoint_validation_rejects_config_only_directory(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint-875"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "generation_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no model weights found"):
        _assert_checkpoint_complete(checkpoint_dir, requires_tokenizer=False)


def test_checkpoint_validation_accepts_weights_and_tokenizer(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    _assert_checkpoint_complete(checkpoint_dir, requires_tokenizer=True)
