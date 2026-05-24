import json
import subprocess
import sys
from pathlib import Path

import yaml


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
