import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from datasets import Dataset

from sql_agent_training.train.verl_spider import build_verl_spider_rows, prepare_verl_spider_dataset


def _write_minimal_spider_dir(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "tables.json").write_text(
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
    rows = [{"uid": "music:0", "db_id": "music", "question": "List names.", "query": "SELECT Name FROM Singer"}]
    (root / "train_spider.json").write_text(json.dumps(rows), encoding="utf-8")
    (root / "dev.json").write_text(json.dumps(rows), encoding="utf-8")
    db_dir = root / "database" / "music"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "music.sqlite")
    try:
        conn.execute("CREATE TABLE Singer (Name TEXT)")
        conn.commit()
    finally:
        conn.close()


def test_build_verl_spider_rows_contains_agent_loop_fields(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    _write_minimal_spider_dir(data_dir)

    rows = build_verl_spider_rows(data_dir=data_dir, split_file="train_spider.json", split="train")

    assert len(rows) == 1
    row = rows[0]
    assert row["agent_name"] == "sql_agent"
    assert row["data_source"] == "spider_sql_agent"
    assert row["prompt"][0]["role"] == "user"
    assert "SQL:" in row["prompt"][0]["content"]
    assert row["reward_model"]["ground_truth"] == "SELECT Name FROM Singer"
    assert row["extra_info"]["uid"] == "music:0"
    assert row["extra_info"]["db_id"] == "music"
    assert Path(row["extra_info"]["sqlite_path"]).is_absolute()


def test_prepare_verl_spider_dataset_writes_parquet(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    output_dir = tmp_path / "verl_spider"
    _write_minimal_spider_dir(data_dir)

    summary = prepare_verl_spider_dataset(data_dir=data_dir, output_dir=output_dir)

    assert summary.num_train_rows == 1
    assert summary.num_validation_rows == 1
    train_rows = Dataset.from_parquet(summary.train_path)
    assert train_rows[0]["agent_name"] == "sql_agent"
    assert train_rows[0]["extra_info"]["gold_sql"] == "SELECT Name FROM Singer"


def test_prepare_verl_spider_cli_outputs_summary(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    output_dir = tmp_path / "verl_spider"
    _write_minimal_spider_dir(data_dir)
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_verl_spider.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["num_train_rows"] == 1
    assert summary["num_validation_rows"] == 1
    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "validation.parquet").exists()
