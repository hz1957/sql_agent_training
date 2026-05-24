import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

from sql_agent_training.data.schema import load_tables_json
from sql_agent_training.data.spider_dataset import SpiderExample
from sql_agent_training.train.sft_eval import (
    evaluate_predictions,
    generate_predictions,
    write_predictions_jsonl,
)


def _write_eval_spider_dir(root: Path) -> None:
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
    rows = [{"db_id": "music", "question": "List names.", "query": "SELECT Name FROM Singer"}]
    (root / "dev.json").write_text(json.dumps(rows), encoding="utf-8")
    (root / "train_spider.json").write_text(json.dumps(rows), encoding="utf-8")
    db_dir = root / "database" / "music"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "music.sqlite")
    try:
        conn.execute("CREATE TABLE Singer (Name TEXT)")
        conn.execute("INSERT INTO Singer VALUES ('Ada')")
        conn.commit()
    finally:
        conn.close()


def test_generate_and_evaluate_predictions(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    _write_eval_spider_dir(data_dir)
    tables = load_tables_json(data_dir / "tables.json")
    examples = [SpiderExample(uid="music:0", db_id="music", question="List names.", gold_sql="SELECT Name FROM Singer")]

    rows = generate_predictions(examples, tables, dry_run_gold=True)
    output = tmp_path / "predictions.jsonl"
    count = write_predictions_jsonl(rows, output)
    metrics = evaluate_predictions(rows, data_dir)

    assert count == 1
    assert output.exists()
    assert metrics == {"total": 1, "executable_rate": 1.0, "execution_accuracy": 1.0}


def test_sft_eval_cli_dry_run_gold(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    _write_eval_spider_dir(data_dir)
    output = tmp_path / "predictions.jsonl"
    config_path = tmp_path / "sft_eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"path": "dummy"},
                "data": {"data_dir": str(data_dir), "train_file": "train_spider.json", "validation_file": "dev.json"},
                "output": {"checkpoint_dir": "dummy", "predictions_jsonl": str(output)},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sql_agent_training.train.sft_eval",
            "--config",
            str(config_path),
            "--dry-run-gold",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["total"] == 1
    assert summary["executable_rate"] == 1.0
    assert summary["execution_accuracy"] == 1.0
    assert output.exists()
