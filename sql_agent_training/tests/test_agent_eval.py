import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

from sql_agent_training.data.schema import load_tables_json
from sql_agent_training.data.spider_dataset import SpiderExample
from sql_agent_training.train.agent_eval import evaluate_agent, summarize_agent_eval, write_eval_outputs
from sql_agent_training.train.eval_sampling import select_eval_examples


def _write_eval_spider_dir(root: Path, *, row_count: int = 1) -> None:
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
    rows = [
        {"uid": f"music:{index}", "db_id": "music", "question": "List names.", "query": "SELECT Name FROM Singer"}
        for index in range(row_count)
    ]
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


def test_select_eval_examples_random_sample_is_seeded() -> None:
    examples = list(range(10))

    assert select_eval_examples(examples, sample_size=3, sample_seed=7) == [5, 2, 6]
    assert select_eval_examples(examples, limit=2, sample_size=3, sample_seed=7) == [0, 1]


def test_evaluate_agent_dry_run_gold_writes_metrics(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    _write_eval_spider_dir(data_dir)
    tables = load_tables_json(data_dir / "tables.json")
    examples = [SpiderExample(uid="music:0", db_id="music", question="List names.", gold_sql="SELECT Name FROM Singer")]

    rows = evaluate_agent(examples, tables, data_dir, dry_run_gold=True, max_turns=1)
    metrics = summarize_agent_eval(rows)
    predictions = tmp_path / "eval_predictions.jsonl"
    metrics_json = tmp_path / "eval_metrics.json"
    write_eval_outputs(rows, metrics, predictions_jsonl=predictions, metrics_json=metrics_json)

    assert metrics["execution_accuracy"] == 1.0
    assert metrics["executable_rate"] == 1.0
    assert metrics["avg_turns"] == 1.0
    assert predictions.exists()
    assert json.loads(metrics_json.read_text(encoding="utf-8"))["total"] == 1


def test_agent_eval_cli_dry_run_gold(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    _write_eval_spider_dir(data_dir)
    output_dir = tmp_path / "eval"
    config_path = tmp_path / "agent_eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"path": "dummy"},
                "data": {"data_dir": str(data_dir), "train_file": "train_spider.json", "validation_file": "dev.json"},
                "rollout": {"max_turns": 1, "max_response_length": 64, "temperature": 0.0},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sql_agent_training.train.agent_eval",
            "--config",
            str(config_path),
            "--dry-run-gold",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["execution_accuracy"] == 1.0
    assert (output_dir / "eval_predictions.jsonl").exists()
    assert (output_dir / "eval_metrics.json").exists()


def test_agent_eval_cli_uses_configured_random_sample_size(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    _write_eval_spider_dir(data_dir, row_count=3)
    output_dir = tmp_path / "eval"
    config_path = tmp_path / "agent_eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"path": "dummy"},
                "data": {"data_dir": str(data_dir), "train_file": "train_spider.json", "validation_file": "dev.json"},
                "rollout": {"max_turns": 1, "max_response_length": 64, "temperature": 0.0},
                "eval": {"sample_size": 2, "sample_seed": 0},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sql_agent_training.train.agent_eval",
            "--config",
            str(config_path),
            "--dry-run-gold",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(completed.stdout)
    predictions = output_dir / "eval_predictions.jsonl"
    assert summary["total"] == 2
    assert len(predictions.read_text(encoding="utf-8").splitlines()) == 2
