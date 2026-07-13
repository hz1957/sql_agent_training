import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

from sql_agent_training.data.schema import load_tables_json
from sql_agent_training.data.spider_dataset import SpiderExample
from sql_agent_training.train.sft_eval import (
    _resolve_adapter_base_model_path,
    _resolve_model_and_tokenizer,
    evaluate_predictions,
    generate_predictions,
    normalize_generated_sql,
    write_predictions_jsonl,
)


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


def test_normalize_generated_sql_keeps_first_statement() -> None:
    assert normalize_generated_sql("SQL: SELECT Name FROM Singer; SELECT COUNT(*) FROM Singer;") == (
        "SELECT Name FROM Singer"
    )
    assert normalize_generated_sql("```sql\nSELECT Name FROM Singer;\n```") == "SELECT Name FROM Singer"


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


def test_sft_eval_cli_uses_configured_random_sample_size(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    _write_eval_spider_dir(data_dir, row_count=3)
    output = tmp_path / "predictions.jsonl"
    config_path = tmp_path / "sft_eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"path": "dummy"},
                "data": {"data_dir": str(data_dir), "train_file": "train_spider.json", "validation_file": "dev.json"},
                "output": {"checkpoint_dir": "dummy", "predictions_jsonl": str(output)},
                "eval": {"sample_size": 2, "sample_seed": 0},
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
    assert summary["total"] == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_resolve_model_and_tokenizer_uses_latest_nested_model_dir(tmp_path: Path) -> None:
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    (base_model / "tokenizer.json").write_text("{}", encoding="utf-8")

    checkpoint_root = tmp_path / "sft_checkpoint"
    checkpoint_875 = checkpoint_root / "checkpoint-875"
    timestamped_run = checkpoint_root / "20260711_061234"
    checkpoint_875.mkdir(parents=True)
    timestamped_run.mkdir()
    (checkpoint_875 / "model.safetensors").write_text("", encoding="utf-8")
    (timestamped_run / "model.safetensors").write_text("", encoding="utf-8")

    model_path, tokenizer_path = _resolve_model_and_tokenizer(
        {
            "model": {"path": str(base_model)},
            "output": {"checkpoint_dir": str(checkpoint_root)},
        },
        checkpoint=None,
        tokenizer_path=None,
    )

    assert model_path == str(timestamped_run)
    assert tokenizer_path == str(base_model)


def test_resolve_model_and_tokenizer_uses_latest_nested_lora_adapter_dir(tmp_path: Path) -> None:
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    (base_model / "tokenizer.json").write_text("{}", encoding="utf-8")

    checkpoint_root = tmp_path / "sft_lora_checkpoint"
    older_run = checkpoint_root / "20260711_061234"
    latest_run = checkpoint_root / "20260711_071234"
    older_run.mkdir(parents=True)
    latest_run.mkdir()
    (older_run / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base_model)}),
        encoding="utf-8",
    )
    (latest_run / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base_model)}),
        encoding="utf-8",
    )

    model_path, tokenizer_path = _resolve_model_and_tokenizer(
        {
            "model": {"path": str(base_model)},
            "output": {"checkpoint_dir": str(checkpoint_root)},
        },
        checkpoint=None,
        tokenizer_path=None,
    )

    assert model_path == str(latest_run)
    assert tokenizer_path == str(base_model)


def test_resolve_adapter_base_model_path_prefers_fallback_when_recorded_path_is_missing(tmp_path: Path) -> None:
    adapter_config = tmp_path / "adapter_config.json"
    fallback_model = tmp_path / "base_model"
    fallback_model.mkdir()
    adapter_config.write_text(
        json.dumps({"base_model_name_or_path": str(tmp_path / "missing_model")}),
        encoding="utf-8",
    )

    assert _resolve_adapter_base_model_path(
        adapter_config,
        fallback_base_model_name_or_path=str(fallback_model),
    ) == str(fallback_model)
