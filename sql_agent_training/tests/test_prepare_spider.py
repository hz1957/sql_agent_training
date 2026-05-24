import json
import sqlite3
import subprocess
import sys
from pathlib import Path


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
    rows = [{"db_id": "music", "question": "List names.", "query": "SELECT Name FROM Singer"}]
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


def test_prepare_spider_verify_only_generates_schema_cache(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    _write_minimal_spider_dir(data_dir)
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_spider.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-dir",
            str(data_dir),
            "--verify-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["ok"] is True
    assert summary["num_train_examples"] == 1
    assert summary["num_validation_examples"] == 1
    assert (data_dir / "schema_cache.json").exists()
