import json
import sqlite3
from pathlib import Path

import pandas as pd

from scripts.prepare_verl_spider import write_verl_spider_parquet


def _write_spider_dir(root: Path) -> None:
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
    (root / "train_spider.json").write_text(
        json.dumps([{"db_id": "music", "question": "List names.", "query": "SELECT Name FROM Singer"}]),
        encoding="utf-8",
    )
    db_dir = root / "database" / "music"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "music.sqlite")
    try:
        conn.execute("CREATE TABLE Singer (Name TEXT)")
        conn.commit()
    finally:
        conn.close()


def test_write_verl_spider_parquet(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    _write_spider_dir(data_dir)
    output = tmp_path / "spider_train.parquet"

    count = write_verl_spider_parquet(data_dir=data_dir, split_file="train_spider.json", output_path=output)

    rows = pd.read_parquet(output).to_dict(orient="records")
    assert count == 1
    assert rows[0]["data_source"] == "spider"
    assert rows[0]["agent_name"] == "sql_agent"
    assert rows[0]["prompt"][0]["role"] == "user"
    assert "rewrite the SQL" in rows[0]["prompt"][0]["content"]
    assert "Return one read-only SQLite SELECT query" in rows[0]["prompt"][0]["content"]
    assert rows[0]["extra_info"]["uid"] == "music:0"
    assert rows[0]["extra_info"]["db_id"] == "music"
    assert rows[0]["extra_info"]["gold_sql"] == "SELECT Name FROM Singer"
    assert rows[0]["extra_info"]["sqlite_path"].endswith("music.sqlite")
