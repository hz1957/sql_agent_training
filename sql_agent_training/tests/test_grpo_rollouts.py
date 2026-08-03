import json
import sqlite3
from pathlib import Path

from sql_agent_training.train.grpo_rollouts import build_rollout_batch_from_config, run_grpo_rollouts


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
        json.dumps(
            [
                {
                    "uid": "music:0",
                    "db_id": "music",
                    "question": "List names.",
                    "query": "SELECT Name FROM Singer",
                }
            ]
        ),
        encoding="utf-8",
    )
    db_dir = root / "database" / "music"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "music.sqlite")
    try:
        conn.execute("CREATE TABLE Singer (Name TEXT)")
        conn.execute("INSERT INTO Singer VALUES ('Ada')")
        conn.commit()
    finally:
        conn.close()


def test_build_rollout_batch_from_config_uses_builtin_demo() -> None:
    batch = build_rollout_batch_from_config(
        {
            "dry_run": True,
            "tokenizer": {"kind": "whitespace"},
            "rollout": {
                "n": 2,
                "max_turns": 1,
                "scripted_responses": ["SELECT COUNT(*) FROM Singer", "SELECT Name FROM Singer"],
            },
        }
    )

    assert len(batch.groups) == 1
    assert batch.num_trajectories == 2
    assert [trajectory.reward for trajectory in batch.trajectories] == [0.0, 1.0]


def test_run_grpo_rollouts_writes_rollout_summary(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    _write_spider_dir(data_dir)
    output = tmp_path / "rollouts.jsonl"

    summary = run_grpo_rollouts(
        {
            "dry_run": False,
            "tokenizer": {"kind": "whitespace"},
            "data": {"data_dir": str(data_dir), "train_file": "train_spider.json"},
            "rollout": {"n": 2, "max_turns": 1, "train_limit": 1},
            "output": {"rollouts_jsonl": str(output)},
        }
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["groups"] == 1
    assert summary["trajectories"] == 2
    assert summary["mean_reward"] == 1.0
    assert len(rows) == 2
    assert rows[0]["uid"] == "music:0"
    assert "prompt" in rows[0]
    assert "response" in rows[0]
    assert "Question" in rows[0]["prompt"]
    assert rows[0]["response"].startswith("assistant:")


def test_run_grpo_rollouts_can_skip_rollout_text(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    _write_spider_dir(data_dir)
    output = tmp_path / "rollouts.jsonl"

    run_grpo_rollouts(
        {
            "dry_run": False,
            "tokenizer": {"kind": "whitespace"},
            "data": {"data_dir": str(data_dir), "train_file": "train_spider.json"},
            "rollout": {"n": 1, "max_turns": 1, "train_limit": 1},
            "output": {"rollouts_jsonl": str(output), "include_text": False},
        }
    )

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert "prompt" not in row
    assert "response" not in row
