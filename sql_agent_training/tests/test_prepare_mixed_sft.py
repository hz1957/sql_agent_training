import json
from pathlib import Path

from scripts.prepare_mixed_sft import build_mixed_sft


def _trajectory_row(uid: str, source_uid: str, *, rewrite: bool) -> dict:
    return {
        "uid": uid,
        "db_id": "music",
        "prompt": f"prompt for {uid}",
        "completion": "SELECT Name FROM Singer",
        "source_uid": source_uid,
        "source_rollout_id": f"{source_uid}:rollout",
        "is_rewrite_trajectory": rewrite,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_mixed_sft_enforces_ratio_and_avoids_gold_trajectory_overlap(tmp_path: Path) -> None:
    data_dir = tmp_path / "spider"
    data_dir.mkdir()
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
    spider_rows = [
        {
            "uid": f"question-{index}",
            "db_id": "music",
            "question": f"Question {index}",
            "query": "SELECT Name FROM Singer",
        }
        for index in range(7)
    ]
    (data_dir / "train_spider.json").write_text(json.dumps(spider_rows), encoding="utf-8")

    trajectory_path = tmp_path / "trajectory_sft.jsonl"
    _write_jsonl(
        trajectory_path,
        [
            _trajectory_row("d0", "question-0", rewrite=False),
            _trajectory_row("d1", "question-0", rewrite=False),
            _trajectory_row("d2", "question-1", rewrite=False),
            _trajectory_row("r0", "question-0", rewrite=True),
            _trajectory_row("r1", "question-2", rewrite=True),
            _trajectory_row("r2", "question-2", rewrite=True),
        ],
    )
    output_path = tmp_path / "mixed.jsonl"
    summary_path = tmp_path / "summary.json"

    summary = build_mixed_sft(
        data_dir=data_dir,
        train_file="train_spider.json",
        trajectory_jsonl=trajectory_path,
        output_jsonl=output_path,
        summary_json=summary_path,
        gold_count=2,
        direct_count=2,
        rewrite_count=2,
        seed=42,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    source_types = [row["source_type"] for row in rows]
    gold_source_uids = {row["source_uid"] for row in rows if row["source_type"] == "gold"}

    assert len(rows) == 6
    assert source_types.count("gold") == 2
    assert source_types.count("trajectory_direct") == 2
    assert source_types.count("trajectory_rewrite") == 2
    assert gold_source_uids.isdisjoint({"question-0", "question-1", "question-2"})
    assert summary["gold_trajectory_question_overlap"] == 0
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_build_mixed_sft_rejects_duplicate_trajectory_uids(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory_sft.jsonl"
    duplicate = _trajectory_row("duplicate", "question-0", rewrite=False)
    _write_jsonl(trajectory_path, [duplicate, duplicate])

    try:
        build_mixed_sft(
            data_dir=tmp_path,
            train_file="unused.json",
            trajectory_jsonl=trajectory_path,
            output_jsonl=tmp_path / "mixed.jsonl",
            summary_json=tmp_path / "summary.json",
            gold_count=0,
            direct_count=1,
            rewrite_count=0,
            seed=42,
        )
    except ValueError as exc:
        assert "duplicate uid" in str(exc)
    else:
        raise AssertionError("Expected duplicate trajectory uid values to raise ValueError")
