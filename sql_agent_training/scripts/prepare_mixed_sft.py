"""Build a controlled mixture of Spider gold SQL and verified agent trajectories.

Example:
    uv run --no-sync python scripts/prepare_mixed_sft.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_agent_training.data.schema import load_tables_json  # noqa: E402
from sql_agent_training.data.sft_formatter import format_sft_record  # noqa: E402
from sql_agent_training.data.spider_dataset import load_spider_file  # noqa: E402

DEFAULT_TRAJECTORY_DIR = "artifacts/sft_trajectory/qwen25_coder_14b_sft_merged_q1000_n4_target1600_seed43_t09"


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"uid", "db_id", "prompt", "completion", "source_uid", "is_rewrite_trajectory"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} is missing fields: {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _select_diverse_rows(
    rows: list[dict[str, Any]],
    count: int,
    *,
    rng: random.Random,
    avoid_source_uids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"Requested {count} trajectory records, but only {len(rows)} are available")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_uid"])].append(row)
    for group in grouped.values():
        rng.shuffle(group)

    source_uids = list(grouped)
    rng.shuffle(source_uids)
    if avoid_source_uids:
        source_uids.sort(key=lambda uid: uid in avoid_source_uids)

    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for source_uid in source_uids:
            group = grouped[source_uid]
            if round_index >= len(group):
                continue
            selected.append(group[round_index])
            added = True
            if len(selected) == count:
                break
        if not added:
            break
        round_index += 1
    if len(selected) != count:
        raise RuntimeError(f"Selected {len(selected)} trajectory records, expected {count}")
    return selected


def build_mixed_sft(
    *,
    data_dir: Path,
    train_file: str,
    trajectory_jsonl: Path,
    output_jsonl: Path,
    summary_json: Path,
    gold_count: int,
    direct_count: int,
    rewrite_count: int,
    seed: int,
) -> dict[str, Any]:
    if min(gold_count, direct_count, rewrite_count) < 0:
        raise ValueError("gold/direct/rewrite counts must be non-negative")

    trajectory_rows = _read_jsonl(trajectory_jsonl)
    trajectory_uids = [str(row["uid"]) for row in trajectory_rows]
    if len(set(trajectory_uids)) != len(trajectory_uids):
        raise ValueError(f"Trajectory JSONL contains duplicate uid values: {trajectory_jsonl}")

    direct_rows = [row for row in trajectory_rows if not bool(row["is_rewrite_trajectory"])]
    rewrite_rows = [row for row in trajectory_rows if bool(row["is_rewrite_trajectory"])]
    rng = random.Random(seed)

    selected_rewrite = _select_diverse_rows(rewrite_rows, rewrite_count, rng=rng)
    rewrite_source_uids = {str(row["source_uid"]) for row in selected_rewrite}
    selected_direct = _select_diverse_rows(
        direct_rows,
        direct_count,
        rng=rng,
        avoid_source_uids=rewrite_source_uids,
    )

    all_trajectory_source_uids = {str(row["source_uid"]) for row in trajectory_rows}
    spider_examples = load_spider_file(data_dir / train_file)
    gold_candidates = [example for example in spider_examples if example.uid not in all_trajectory_source_uids]
    if gold_count > len(gold_candidates):
        raise ValueError(
            f"Requested {gold_count} gold records after excluding trajectory questions, "
            f"but only {len(gold_candidates)} are available"
        )
    selected_gold = rng.sample(gold_candidates, gold_count)
    tables_index = load_tables_json(data_dir / "tables.json")

    mixed_rows: list[dict[str, Any]] = []
    for example in selected_gold:
        mixed_rows.append(
            {
                **format_sft_record(example, tables_index),
                "source_uid": example.uid,
                "source_type": "gold",
            }
        )
    for row in selected_direct:
        mixed_rows.append({**row, "source_type": "trajectory_direct"})
    for row in selected_rewrite:
        mixed_rows.append({**row, "source_type": "trajectory_rewrite"})
    rng.shuffle(mixed_rows)

    expected_count = gold_count + direct_count + rewrite_count
    if len(mixed_rows) != expected_count:
        raise RuntimeError(f"Built {len(mixed_rows)} mixed records, expected {expected_count}")
    _write_jsonl(output_jsonl, mixed_rows)

    direct_source_uids = {str(row["source_uid"]) for row in selected_direct}
    selected_trajectory_source_uids = direct_source_uids | rewrite_source_uids
    summary: dict[str, Any] = {
        "seed": seed,
        "data_dir": str(data_dir),
        "train_file": train_file,
        "trajectory_jsonl": str(trajectory_jsonl),
        "output_jsonl": str(output_jsonl),
        "total_records": len(mixed_rows),
        "gold_records": gold_count,
        "direct_trajectory_records": direct_count,
        "rewrite_trajectory_records": rewrite_count,
        "available_direct_trajectories": len(direct_rows),
        "available_rewrite_trajectories": len(rewrite_rows),
        "gold_unique_questions": len({example.uid for example in selected_gold}),
        "direct_unique_questions": len(direct_source_uids),
        "rewrite_unique_questions": len(rewrite_source_uids),
        "trajectory_unique_questions": len(selected_trajectory_source_uids),
        "all_unique_questions": len({str(row["source_uid"]) for row in mixed_rows}),
        "excluded_gold_questions": len(all_trajectory_source_uids),
        "gold_trajectory_question_overlap": len(
            {example.uid for example in selected_gold} & selected_trajectory_source_uids
        ),
    }
    _write_json(summary_json, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/spider")
    parser.add_argument("--train-file", default="train_spider.json")
    parser.add_argument("--trajectory-jsonl", default=f"{DEFAULT_TRAJECTORY_DIR}/trajectory_sft.jsonl")
    parser.add_argument(
        "--output-jsonl",
        default=f"{DEFAULT_TRAJECTORY_DIR}/mixed_sft_gold3200_direct1137_rewrite463_seed42.jsonl",
    )
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--gold-count", type=int, default=3200)
    parser.add_argument("--direct-count", type=int, default=1137)
    parser.add_argument("--rewrite-count", type=int, default=463)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_jsonl = _resolve_project_path(args.output_jsonl)
    summary_json = (
        _resolve_project_path(args.summary_json)
        if args.summary_json
        else output_jsonl.with_name(f"{output_jsonl.stem}_summary.json")
    )
    summary = build_mixed_sft(
        data_dir=_resolve_project_path(args.data_dir),
        train_file=args.train_file,
        trajectory_jsonl=_resolve_project_path(args.trajectory_jsonl),
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        gold_count=args.gold_count,
        direct_count=args.direct_count,
        rewrite_count=args.rewrite_count,
        seed=args.seed,
    )
    print(json.dumps({**summary, "summary_json": str(summary_json)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
