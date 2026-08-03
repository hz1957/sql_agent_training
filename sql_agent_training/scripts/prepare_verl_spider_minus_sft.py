"""Prepare verl Spider parquet files after excluding SFT-seen train questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_agent_training.train.verl_spider import build_verl_spider_rows, write_verl_spider_parquet  # noqa: E402


DEFAULT_SFT_JSONL = (
    "artifacts/sft_trajectory/qwen25_coder_14b_sft_merged_q1000_n4_target1600_seed43_t09/"
    "mixed_sft_ratio_gold3200_d1137_r463_seed42.jsonl"
)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_sft_source_uids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"SFT JSONL does not exist: {path}")

    source_uids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_uid = row.get("source_uid")
            if source_uid is None:
                raise ValueError(f"{path}:{line_number} is missing source_uid")
            source_uids.add(str(source_uid))
    if not source_uids:
        raise ValueError(f"SFT JSONL contains no source_uid values: {path}")
    return source_uids


def _row_uid(row: dict[str, Any]) -> str:
    return str(row["extra_info"]["uid"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/spider")
    parser.add_argument("--output-dir", default="data/verl_spider_minus_sft_gold3200_d1137_r463_seed42")
    parser.add_argument("--sft-jsonl", default=DEFAULT_SFT_JSONL)
    parser.add_argument("--train-file", default="train_spider.json")
    parser.add_argument("--validation-file", default="dev.json")
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Store SQLite paths relative to the current working directory instead of absolute paths.",
    )
    args = parser.parse_args()

    data_dir = _resolve_project_path(args.data_dir)
    output_dir = _resolve_project_path(args.output_dir)
    used_uids = _read_sft_source_uids(_resolve_project_path(args.sft_jsonl))
    absolute_paths = not args.relative_paths

    train_rows = build_verl_spider_rows(
        data_dir=data_dir,
        split_file=args.train_file,
        split="train",
        absolute_paths=absolute_paths,
    )
    validation_rows = build_verl_spider_rows(
        data_dir=data_dir,
        split_file=args.validation_file,
        split="validation",
        absolute_paths=absolute_paths,
    )
    remaining_train_rows = [row for row in train_rows if _row_uid(row) not in used_uids]

    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "validation.parquet"
    write_verl_spider_parquet(remaining_train_rows, train_path)
    write_verl_spider_parquet(validation_rows, validation_path)

    all_train_uids = {_row_uid(row) for row in train_rows}
    summary = {
        "data_dir": str(data_dir),
        "sft_jsonl": str(_resolve_project_path(args.sft_jsonl)),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "train_total": len(train_rows),
        "sft_unique_source_uids": len(used_uids),
        "sft_source_uids_in_train": len(all_train_uids & used_uids),
        "remaining_train_rows": len(remaining_train_rows),
        "validation_rows": len(validation_rows),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "summary_path": str(summary_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
