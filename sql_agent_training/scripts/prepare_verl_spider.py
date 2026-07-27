"""Prepare Spider parquet files for verl SQL-agent GRPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_agent_training.train.verl_spider import prepare_verl_spider_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Spider train/validation parquet files for verl.")
    parser.add_argument("--data-dir", default="data/spider")
    parser.add_argument("--output-dir", default="data/verl_spider")
    parser.add_argument("--train-file", default="train_spider.json")
    parser.add_argument("--validation-file", default="dev.json")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Store SQLite paths relative to the current working directory instead of absolute paths.",
    )
    args = parser.parse_args()

    summary = prepare_verl_spider_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_file=args.train_file,
        validation_file=args.validation_file,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        absolute_paths=not args.relative_paths,
    )
    print(json.dumps(summary.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
