"""Prepare or verify Spider assets for SQL agent training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_agent_training.data.schema import load_tables_json
from sql_agent_training.data.spider_dataset import (
    SpiderExample,
    load_hf_spider,
    load_spider_file,
    verify_spider_assets,
    write_spider_json,
)


def _load_examples_if_present(data_dir: Path, train_file: str) -> list:
    path = data_dir / train_file
    if not path.exists():
        return []
    return load_spider_file(path)


def _download_hf_text(data_dir: Path, train_file: str, validation_file: str) -> dict[str, int]:
    train_examples = load_hf_spider("train")
    # xlangai/spider uses validation rather than dev in Hugging Face naming.
    validation_examples = load_hf_spider("validation")
    return {
        "train_written": write_spider_json(train_examples, data_dir / train_file),
        "validation_written": write_spider_json(validation_examples, data_dir / validation_file),
    }


def _verify_split(
    data_dir: Path,
    split_file: str,
    schema_db_ids: set[str] | None = None,
) -> tuple[list[SpiderExample], dict]:
    examples = _load_examples_if_present(data_dir, split_file)
    summary = verify_spider_assets(data_dir, examples if examples else None)
    summary["split_file"] = split_file
    summary["split_file_exists"] = (data_dir / split_file).exists()
    if examples:
        summary["num_examples"] = len(examples)
    if schema_db_ids is not None and examples:
        missing_schema_ids = {example.db_id for example in examples if example.db_id not in schema_db_ids}
        summary["missing_schema_ids"] = sorted(missing_schema_ids)
        summary["ok"] = bool(summary["ok"] and not summary["missing_schema_ids"])
    return examples, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or verify Spider text and database assets.")
    parser.add_argument("--data-dir", default="data/spider")
    parser.add_argument("--train-file", default="train_spider.json")
    parser.add_argument("--validation-file", default="dev.json")
    parser.add_argument("--download-hf-text", action="store_true", help="Download xlangai/spider text labels.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"data_dir": str(data_dir)}
    if args.download_hf_text:
        summary["hf_text"] = _download_hf_text(data_dir, args.train_file, args.validation_file)

    tables_path = data_dir / "tables.json"
    tables_index = None
    if tables_path.exists():
        tables_index = load_tables_json(tables_path)
        summary["num_schemas"] = len(tables_index)
    else:
        summary["num_schemas"] = 0

    schema_db_ids = set(tables_index) if tables_index is not None else None
    train_examples, train_summary = _verify_split(data_dir, args.train_file, schema_db_ids)
    validation_examples, validation_summary = _verify_split(data_dir, args.validation_file, schema_db_ids)
    summary["train"] = train_summary
    summary["validation"] = validation_summary

    summary["ok"] = bool(train_summary["ok"] and validation_summary["ok"])
    summary["num_train_examples"] = len(train_examples)
    summary["num_validation_examples"] = len(validation_examples)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not summary["ok"]:
        raise SystemExit(1)

    if not args.verify_only:
        print("Spider text/assets are prepared. Use --verify-only for validation-only runs.")


if __name__ == "__main__":
    main()
