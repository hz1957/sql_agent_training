"""Prepare Spider rows in the parquet format expected by VERL RLHFDataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from sql_agent_training.agent.prompts import build_agent_prompt
from sql_agent_training.data.schema import build_schema_prompt, load_tables_json
from sql_agent_training.data.spider_dataset import expected_sqlite_path, load_spider_file


def write_verl_spider_parquet(
    *,
    data_dir: Path,
    split_file: str,
    output_path: Path,
    limit: int | None = None,
    agent_name: str = "sql_agent",
) -> int:
    """Write a VERL-compatible Spider parquet file."""

    examples = load_spider_file(data_dir / split_file)
    if limit is not None:
        examples = examples[:limit]
    tables_index = load_tables_json(data_dir / "tables.json")
    rows = []
    for index, example in enumerate(examples):
        schema_prompt = build_schema_prompt(example.db_id, tables_index)
        prompt = build_agent_prompt(example.question, schema_prompt)
        sqlite_path = expected_sqlite_path(data_dir, example.db_id)
        rows.append(
            {
                "data_source": "spider",
                "prompt": [{"role": "user", "content": prompt}],
                "agent_name": agent_name,
                "extra_info": {
                    "index": index,
                    "uid": example.uid,
                    "question": example.question,
                    "db_id": example.db_id,
                    "schema_prompt": schema_prompt,
                    "gold_sql": example.gold_sql,
                    "sqlite_path": str(sqlite_path),
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VERL parquet data from Spider.")
    parser.add_argument("--data-dir", default="data/spider")
    parser.add_argument("--split-file", default="train_spider.json")
    parser.add_argument("--output", default="artifacts/verl/spider_train.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--agent-name", default="sql_agent")
    args = parser.parse_args()

    count = write_verl_spider_parquet(
        data_dir=Path(args.data_dir),
        split_file=args.split_file,
        output_path=Path(args.output),
        limit=args.limit,
        agent_name=args.agent_name,
    )
    print(f"wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
