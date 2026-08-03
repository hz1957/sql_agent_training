"""Prepare Spider SQL-agent data for verl GRPO/AgentLoop training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset

from sql_agent_training.agent.prompts import build_write_query_prompt
from sql_agent_training.data.schema import build_schema_prompt, load_tables_json
from sql_agent_training.data.spider_dataset import expected_sqlite_path, load_spider_file

DEFAULT_AGENT_NAME = "sql_agent"
DEFAULT_DATA_SOURCE = "spider_sql_agent"


@dataclass(frozen=True)
class VerlSpiderDatasetSummary:
    """Summary of generated verl parquet files."""

    train_path: str
    validation_path: str
    num_train_rows: int
    num_validation_rows: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _limit_rows(rows: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return rows
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return rows[:limit]


def build_verl_spider_rows(
    *,
    data_dir: str | Path,
    split_file: str | Path,
    split: str,
    max_samples: int | None = None,
    absolute_paths: bool = True,
    agent_name: str = DEFAULT_AGENT_NAME,
    data_source: str = DEFAULT_DATA_SOURCE,
) -> list[dict[str, Any]]:
    """Build rows consumable by verl's default RLHFDataset.

    Each row keeps the first user prompt in ``prompt`` and stores SQL-agent
    environment fields in ``extra_info`` so the custom AgentLoop can run SQLite
    execution checks and compute the final Spider execution reward.
    """

    root = Path(data_dir)
    examples = _limit_rows(load_spider_file(root / split_file), max_samples)
    tables_index = load_tables_json(root / "tables.json")

    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        schema_prompt = build_schema_prompt(example.db_id, tables_index)
        sqlite_path = expected_sqlite_path(root, example.db_id)
        if absolute_paths:
            sqlite_path = sqlite_path.resolve()

        prompt = build_write_query_prompt(example.question, schema_prompt)
        rows.append(
            {
                "data_source": data_source,
                "agent_name": agent_name,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "sql",
                "reward_model": {"style": "rule", "ground_truth": example.gold_sql},
                "extra_info": {
                    "index": index,
                    "split": split,
                    "uid": example.uid,
                    "question": example.question,
                    "db_id": example.db_id,
                    "gold_sql": example.gold_sql,
                    "schema_prompt": schema_prompt,
                    "sqlite_path": str(sqlite_path),
                },
            }
        )
    return rows


def write_verl_spider_parquet(rows: list[dict[str, Any]], output_path: str | Path) -> int:
    """Write verl rows to parquet and return the row count."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))
    return len(rows)


def prepare_verl_spider_dataset(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    train_file: str | Path = "train_spider.json",
    validation_file: str | Path = "dev.json",
    train_limit: int | None = None,
    validation_limit: int | None = None,
    absolute_paths: bool = True,
) -> VerlSpiderDatasetSummary:
    """Prepare train/validation parquet files for verl GRPO."""

    output_root = Path(output_dir)
    train_rows = build_verl_spider_rows(
        data_dir=data_dir,
        split_file=train_file,
        split="train",
        max_samples=train_limit,
        absolute_paths=absolute_paths,
    )
    validation_rows = build_verl_spider_rows(
        data_dir=data_dir,
        split_file=validation_file,
        split="validation",
        max_samples=validation_limit,
        absolute_paths=absolute_paths,
    )

    train_path = output_root / "train.parquet"
    validation_path = output_root / "validation.parquet"
    write_verl_spider_parquet(train_rows, train_path)
    write_verl_spider_parquet(validation_rows, validation_path)
    return VerlSpiderDatasetSummary(
        train_path=str(train_path),
        validation_path=str(validation_path),
        num_train_rows=len(train_rows),
        num_validation_rows=len(validation_rows),
    )
