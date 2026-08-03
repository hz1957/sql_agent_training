"""SFT data formatting for Spider question/schema to SQL training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from sql_agent_training.agent.prompts import build_sft_prompt
from sql_agent_training.data.schema import build_schema_prompt
from sql_agent_training.data.spider_dataset import SpiderExample


def format_sft_record(example: SpiderExample, tables_index: dict) -> dict[str, str]:
    """Format one Spider example into prompt/completion text."""

    schema_prompt = build_schema_prompt(example.db_id, tables_index)
    return {
        "uid": example.uid,
        "db_id": example.db_id,
        "prompt": build_sft_prompt(example.question, schema_prompt),
        "completion": example.gold_sql.strip(),
    }


def write_sft_jsonl(examples: Iterable[SpiderExample], tables_index: dict, output_path: str | Path) -> int:
    """Write SFT records as JSONL and return the number of records."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(format_sft_record(example, tables_index), ensure_ascii=False) + "\n")
            count += 1
    return count
