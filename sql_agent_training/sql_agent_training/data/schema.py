"""Schema prompt utilities for Spider databases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TableSchema:
    """A table and its columns for prompt rendering."""

    name: str
    columns: list[str]


@dataclass(frozen=True)
class DatabaseSchema:
    """Prompt-ready schema for one Spider database."""

    db_id: str
    tables: list[TableSchema]

    def to_prompt(self) -> str:
        """Render schema as compact text for model prompts."""

        lines = [f"Database: {self.db_id}"]
        for table in self.tables:
            columns = ", ".join(table.columns)
            lines.append(f"- {table.name}({columns})")
        return "\n".join(lines)


def load_tables_json(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load Spider tables.json indexed by db_id."""

    table_path = Path(path)
    with table_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return {record["db_id"]: record for record in records}


def schema_from_spider_record(record: dict[str, Any]) -> DatabaseSchema:
    """Build a DatabaseSchema from one Spider tables.json record."""

    table_names = list(record.get("table_names_original") or record.get("table_names") or [])
    columns_by_table: list[list[str]] = [[] for _ in table_names]
    for table_index, column_name in record.get("column_names_original", []):
        if table_index == -1:
            continue
        if 0 <= table_index < len(columns_by_table):
            columns_by_table[table_index].append(str(column_name))

    tables = [TableSchema(name=name, columns=columns_by_table[index]) for index, name in enumerate(table_names)]
    return DatabaseSchema(db_id=record["db_id"], tables=tables)


def build_schema_prompt(db_id: str, tables_index: dict[str, dict[str, Any]]) -> str:
    """Render a schema prompt from a loaded Spider tables index."""

    if db_id not in tables_index:
        raise KeyError(f"db_id {db_id!r} not found in tables.json")
    return schema_from_spider_record(tables_index[db_id]).to_prompt()
