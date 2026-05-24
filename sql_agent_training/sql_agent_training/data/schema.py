"""Schema prompt utilities for Spider databases."""

from __future__ import annotations

import json
import sqlite3
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


def schema_from_sqlite(db_id: str, sqlite_path: str | Path) -> DatabaseSchema:
    """Introspect a SQLite database into a DatabaseSchema."""

    path = Path(sqlite_path)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        tables: list[TableSchema] = []
        for (table_name,) in table_rows:
            column_rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            columns = [str(row[1]) for row in column_rows]
            tables.append(TableSchema(name=str(table_name), columns=columns))
        return DatabaseSchema(db_id=db_id, tables=tables)
    finally:
        conn.close()


def build_schema_prompt(db_id: str, tables_index: dict[str, dict[str, Any]]) -> str:
    """Render a schema prompt from a loaded Spider tables index."""

    if db_id not in tables_index:
        raise KeyError(f"db_id {db_id!r} not found in tables.json")
    return schema_from_spider_record(tables_index[db_id]).to_prompt()


def build_schema_cache(tables_index: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Build prompt text for every db_id in a Spider tables index."""

    return {db_id: schema_from_spider_record(record).to_prompt() for db_id, record in tables_index.items()}


def write_schema_cache(tables_index: dict[str, dict[str, Any]], output_path: str | Path) -> int:
    """Write schema prompts to JSON and return number of schemas."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = build_schema_cache(tables_index)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)
    return len(cache)


def load_schema_cache(path: str | Path) -> dict[str, str]:
    """Load schema prompt cache."""

    cache_path = Path(path)
    with cache_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
