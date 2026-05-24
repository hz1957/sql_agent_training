"""SQLite execution tool for Spider databases."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sql_agent_training.env.sql_safety import is_read_only_sql


@dataclass(frozen=True)
class SqlExecutionResult:
    """Structured result from executing a SQL query."""

    ok: bool
    rows: list[tuple[Any, ...]]
    error: str | None
    elapsed_seconds: float
    sql: str
    safety_reason: str


class SQLiteTool:
    """Safe read-only SQLite executor."""

    def __init__(self, timeout_steps: int = 100_000) -> None:
        self.timeout_steps = timeout_steps

    def execute(self, sqlite_path: str | Path, sql: str, *, copy_database: bool = True) -> SqlExecutionResult:
        """Execute a read-only SQL query against a SQLite database."""

        start = time.monotonic()
        safety = is_read_only_sql(sql)
        if not safety.is_safe:
            return SqlExecutionResult(False, [], safety.reason, 0.0, sql, safety.reason)

        source = Path(sqlite_path)
        if not source.exists():
            return SqlExecutionResult(False, [], f"database_not_found:{source}", 0.0, sql, "ok")

        try:
            if copy_database:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir) / source.name
                    shutil.copyfile(source, temp_path)
                    return self._execute_in_place(temp_path, sql, start)
            return self._execute_in_place(source, sql, start)
        except Exception as exc:
            return SqlExecutionResult(False, [], str(exc), time.monotonic() - start, sql, "ok")

    def _execute_in_place(self, sqlite_path: Path, sql: str, start: float) -> SqlExecutionResult:
        uri = f"file:{sqlite_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            conn.execute("PRAGMA query_only = ON")
            conn.set_progress_handler(lambda: 1, self.timeout_steps)
            rows = conn.execute(sql).fetchall()
            return SqlExecutionResult(True, rows, None, time.monotonic() - start, sql, "ok")
        finally:
            conn.close()
