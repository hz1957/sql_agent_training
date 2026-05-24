"""Spider execution reward wrapper."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _execute_rows(sqlite_path: Path, sql: str) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _fallback_exec_match(generated_sql: str, gold_sql: str, sqlite_path: Path) -> float:
    generated_rows = _execute_rows(sqlite_path, generated_sql)
    gold_rows = _execute_rows(sqlite_path, gold_sql)
    return 1.0 if generated_rows == gold_rows else 0.0


def spider_execution_reward(generated_sql: str, gold_sql: str, sqlite_path: str | Path, *, raise_on_error: bool = False) -> float:
    """Return 1.0 when generated SQL execution matches gold SQL, else 0.0."""

    try:
        db_path = Path(sqlite_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file does not exist: {db_path}")
        try:
            from sql_agent_training.reward.spider_eval.exec_eval import eval_exec_match
        except ImportError:
            return _fallback_exec_match(generated_sql, gold_sql, db_path)

        score = eval_exec_match(
            db=str(db_path.resolve()),
            p_str=generated_sql,
            g_str=gold_sql,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
        return 1.0 if score == 1 else 0.0
    except Exception:
        if raise_on_error:
            raise
        logger.exception("Failed to compute Spider execution reward.")
        return 0.0
