import sqlite3
from pathlib import Path

from sql_agent_training.env.sql_safety import is_read_only_sql
from sql_agent_training.env.sqlite_tool import SQLiteTool


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE Singer (Name TEXT)")
        conn.execute("INSERT INTO Singer VALUES ('Ada')")
        conn.commit()
    finally:
        conn.close()


def test_sql_safety_allows_select() -> None:
    result = is_read_only_sql("SELECT Name FROM Singer")
    assert result.is_safe


def test_sql_safety_rejects_write() -> None:
    result = is_read_only_sql("DELETE FROM Singer")
    assert not result.is_safe
    assert result.reason.startswith("disallowed_start") or result.reason.startswith("disallowed_keyword")


def test_sqlite_tool_executes_read_only_query(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)

    result = SQLiteTool().execute(db_path, "SELECT Name FROM Singer")

    assert result.ok
    assert result.rows == [("Ada",)]


def test_sqlite_tool_rejects_unsafe_query(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)

    result = SQLiteTool().execute(db_path, "DROP TABLE Singer")

    assert not result.ok
    assert "disallowed" in (result.error or "")
