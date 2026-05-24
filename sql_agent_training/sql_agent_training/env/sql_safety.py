"""Read-only SQL safety checks."""

from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import sqlparse
except ImportError:  # pragma: no cover - optional fallback
    sqlparse = None  # type: ignore


DISALLOWED_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "REPLACE",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "REINDEX",
}


@dataclass(frozen=True)
class SqlSafetyResult:
    """Result of a read-only SQL safety check."""

    is_safe: bool
    reason: str


def _statements(sql: str) -> list[str]:
    if sqlparse is None:
        parts = [part.strip() for part in sql.split(";") if part.strip()]
        return parts
    return [str(statement).strip() for statement in sqlparse.parse(sql) if str(statement).strip()]


def is_read_only_sql(sql: str) -> SqlSafetyResult:
    """Allow only a single SQLite SELECT/WITH query."""

    stripped = sql.strip()
    if not stripped:
        return SqlSafetyResult(False, "empty_sql")

    statements = _statements(stripped)
    if len(statements) != 1:
        return SqlSafetyResult(False, "multiple_statements")

    normalized = re.sub(r"\s+", " ", statements[0]).strip()
    upper = normalized.upper()
    first_token_match = re.match(r"^[A-Z_]+", upper)
    first_token = first_token_match.group(0) if first_token_match else ""
    if first_token not in {"SELECT", "WITH"}:
        return SqlSafetyResult(False, f"disallowed_start:{first_token or 'unknown'}")

    for keyword in DISALLOWED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            return SqlSafetyResult(False, f"disallowed_keyword:{keyword}")

    return SqlSafetyResult(True, "ok")
