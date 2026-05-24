"""Action parsing for SQL agent rollouts."""

from __future__ import annotations

import re


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:sql)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def extract_sql_candidate(text: str) -> str | None:
    """Extract a plain SQL candidate from model output.

    The preferred protocol is simple: the model returns one SQLite SELECT/WITH query.
    Optional markdown fences or prefixes such as `SQL:` and `FINAL:` are tolerated.
    """

    stripped = _strip_markdown_fence(text).strip()
    for prefix in ("FINAL:", "Final:", "final:", "SQL:", "Sql:", "sql:"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
            break

    match = re.search(r"\b(select|with)\b", stripped, flags=re.IGNORECASE)
    if not match:
        return None
    sql = stripped[match.start() :].strip()
    if "```" in sql:
        sql = sql.split("```", 1)[0].strip()
    return sql or None
