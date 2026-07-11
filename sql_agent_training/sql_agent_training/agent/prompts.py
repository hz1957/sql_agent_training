"""Prompt templates for Spider text-to-SQL and SQL agent rollouts."""

SFT_SYSTEM_PROMPT = """You are a careful text-to-SQL model.
Generate a single SQLite SQL query that answers the user's question.
Use only the tables and columns shown in the schema.
Return only SQL, without markdown fences or explanation."""

AGENT_SYSTEM_PROMPT = """You are a SQL agent.
Use the database schema to solve the question.
Return one read-only SQLite SELECT query.
If the previous SQL failed to execute, use the tool error to rewrite the SQL.
Return only SQL, without markdown fences or explanation."""


def build_sft_prompt(question: str, schema_prompt: str) -> str:
    """Build the SFT prompt for question/schema to SQL training."""

    return f"{SFT_SYSTEM_PROMPT}\n\n## Schema\n{schema_prompt}\n\n## Question\n{question}\n\nSQL:"


def build_agent_prompt(
    question: str,
    schema_prompt: str,
    *,
    previous_failed_sql: str | None = None,
    previous_error: str | None = None,
) -> str:
    """Build the SQL rewrite-agent prompt for GRPO rollouts."""

    sections = [
        AGENT_SYSTEM_PROMPT,
        f"## Question\n{question}",
        f"## Schema\n{schema_prompt}",
    ]
    if previous_failed_sql:
        sections.append(f"## Previous failed SQL\n{previous_failed_sql}")
    if previous_error:
        sections.append(f"## Previous error\n{previous_error}")
    sections.append("SQL:")
    return "\n\n".join(sections)
