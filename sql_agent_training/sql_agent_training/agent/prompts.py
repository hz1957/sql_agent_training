"""Prompt templates for Spider text-to-SQL and SQL agent rollouts."""

SFT_SYSTEM_PROMPT = """You are a careful text-to-SQL model.
Generate a single SQLite SQL query that answers the user's question.
Use only the tables and columns shown in the schema.
Return only SQL, without markdown fences or explanation."""

AGENT_SYSTEM_PROMPT = """You are a SQL agent.
Use the database schema to solve the question.
Return one read-only SQLite SELECT query.
If the previous SQL failed or returned the wrong result, rewrite the SQL.
Return only SQL, without markdown fences or explanation."""


def build_sft_prompt(question: str, schema_prompt: str) -> str:
    """Build the SFT prompt for question/schema to SQL training."""

    return f"{SFT_SYSTEM_PROMPT}\n\n## Schema\n{schema_prompt}\n\n## Question\n{question}\n\nSQL:"


def build_agent_prompt(question: str, schema_prompt: str) -> str:
    """Build the SQL rewrite-agent prompt for GRPO rollouts."""

    return f"{AGENT_SYSTEM_PROMPT}\n\n## Schema\n{schema_prompt}\n\n## Question\n{question}\n\nSQL:"
