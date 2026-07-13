"""Prompt templates for Spider text-to-SQL and SQL agent rollouts."""

SFT_SYSTEM_PROMPT = """You are a careful text-to-SQL model.
Generate a single SQLite SQL query that answers the user's question.
Use only the tables and columns shown in the schema.
Return only SQL, without markdown fences or explanation."""

WRITE_QUERY_PROMPT = """You are an agent designed to interact with a SQLite database.
Given an input question, create a syntactically correct read-only SQLite SELECT query to run.
Use only the tables and columns shown in the schema description.
Be careful to not query for columns that do not exist, and pay attention to which column is in which table.
Return only SQL, without markdown fences or explanation."""

CHECK_QUERY_PROMPT = """You are a SQL expert with strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using columns or tables that are not in the schema
- Joining on the wrong columns
- Using the wrong aggregation, grouping, filtering, ordering, or set operation
- Data type mismatch in predicates
- Explicit query execution failures
- Clearly unreasonable query execution results

After listing mistakes if any, conclude with exactly one of these phrases:
THE QUERY IS INCORRECT.
THE QUERY IS CORRECT.

Do not write a corrected query in the response."""

REWRITE_QUERY_PROMPT = """You are an agent designed to interact with a SQLite database.
Rewrite the previous SQLite query to fix errors based on the execution result and checker feedback.
The goal is to answer the original question.
Use only the tables and columns shown in the schema description.
Return only SQL, without markdown fences or explanation."""

AGENT_SYSTEM_PROMPT = WRITE_QUERY_PROMPT


def build_sft_prompt(question: str, schema_prompt: str) -> str:
    """Build the SFT prompt for question/schema to SQL training."""

    return f"{SFT_SYSTEM_PROMPT}\n\n## Schema\n{schema_prompt}\n\n## Question\n{question}\n\nSQL:"


def build_write_query_prompt(question: str, schema_prompt: str) -> str:
    """Build the initial SQL writer prompt."""

    return "\n\n".join(
        [
            WRITE_QUERY_PROMPT,
            f"## Question\n{question}",
            f"## Schema\n{schema_prompt}",
            "SQL:",
        ]
    )


def build_check_query_prompt(question: str, schema_prompt: str, query: str, execution: str) -> str:
    """Build the SQL checker prompt."""

    return "\n\n".join(
        [
            CHECK_QUERY_PROMPT,
            f"## Question\n{question}",
            f"## Schema\n{schema_prompt}",
            f"## Query\n{query}",
            f"## Execution result\n{execution}",
            "Feedback:",
        ]
    )


def build_rewrite_query_prompt(
    question: str,
    schema_prompt: str,
    *,
    previous_sql: str,
    previous_execution: str,
    feedback: str,
) -> str:
    """Build the SQL rewrite prompt after checker feedback."""

    return "\n\n".join(
        [
            REWRITE_QUERY_PROMPT,
            f"## Question\n{question}",
            f"## Schema\n{schema_prompt}",
            f"## Previous query\n{previous_sql}",
            f"## Previous execution result\n{previous_execution}",
            f"## Feedback\n{feedback}",
            "SQL:",
        ]
    )


def build_agent_prompt(
    question: str,
    schema_prompt: str,
    *,
    previous_failed_sql: str | None = None,
    previous_error: str | None = None,
) -> str:
    """Build the legacy SQL-agent prompt for GRPO rollouts."""

    if previous_failed_sql and previous_error:
        return build_rewrite_query_prompt(
            question,
            schema_prompt,
            previous_sql=previous_failed_sql,
            previous_execution=previous_error,
            feedback=previous_error,
        )

    sections = [AGENT_SYSTEM_PROMPT, f"## Question\n{question}", f"## Schema\n{schema_prompt}"]
    if previous_failed_sql:
        sections.append(f"## Previous failed SQL\n{previous_failed_sql}")
    if previous_error:
        sections.append(f"## Previous error\n{previous_error}")
    sections.append("SQL:")
    return "\n\n".join(sections)
