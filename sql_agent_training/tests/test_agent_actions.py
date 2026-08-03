from sql_agent_training.agent.actions import extract_sql_candidate


def test_extract_plain_sql_candidate() -> None:
    assert extract_sql_candidate("SELECT Name FROM Singer;") == "SELECT Name FROM Singer;"


def test_extract_markdown_sql_candidate() -> None:
    assert extract_sql_candidate("```sql\nSELECT count(*) FROM singer;\n```") == "SELECT count(*) FROM singer;"


def test_extract_final_prefixed_sql_candidate() -> None:
    assert extract_sql_candidate("FINAL: SELECT 1") == "SELECT 1"


def test_extract_sql_candidate_ignores_non_sql_text() -> None:
    assert extract_sql_candidate("I think the answer is one.") is None
