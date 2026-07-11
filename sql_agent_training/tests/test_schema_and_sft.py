from sql_agent_training.data.schema import build_schema_prompt, schema_from_spider_record
from sql_agent_training.data.sft_formatter import format_sft_record
from sql_agent_training.data.spider_dataset import SpiderExample


def test_schema_prompt_from_spider_record() -> None:
    record = {
        "db_id": "music",
        "table_names_original": ["Singer", "Song"],
        "column_names_original": [[-1, "*"], [0, "Singer_ID"], [0, "Name"], [1, "Song_ID"], [1, "Title"]],
    }

    schema = schema_from_spider_record(record)

    assert schema.db_id == "music"
    assert "Singer(Singer_ID, Name)" in schema.to_prompt()
    assert "Song(Song_ID, Title)" in schema.to_prompt()


def test_sft_record_uses_gold_sql_as_completion() -> None:
    tables = {
        "music": {
            "db_id": "music",
            "table_names_original": ["Singer"],
            "column_names_original": [[-1, "*"], [0, "Name"]],
        }
    }
    example = SpiderExample(uid="1", db_id="music", question="List singer names.", gold_sql="SELECT Name FROM Singer")

    record = format_sft_record(example, tables)

    assert "List singer names." in record["prompt"]
    assert "Singer(Name)" in record["prompt"]
    assert record["completion"] == "SELECT Name FROM Singer"
    assert "SELECT Name FROM Singer" not in record["prompt"]


def test_build_schema_prompt_missing_db_raises() -> None:
    try:
        build_schema_prompt("missing", {})
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected KeyError")
