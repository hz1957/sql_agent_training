from sql_agent_training.train.verl_sql_agent_loop import _as_dict, _sample_fields


def test_verl_sql_agent_loop_module_imports_without_verl() -> None:
    fields = _sample_fields(
        {
            "raw_prompt": [{"role": "user", "content": "SQL:"}],
            "extra_info": {
                "uid": "music:0",
                "question": "List names.",
                "db_id": "music",
                "schema_prompt": "Database: music\n- Singer(Name)",
                "gold_sql": "SELECT Name FROM Singer",
                "sqlite_path": "/tmp/music.sqlite",
            },
        }
    )

    assert fields["uid"] == "music:0"
    assert fields["initial_prompt"] == "SQL:"


def test_as_dict_rejects_non_mapping_extra_info() -> None:
    try:
        _as_dict("bad")
    except TypeError as exc:
        assert "extra_info" in str(exc)
    else:
        raise AssertionError("_as_dict should reject non-mapping extra_info")
