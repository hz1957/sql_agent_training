from sql_agent_training.train.verl_sql_agent_loop import _as_dict, _rollout_extra_fields, _sample_fields


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


def test_rollout_extra_fields_do_not_duplicate_verl_sample_keys() -> None:
    fields = _rollout_extra_fields(
        final_sql="SELECT Name FROM Singer",
        final_sql_source="checker_accept",
        num_execute_calls=1,
        num_check_calls=1,
        num_parse_errors=0,
    )

    assert fields == {
        "final_sql": "SELECT Name FROM Singer",
        "final_sql_source": "checker_accept",
        "num_execute_calls": 1,
        "num_check_calls": 1,
        "num_parse_errors": 0,
    }
    assert "uid" not in fields
    assert "db_id" not in fields
