import pytest

from sql_agent_training.train.verl_sql_agent_loop import (
    _TRANSITION_COUNTERS,
    _as_dict,
    _compute_transition_rewards,
    _normalize_reward_gamma,
    _normalize_reward_scheme,
    _normalize_transition_selection,
    _rollout_extra_fields,
    _sample_fields,
    _select_transition_index,
)


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
        final_execution_reward=1.0,
        transition_reward=0.9,
        reward_scheme="chain_final",
        reward_gamma=0.9,
        reward_discount_power=1,
        final_success_turn_index=1,
        selected_transition_index=0,
        num_sql_transitions=2,
        transition_selection="round_robin",
        selected_transition_turn_index=0,
        selected_transition_agent_step="write_query",
        selected_transition_tool_ok=False,
    )

    assert fields == {
        "final_sql": "SELECT Name FROM Singer",
        "final_sql_source": "checker_accept",
        "num_execute_calls": 1,
        "num_check_calls": 1,
        "num_parse_errors": 0,
        "final_execution_reward": 1.0,
        "trajectory_reward": 1.0,
        "transition_reward": 0.9,
        "reward_scheme": "chain_final",
        "reward_gamma": 0.9,
        "reward_discount_power": 1,
        "final_success_turn_index": 1,
        "selected_transition_index": 0,
        "num_sql_transitions": 2,
        "transition_selection": "round_robin",
        "selected_transition_turn_index": 0,
        "selected_transition_agent_step": "write_query",
        "selected_transition_tool_ok": False,
    }
    assert "uid" not in fields
    assert "db_id" not in fields


def test_final_shared_reward_preserves_previous_scalar_reward() -> None:
    rewards = _compute_transition_rewards(
        final_execution_reward=1.0,
        reward_scheme="final_shared",
        reward_gamma=0.9,
        num_transitions=3,
    )

    assert rewards == [(1.0, 0)]


def test_chain_final_reward_discounts_previous_sql_actions() -> None:
    rewards = _compute_transition_rewards(
        final_execution_reward=1.0,
        reward_scheme="chain_final",
        reward_gamma=0.9,
        num_transitions=3,
    )

    assert [reward for reward, _ in rewards] == pytest.approx([0.81, 0.9, 1.0])
    assert [power for _, power in rewards] == [2, 1, 0]


def test_chain_final_reward_keeps_failed_trajectories_at_zero() -> None:
    rewards = _compute_transition_rewards(
        final_execution_reward=0.0,
        reward_scheme="chain_final",
        reward_gamma=0.9,
        num_transitions=2,
    )

    assert rewards == [(0.0, 1), (0.0, 0)]


def test_transition_selection_round_robins_by_uid() -> None:
    _TRANSITION_COUNTERS.clear()

    selected = [
        _select_transition_index(uid="music:0", num_transitions=3, selection="round_robin")
        for _ in range(5)
    ]

    assert selected == [0, 1, 2, 0, 1]
    assert _select_transition_index(uid="music:0", num_transitions=3, selection="final") == 2


def test_reward_config_validation() -> None:
    assert _normalize_reward_scheme("chain_final") == "chain_final"
    assert _normalize_reward_scheme("final_shared") == "final_shared"
    assert _normalize_reward_gamma("0.9") == 0.9
    assert _normalize_transition_selection("round_robin") == "round_robin"

    with pytest.raises(ValueError):
        _normalize_reward_scheme("tree_final")
    with pytest.raises(ValueError):
        _normalize_reward_gamma("1.1")
    with pytest.raises(ValueError):
        _normalize_transition_selection("all")
