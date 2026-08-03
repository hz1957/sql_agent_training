from scripts.generate_sft_trajectories import _select_verified_rows, _trajectory_to_sft_row


def _trajectory_row(uid: str, rollout_index: int, *, reward: float = 1.0, rewrite: bool = False) -> dict:
    turns = [
        {"role": "user", "content": f"write prompt {uid}", "metadata": {}},
        {
            "role": "assistant",
            "content": f"SELECT wrong_{rollout_index}",
            "metadata": {
                "trainable": True,
                "agent_step": "write_query",
                "turn_index": 0,
                "prompt_text": f"formatted write prompt {uid}",
            },
        },
    ]
    if rewrite:
        turns.extend(
            [
                {"role": "tool", "content": "execution feedback", "metadata": {"ok": True}},
                {"role": "user", "content": f"rewrite prompt {uid}", "metadata": {}},
                {
                    "role": "assistant",
                    "content": f"SELECT final_{rollout_index}",
                    "metadata": {
                        "trainable": True,
                        "agent_step": "rewrite_query",
                        "turn_index": 1,
                        "prompt_text": f"formatted rewrite prompt {uid}",
                    },
                },
            ]
        )
    return {
        "uid": uid,
        "db_id": "db",
        "rollout_id": f"{uid}:rollout={rollout_index}",
        "reward": reward,
        "final_sql": f"SELECT final_{rollout_index}" if reward else None,
        "is_rewrite_trajectory": rewrite,
        "turns": turns,
    }


def test_select_verified_rows_spreads_selection_across_questions() -> None:
    rows = [
        _trajectory_row("q1", 0),
        _trajectory_row("q1", 1),
        _trajectory_row("q2", 0),
        _trajectory_row("q2", 1),
        _trajectory_row("q3", 0, reward=0.0),
    ]

    selected, available = _select_verified_rows(rows, target_correct=3, seed=42)

    assert available == 4
    assert len(selected) == 3
    assert {row["uid"] for row in selected} == {"q1", "q2"}


def test_select_verified_rows_removes_exact_duplicate_trajectories() -> None:
    row = _trajectory_row("q1", 0)

    selected, available = _select_verified_rows([row, dict(row)], target_correct=2, seed=42)

    assert available == 1
    assert selected == [row]


def test_trajectory_to_sft_row_supervises_only_final_correct_action() -> None:
    row = _trajectory_row("q1", 2, rewrite=True)

    sft_row = _trajectory_to_sft_row(row)

    assert sft_row["prompt"] == "formatted rewrite prompt q1"
    assert sft_row["completion"] == "SELECT final_2"
    assert sft_row["agent_step"] == "rewrite_query"
    assert sft_row["is_rewrite_trajectory"] is True
