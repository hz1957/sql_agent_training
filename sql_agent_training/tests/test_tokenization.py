from sql_agent_training.agent.tokenization import (
    WhitespaceTokenizer,
    trajectory_to_tokenized_transitions,
)
from sql_agent_training.agent.trace_format import AgentTrajectory, AgentTurn


def test_trajectory_to_tokenized_transitions_splits_assistant_actions() -> None:
    trajectory = AgentTrajectory(
        uid="music:0",
        rollout_id="music:0:0",
        turns=[
            AgentTurn(role="user", content="Question"),
            AgentTurn(role="assistant", content="SELECT Missing FROM Singer", metadata={"turn_index": 0}),
            AgentTurn(
                role="tool",
                content="no such column: Missing",
                metadata={"ok": False, "sql": "SELECT Missing FROM Singer", "reward": None},
            ),
            AgentTurn(role="user", content="Check query"),
            AgentTurn(
                role="assistant",
                content="The query is wrong.\nTHE QUERY IS INCORRECT.",
                metadata={"agent_step": "check_query", "trainable": False, "turn_index": 0},
            ),
            AgentTurn(role="user", content="Question\nPrevious error: no such column: Missing"),
            AgentTurn(
                role="assistant",
                content="SELECT Name FROM Singer",
                metadata={"agent_step": "rewrite_query", "trainable": True, "turn_index": 1},
            ),
            AgentTurn(
                role="tool",
                content="rows=[('Ada',)]; row_count=1",
                metadata={"ok": True, "sql": "SELECT Name FROM Singer", "reward": 1.0},
            ),
        ],
        final_sql="SELECT Name FROM Singer",
        final_sql_source="executed_successfully",
        reward=1.0,
    )

    transitions = trajectory_to_tokenized_transitions(trajectory, WhitespaceTokenizer())

    assert [transition.rollout_id for transition in transitions] == ["music:0:0:turn0", "music:0:0:turn1"]
    assert [transition.reward for transition in transitions] == [1.0, 1.0]
    assert transitions[0].prompt_text == "user: Question"
    assert transitions[0].response_text == "assistant: SELECT Missing FROM Singer"
    assert "Previous error" in str(transitions[1].prompt_text)
    assert transitions[0].metadata["tool_ok"] is False
    assert transitions[1].metadata["tool_ok"] is True
    assert [transition.group_id for transition in transitions] == ["music:0", "music:0"]
    assert all(set(transition.response_mask) == {1} for transition in transitions)


def test_trajectory_to_tokenized_transitions_can_discount_final_reward() -> None:
    trajectory = AgentTrajectory(
        uid="music:0",
        rollout_id="music:0:0",
        turns=[
            AgentTurn(role="user", content="Question"),
            AgentTurn(role="assistant", content="SELECT Missing FROM Singer", metadata={"turn_index": 0}),
            AgentTurn(
                role="tool",
                content="no such column: Missing",
                metadata={"ok": False, "sql": "SELECT Missing FROM Singer", "reward": None},
            ),
            AgentTurn(role="user", content="Check query"),
            AgentTurn(
                role="assistant",
                content="The query is wrong.\nTHE QUERY IS INCORRECT.",
                metadata={"agent_step": "check_query", "trainable": False, "turn_index": 0},
            ),
            AgentTurn(role="user", content="Question\nPrevious error: no such column: Missing"),
            AgentTurn(
                role="assistant",
                content="SELECT Name FROM Singer",
                metadata={"agent_step": "rewrite_query", "trainable": True, "turn_index": 1},
            ),
            AgentTurn(
                role="tool",
                content="rows=[('Ada',)]; row_count=1",
                metadata={"ok": True, "sql": "SELECT Name FROM Singer", "reward": 1.0},
            ),
        ],
        final_sql="SELECT Name FROM Singer",
        final_sql_source="checker_approved",
        reward=1.0,
    )

    transitions = trajectory_to_tokenized_transitions(
        trajectory,
        WhitespaceTokenizer(),
        reward_mode="discounted_final",
        reward_gamma=0.4,
    )

    assert [transition.reward for transition in transitions] == [0.4, 1.0]
    assert transitions[0].metadata["trajectory_reward"] == 1.0
    assert transitions[0].metadata["own_sql_reward"] == 0.0
    assert transitions[0].metadata["transition_reward_mode"] == "discounted_final"
    assert transitions[0].metadata["transition_reward_gamma"] == 0.4


def test_discounted_final_reward_preserves_correct_intermediate_sql() -> None:
    trajectory = AgentTrajectory(
        uid="music:0",
        rollout_id="music:0:0",
        turns=[
            AgentTurn(role="user", content="Question"),
            AgentTurn(role="assistant", content="SELECT Name FROM Singer", metadata={"turn_index": 0}),
            AgentTurn(
                role="tool",
                content="rows=[('Ada',)]; row_count=1",
                metadata={"ok": True, "sql": "SELECT Name FROM Singer", "reward": 1.0},
            ),
            AgentTurn(role="user", content="Question\nChecker was wrong"),
            AgentTurn(
                role="assistant",
                content="SELECT COUNT(*) FROM Singer",
                metadata={"agent_step": "rewrite_query", "trainable": True, "turn_index": 1},
            ),
            AgentTurn(
                role="tool",
                content="rows=[(1,)]; row_count=1",
                metadata={"ok": True, "sql": "SELECT COUNT(*) FROM Singer", "reward": 0.0},
            ),
        ],
        final_sql="SELECT COUNT(*) FROM Singer",
        final_sql_source="ran_out_of_turns",
        reward=0.0,
    )

    transitions = trajectory_to_tokenized_transitions(
        trajectory,
        WhitespaceTokenizer(),
        reward_mode="discounted_final",
        reward_gamma=0.4,
    )

    assert [transition.reward for transition in transitions] == [1.0, 0.0]


def test_trajectory_to_tokenized_transitions_prefers_model_token_ids() -> None:
    trajectory = AgentTrajectory(
        uid="music:0",
        rollout_id="music:0:0",
        turns=[
            AgentTurn(role="user", content="Question"),
            AgentTurn(
                role="assistant",
                content="SELECT Name FROM Singer",
                metadata={
                    "turn_index": 0,
                    "prompt_ids": [101, 102],
                    "response_ids": [201, 202, 203],
                    "prompt_text": "<chat prompt>",
                    "response_text": "SELECT Name FROM Singer",
                },
            ),
            AgentTurn(
                role="tool",
                content="rows=[('Ada',)]; row_count=1",
                metadata={"ok": True, "sql": "SELECT Name FROM Singer", "reward": 1.0},
            ),
        ],
        final_sql="SELECT Name FROM Singer",
        final_sql_source="executed_successfully",
        reward=1.0,
    )

    transitions = trajectory_to_tokenized_transitions(trajectory, WhitespaceTokenizer())

    assert transitions[0].prompt_ids == [101, 102]
    assert transitions[0].response_ids == [201, 202, 203]
    assert transitions[0].prompt_text == "<chat prompt>"
    assert transitions[0].metadata["used_model_token_ids"] is True


def test_trajectory_to_tokenized_transitions_rejects_empty_turns() -> None:
    trajectory = AgentTrajectory(
        uid="x",
        rollout_id="x:0",
        turns=[],
        final_sql=None,
        final_sql_source="none",
    )

    try:
        trajectory_to_tokenized_transitions(trajectory, WhitespaceTokenizer())
    except ValueError as exc:
        assert "at least one turn" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
