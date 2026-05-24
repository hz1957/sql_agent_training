from sql_agent_training.agent.tokenization import WhitespaceTokenizer, trajectory_to_tokenized
from sql_agent_training.agent.trace_format import AgentTrajectory, AgentTurn


def test_trajectory_to_tokenized_masks_assistant_and_tool_tokens() -> None:
    trajectory = AgentTrajectory(
        uid="music:0",
        rollout_id="music:0:0",
        turns=[
            AgentTurn(role="user", content="Question"),
            AgentTurn(role="assistant", content="SELECT 1"),
            AgentTurn(role="tool", content="[(1,)]"),
            AgentTurn(role="assistant", content="SELECT 1"),
        ],
        final_sql="SELECT 1",
        final_sql_source="passed_execution_reward",
        reward=1.0,
    )

    tokenized = trajectory_to_tokenized(trajectory, WhitespaceTokenizer())

    assert tokenized.uid == "music:0"
    assert tokenized.rollout_id == "music:0:0"
    assert len(tokenized.response_ids) == len(tokenized.response_mask)
    assert 0 in tokenized.response_mask
    assert 1 in tokenized.response_mask
    assert tokenized.reward == 1.0


def test_trajectory_to_tokenized_rejects_empty_turns() -> None:
    trajectory = AgentTrajectory(
        uid="x",
        rollout_id="x:0",
        turns=[],
        final_sql=None,
        final_sql_source="none",
    )

    try:
        trajectory_to_tokenized(trajectory, WhitespaceTokenizer())
    except ValueError as exc:
        assert "at least one turn" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
