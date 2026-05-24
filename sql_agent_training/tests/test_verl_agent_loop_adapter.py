from dataclasses import dataclass

import pytest

from sql_agent_training.train.grpo_batch import TokenizedTrajectory
from sql_agent_training.train.verl_agent_loop_adapter import resolve_verl_agent_loop_api, to_verl_agent_loop_output


@dataclass
class DummyAgentLoopOutput:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    reward_score: float | None = None
    num_turns: int = 0
    metrics: dict | None = None
    extra_fields: dict | None = None


def test_to_verl_agent_loop_output_uses_internal_contract() -> None:
    tokenized = TokenizedTrajectory(
        uid="music:0",
        rollout_id="music:0:0",
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 0],
        reward=1.0,
    )

    output = to_verl_agent_loop_output(tokenized, DummyAgentLoopOutput)

    assert output.prompt_ids == [1, 2]
    assert output.response_ids == [3, 4]
    assert output.response_mask == [1, 0]
    assert output.reward_score == 1.0
    assert output.num_turns == 0
    assert output.metrics == {}
    assert output.extra_fields == {"uid": "music:0", "rollout_id": "music:0:0", "metadata": {}}


def test_to_real_verl_agent_loop_output_when_verl_installed() -> None:
    try:
        api = resolve_verl_agent_loop_api()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    tokenized = TokenizedTrajectory(
        uid="music:0",
        rollout_id="music:0:1",
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 0],
        reward=0.5,
        metadata={"db_id": "music"},
    )

    output = to_verl_agent_loop_output(tokenized, api.agent_loop_output)

    assert output.prompt_ids == [1, 2]
    assert output.response_ids == [3, 4]
    assert output.response_mask == [1, 0]
    assert output.reward_score == 0.5
    assert output.metrics.generate_sequences == 0.0
    assert output.extra_fields["uid"] == "music:0"
    assert output.extra_fields["metadata"] == {"db_id": "music"}
