import sqlite3
from pathlib import Path

from sql_agent_training.agent.model_client import ModelRequest, ModelResponse, ScriptedModelClient
from sql_agent_training.agent.sql_agent_loop import SqlAgentInput, SqlAgentLoop


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE Singer (Name TEXT)")
        conn.execute("INSERT INTO Singer VALUES ('Ada')")
        conn.execute("INSERT INTO Singer VALUES ('Grace')")
        conn.commit()
    finally:
        conn.close()


def _sample() -> SqlAgentInput:
    return SqlAgentInput(
        uid="music:0",
        rollout_id="rollout-1",
        question="List singer names.",
        db_id="music",
        schema_prompt="Database: music\n- Singer(Name)",
        gold_sql="SELECT Name FROM Singer",
    )


def test_rollout_plain_sql_scores_reward_and_finalizes(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)

    trajectory = SqlAgentLoop(max_turns=3).run_with_responses(
        _sample(),
        ["SELECT Name FROM Singer"],
        db_path,
    )

    assert trajectory.final_sql == "SELECT Name FROM Singer"
    assert trajectory.final_sql_source == "passed_execution_reward"
    assert trajectory.reward == 1.0
    assert trajectory.metadata["ran_out_of_turns"] is False


def test_rollout_rewrites_until_sql_passes(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)

    trajectory = SqlAgentLoop(max_turns=3).run_with_responses(
        _sample(),
        [
            "SELECT COUNT(*) FROM Singer",
            "SELECT Name FROM Singer",
        ],
        db_path,
    )

    assert trajectory.final_sql == "SELECT Name FROM Singer"
    assert trajectory.final_sql_source == "passed_execution_reward"
    assert trajectory.metadata["num_execute_calls"] == 2
    assert trajectory.reward == 1.0


def test_rollout_max_turns_uses_last_candidate_sql(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)

    trajectory = SqlAgentLoop(max_turns=1).run_with_responses(
        _sample(),
        [
            "SELECT COUNT(*) FROM Singer",
            "SELECT Name FROM Singer",
        ],
        db_path,
    )

    assert trajectory.final_sql == "SELECT COUNT(*) FROM Singer"
    assert trajectory.final_sql_source == "last_candidate_sql"
    assert trajectory.metadata["ran_out_of_turns"] is True
    assert trajectory.reward == 0.0


def test_rollout_no_parseable_sql_gets_zero_reward(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)

    trajectory = SqlAgentLoop(max_turns=2).run_with_responses(_sample(), ["I cannot decide."], db_path)

    assert trajectory.final_sql is None
    assert trajectory.reward == 0.0
    assert trajectory.metadata["no_parseable_sql"] is True
    assert trajectory.metadata["num_parse_errors"] == 1


def test_rollout_unsafe_sql_scores_zero(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)

    trajectory = SqlAgentLoop(max_turns=1).run_with_responses(
        _sample(),
        ["DROP TABLE Singer"],
        db_path,
    )

    assert trajectory.final_sql is None
    assert trajectory.reward == 0.0


def test_interactive_rollout_rewrites_with_scripted_model_client(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)
    client = ScriptedModelClient(
        [
            "SELECT COUNT(*) FROM Singer",
            "SELECT Name FROM Singer",
        ]
    )

    trajectory = SqlAgentLoop(max_turns=3).run(_sample(), client, db_path)

    assert client.calls == 2
    assert trajectory.final_sql_source == "passed_execution_reward"
    assert trajectory.reward == 1.0
    assert [turn.role for turn in trajectory.turns] == ["user", "assistant", "tool", "assistant", "tool"]


class InspectingClient:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(content="SELECT COUNT(*) FROM Singer")
        return ModelResponse(content="SELECT Name FROM Singer")


def test_model_client_receives_tool_observation(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)
    client = InspectingClient()

    SqlAgentLoop(max_turns=3).run(_sample(), client, db_path)

    assert len(client.requests) == 2
    assert [turn.role for turn in client.requests[0].turns] == ["user"]
    assert [turn.role for turn in client.requests[1].turns] == ["user", "assistant", "tool"]
