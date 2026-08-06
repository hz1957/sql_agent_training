import sqlite3
from pathlib import Path

from sql_agent_training.agent.model_client import ScriptedModelClient
from sql_agent_training.agent.sql_agent_loop import SqlAgentInput
from sql_agent_training.agent.tree_sql_agent_loop import TreeSqlAgentEvalLoop, tree_eval_slot_count


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE Singer (Name TEXT)")
        conn.execute("INSERT INTO Singer VALUES ('Ada')")
        conn.execute("INSERT INTO Singer VALUES ('Grace')")
        conn.commit()
    finally:
        conn.close()


def _sample(*, gold_sql: str = "SELECT Name FROM Singer") -> SqlAgentInput:
    return SqlAgentInput(
        uid="music:0",
        rollout_id="music:0:tree-eval",
        question="List singer names.",
        db_id="music",
        schema_prompt="Database: music\n- Singer(Name)",
        gold_sql=gold_sql,
    )


def test_tree_eval_slot_count_matches_bounded_frontier() -> None:
    assert tree_eval_slot_count(branch_n=4, beam_size=2, max_turns=3) == 20
    assert tree_eval_slot_count(branch_n=8, beam_size=1, max_turns=3) == 24


def test_checker_approved_node_is_leaf_but_sibling_frontier_expands(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)
    client = ScriptedModelClient(
        [
            "SELECT Name FROM Singer",
            "The query answers the question.\nTHE QUERY IS CORRECT.",
            "SELECT Missing FROM Singer",
            "The query references a missing column.\nTHE QUERY IS INCORRECT.",
            "SELECT Name FROM Singer",
            "The query answers the question.\nTHE QUERY IS CORRECT.",
            "SELECT COUNT(*) FROM Singer",
            "The query returns a count.\nTHE QUERY IS INCORRECT.",
        ]
    )

    trajectory = TreeSqlAgentEvalLoop(max_turns=2, branch_n=2, beam_size=1, seed=0).run(
        _sample(),
        client,
        db_path,
        temperature=1.0,
    )

    assert client.calls == 8
    assert trajectory.final_sql == "SELECT Name FROM Singer"
    assert trajectory.final_sql_source == "tree_checker_approved"
    assert trajectory.reward == 1.0
    assert trajectory.metadata["tree_nodes"] == 4
    assert trajectory.metadata["tree_terminal_candidates"] == 2

    rewrite_turns = [
        turn for turn in trajectory.turns if turn.role == "user" and turn.metadata.get("agent_step") == "rewrite_query"
    ]
    assert rewrite_turns
    assert all("## Previous query\nSELECT Missing FROM Singer" in turn.content for turn in rewrite_turns)


def test_tree_eval_can_use_separate_checker_model_client(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)
    writer_client = ScriptedModelClient(["SELECT COUNT(*) FROM Singer", "SELECT Name FROM Singer"])
    checker_client = ScriptedModelClient(
        [
            "The query returns a count, not names.\nTHE QUERY IS INCORRECT.",
            "The query returns names.\nTHE QUERY IS CORRECT.",
        ]
    )

    trajectory = TreeSqlAgentEvalLoop(max_turns=1, branch_n=2, beam_size=1, seed=0).run(
        _sample(),
        writer_client,
        db_path,
        checker_model_client=checker_client,
        temperature=1.0,
    )

    assert writer_client.calls == 2
    assert checker_client.calls == 2
    assert trajectory.final_sql == "SELECT Name FROM Singer"
    assert trajectory.final_sql_source == "tree_checker_approved"
    assert trajectory.reward == 1.0


def test_tree_eval_terminal_decision_does_not_use_gold_sql(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)
    client = ScriptedModelClient(
        [
            "SELECT Name FROM Singer",
            "The query looks valid.\nTHE QUERY IS CORRECT.",
        ]
    )

    trajectory = TreeSqlAgentEvalLoop(max_turns=1, branch_n=1, beam_size=1, seed=0).run(
        _sample(gold_sql="SELECT COUNT(*) FROM Singer"),
        client,
        db_path,
        temperature=1.0,
    )

    assert trajectory.final_sql == "SELECT Name FROM Singer"
    assert trajectory.final_sql_source == "tree_checker_approved"
    assert trajectory.reward == 0.0


def test_tree_eval_falls_back_to_best_executable_when_checker_rejects(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)
    client = ScriptedModelClient(
        [
            "SELECT COUNT(*) FROM Singer",
            "The query returns a count.\nTHE QUERY IS INCORRECT.",
            "SELECT Missing FROM Singer",
            "The query references a missing column.\nTHE QUERY IS INCORRECT.",
        ]
    )

    trajectory = TreeSqlAgentEvalLoop(max_turns=1, branch_n=2, beam_size=1, seed=0).run(
        _sample(gold_sql="SELECT COUNT(*) FROM Singer"),
        client,
        db_path,
        temperature=1.0,
    )

    assert trajectory.final_sql == "SELECT COUNT(*) FROM Singer"
    assert trajectory.final_sql_source == "tree_executable_fallback"
    assert trajectory.reward == 1.0


def test_tree_eval_rule_prunes_unrecoverable_missing_table(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)
    client = ScriptedModelClient(["SELECT Name FROM Phantom"])

    trajectory = TreeSqlAgentEvalLoop(max_turns=3, branch_n=1, beam_size=1, seed=0).run(
        _sample(),
        client,
        db_path,
        temperature=1.0,
    )

    assert trajectory.final_sql is None
    assert trajectory.reward == 0.0
    assert trajectory.metadata["tree_nodes"] == 1
    assert trajectory.metadata["tree_rule_pruned_candidates"] == 1
    assert trajectory.metadata["num_check_calls"] == 0
