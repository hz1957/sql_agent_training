import sqlite3
from pathlib import Path

from sql_agent_training.reward.spider_reward import spider_execution_reward


def test_spider_execution_reward_matches_equivalent_sql(tmp_path: Path) -> None:
    db_path = tmp_path / "music.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE Singer (Name TEXT)")
        conn.execute("INSERT INTO Singer VALUES ('Ada')")
        conn.commit()
    finally:
        conn.close()

    reward = spider_execution_reward("SELECT Name FROM Singer", "SELECT Name FROM Singer", db_path)

    assert reward == 1.0
