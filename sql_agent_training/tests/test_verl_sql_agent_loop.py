import sqlite3
import asyncio
from pathlib import Path

import pytest

from sql_agent_training.train.verl_sql_agent_loop import AgentLoopOutput, SqlAgentVerlLoop, build_sql_agent_sample_from_verl_kwargs


def test_build_sql_agent_sample_from_verl_extra_info() -> None:
    sample = build_sql_agent_sample_from_verl_kwargs(
        {
            "raw_prompt": [{"role": "user", "content": "fallback question"}],
            "extra_info": {
                "uid": "concert_singer:0",
                "question": "How many singers do we have?",
                "db_id": "concert_singer",
                "schema_prompt": "Database: concert_singer",
                "sqlite_path": "data/spider/database/concert_singer/concert_singer.sqlite",
                "gold_sql": "SELECT count(*) FROM singer",
            },
        }
    )

    assert sample.uid == "concert_singer:0"
    assert sample.question == "How many singers do we have?"
    assert sample.db_id == "concert_singer"
    assert sample.schema_prompt == "Database: concert_singer"
    assert sample.sqlite_path == "data/spider/database/concert_singer/concert_singer.sqlite"
    assert sample.gold_sql == "SELECT count(*) FROM singer"


def test_build_sql_agent_sample_uses_raw_prompt_fallback() -> None:
    sample = build_sql_agent_sample_from_verl_kwargs(
        {
            "raw_prompt": [{"role": "user", "content": "Question text"}],
            "extra_info": {"db_id": "music"},
        }
    )

    assert sample.uid == "music"
    assert sample.question == "Question text"
    assert sample.db_id == "music"


class DummyRolloutConfig:
    prompt_length = 256
    response_length = 128


class DummyTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self._texts: dict[int, str] = {}
        self._next_id = 1

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        token_id = self._next_id
        self._next_id += 1
        self._texts[token_id] = text
        return [token_id]

    def decode(self, ids, skip_special_tokens: bool = True):
        del skip_special_tokens
        return " ".join(self._texts.get(int(token_id), "") for token_id in ids)


class DummyOutput:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class DummyServerManager:
    def __init__(self, tokenizer: DummyTokenizer, responses: list[str]) -> None:
        self.tokenizer = tokenizer
        self.responses = iter(responses)

    async def generate(self, **kwargs):
        del kwargs
        return DummyOutput(self.tokenizer.encode(next(self.responses)))


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE Singer (Name TEXT)")
        conn.execute("INSERT INTO Singer VALUES ('Ada')")
        conn.execute("INSERT INTO Singer VALUES ('Grace')")
        conn.commit()
    finally:
        conn.close()


def test_verl_sql_agent_loop_rewrites_until_pass(tmp_path: Path) -> None:
    if AgentLoopOutput is None:
        pytest.skip("VERL is not installed")
    db_path = tmp_path / "music.sqlite"
    _make_db(db_path)
    tokenizer = DummyTokenizer()
    loop = SqlAgentVerlLoop.__new__(SqlAgentVerlLoop)
    loop.max_turns = 3
    loop.sqlite_tool = __import__("sql_agent_training.env.sqlite_tool", fromlist=["SQLiteTool"]).SQLiteTool()
    loop.prompt_length = 256
    loop.response_length = 128
    loop.rollout_config = DummyRolloutConfig()
    loop.tokenizer = tokenizer
    loop.server_manager = DummyServerManager(tokenizer, ["SELECT COUNT(*) FROM Singer", "SELECT Name FROM Singer"])

    async def apply_chat_template(messages):
        return tokenizer.encode(messages[0]["content"])

    loop.apply_chat_template = apply_chat_template

    output = asyncio.run(
        loop.run(
            {},
            raw_prompt=[{"role": "user", "content": "List singer names."}],
            extra_info={
                "uid": "music:0",
                "question": "List singer names.",
                "db_id": "music",
                "schema_prompt": "Database: music",
                "gold_sql": "SELECT Name FROM Singer",
                "sqlite_path": str(db_path),
            },
        )
    )

    assert output.reward_score == 1.0
    assert output.extra_fields["final_sql"] == "SELECT Name FROM Singer"
    assert output.extra_fields["final_sql_source"] == "passed_execution_reward"
