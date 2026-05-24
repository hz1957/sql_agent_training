"""VERL AgentLoop implementation for Spider SQL rollouts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from sql_agent_training.agent.actions import extract_sql_candidate
from sql_agent_training.env.sqlite_tool import SQLiteTool
from sql_agent_training.reward.spider_reward import spider_execution_reward

try:  # pragma: no cover - exercised only when the train extra is installed
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
except Exception:  # pragma: no cover - keeps module import-safe without VERL
    AgentLoopBase = object  # type: ignore[assignment,misc]
    AgentLoopOutput = None  # type: ignore[assignment]

    def register(_: str):
        def decorator(cls):
            return cls

        return decorator


@dataclass(frozen=True)
class VerlSqlAgentSample:
    """Normalized fields that a VERL dataset row must provide to the SQL agent loop."""

    uid: str
    question: str
    db_id: str
    schema_prompt: str
    sqlite_path: str | None = None
    gold_sql: str | None = None


def build_sql_agent_sample_from_verl_kwargs(kwargs: dict[str, Any]) -> VerlSqlAgentSample:
    """Extract SQL-agent fields from VERL AgentLoop `run` kwargs."""

    extra_info = kwargs.get("extra_info") or {}
    if not isinstance(extra_info, dict):
        extra_info = {}

    raw_prompt = kwargs.get("raw_prompt") or []
    prompt_text = ""
    if raw_prompt:
        last_message = raw_prompt[-1]
        if isinstance(last_message, dict):
            prompt_text = str(last_message.get("content") or "")
        else:
            prompt_text = str(last_message)

    question = str(kwargs.get("question") or extra_info.get("question") or prompt_text)
    db_id = str(kwargs.get("db_id") or extra_info.get("db_id") or "")
    uid = str(kwargs.get("uid") or extra_info.get("uid") or extra_info.get("index") or db_id or uuid4().hex)
    schema_prompt = str(kwargs.get("schema_prompt") or extra_info.get("schema_prompt") or "")
    sqlite_path = kwargs.get("sqlite_path") or extra_info.get("sqlite_path")
    gold_sql = kwargs.get("gold_sql") or kwargs.get("query") or extra_info.get("gold_sql") or extra_info.get("query")
    return VerlSqlAgentSample(
        uid=uid,
        question=question,
        db_id=db_id,
        schema_prompt=schema_prompt,
        sqlite_path=str(sqlite_path) if sqlite_path else None,
        gold_sql=str(gold_sql) if gold_sql else None,
    )


@register("sql_agent")
class SqlAgentVerlLoop(AgentLoopBase):
    """SQL agent loop registered for VERL experimental AgentLoop rollout."""

    def __init__(self, *args, max_turns: int = 3, sqlite_timeout_steps: int = 100_000, **kwargs) -> None:
        if AgentLoopOutput is None:
            raise RuntimeError("Install the train extra to use SqlAgentVerlLoop.")
        super().__init__(*args, **kwargs)
        self.max_turns = max_turns
        self.sqlite_tool = SQLiteTool(timeout_steps=sqlite_timeout_steps)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run one SQL agent trajectory inside VERL."""

        start = time.monotonic()
        sample = build_sql_agent_sample_from_verl_kwargs(kwargs)
        messages = list(kwargs.get("raw_prompt") or [])
        if not messages:
            messages = [
                {
                    "role": "user",
                    "content": f"Question: {sample.question}\n\nSchema:\n{sample.schema_prompt}",
                }
            ]

        prompt_ids = await self.apply_chat_template(messages)
        response_ids: list[int] = []
        response_mask: list[int] = []
        turn_scores: list[float] = []
        tool_rewards: list[float] = []
        last_candidate_sql: str | None = None
        last_executed_sql: str | None = None
        final_sql: str | None = None
        final_sql_source = "none"
        num_parse_errors = 0
        num_execute_calls = 0

        for _ in range(self.max_turns):
            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                break

            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=(prompt_ids + response_ids)[-self.prompt_length :],
                sampling_params=sampling_params,
            )
            assistant_ids = list(output.token_ids[:remaining])
            assistant_text = self.tokenizer.decode(assistant_ids, skip_special_tokens=True).strip()
            response_ids.extend(assistant_ids)
            response_mask.extend([1] * len(assistant_ids))

            candidate_sql = extract_sql_candidate(assistant_text)
            if candidate_sql is None:
                num_parse_errors += 1
                feedback = "No SQL query found. Return only one read-only SQLite SELECT query."
                feedback_ids = self.tokenizer.encode(feedback, add_special_tokens=False)
                feedback_ids = feedback_ids[: max(0, self.response_length - len(response_ids))]
                response_ids.extend(feedback_ids)
                response_mask.extend([0] * len(feedback_ids))
                continue

            last_candidate_sql = candidate_sql
            num_execute_calls += 1
            if sample.sqlite_path:
                execution = self.sqlite_tool.execute(sample.sqlite_path, candidate_sql)
                if execution.ok:
                    last_executed_sql = candidate_sql
                candidate_reward = (
                    spider_execution_reward(candidate_sql, sample.gold_sql, Path(sample.sqlite_path))
                    if sample.gold_sql and execution.ok
                    else 0.0
                )
                feedback = f"Execution ok={execution.ok}; result={execution.rows if execution.ok else execution.error}; reward={candidate_reward}"
            else:
                candidate_reward = 0.0
                feedback = "Execution failed: sqlite_path_missing; reward=0.0"

            if candidate_reward >= 1.0:
                final_sql = candidate_sql
                final_sql_source = "passed_execution_reward"
                turn_scores.append(candidate_reward)
                break

            tool_ids = self.tokenizer.encode(feedback, add_special_tokens=False)
            tool_ids = tool_ids[: max(0, self.response_length - len(response_ids))]
            response_ids.extend(tool_ids)
            response_mask.extend([0] * len(tool_ids))
            tool_rewards.append(0.0)

        if final_sql is None:
            if last_candidate_sql is not None:
                final_sql = last_candidate_sql
                final_sql_source = "last_candidate_sql"
            elif last_executed_sql is not None:
                final_sql = last_executed_sql
                final_sql_source = "last_executed_sql"

        reward_score = 0.0
        if final_sql and sample.gold_sql and sample.sqlite_path:
            reward_score = spider_execution_reward(final_sql, sample.gold_sql, Path(sample.sqlite_path))
        if not turn_scores:
            turn_scores.append(reward_score)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids or [self.tokenizer.eos_token_id or 0],
            response_mask=response_mask or [0],
            reward_score=reward_score,
            num_turns=len(turn_scores) + num_execute_calls + num_parse_errors,
            metrics={
                "generate_sequences": time.monotonic() - start,
                "tool_calls": float(num_execute_calls),
                "num_preempted": -1,
            },
            extra_fields={
                "uid": sample.uid,
                "db_id": sample.db_id,
                "final_sql": final_sql,
                "final_sql_source": final_sql_source,
                "num_parse_errors": num_parse_errors,
                "turn_scores": turn_scores,
                "tool_rewards": tool_rewards,
            },
        )
