"""Local SQL agent rollout loop used by dry runs and unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sql_agent_training.agent.actions import extract_sql_candidate
from sql_agent_training.agent.model_client import ModelClient, ModelRequest, ModelResponse
from sql_agent_training.agent.prompts import build_agent_prompt
from sql_agent_training.agent.trace_format import AgentTrajectory, AgentTurn
from sql_agent_training.env.sqlite_tool import SQLiteTool
from sql_agent_training.reward.spider_reward import spider_execution_reward


def _format_execution_feedback(ok: bool, rows: list[tuple[object, ...]], error: str | None) -> str:
    """Render bounded tool feedback for the next model turn."""

    if not ok:
        return str(error)
    preview = rows[:5]
    suffix = "" if len(rows) <= len(preview) else f"; truncated={len(rows) - len(preview)}"
    return f"rows={preview}; row_count={len(rows)}{suffix}"


@dataclass(frozen=True)
class SqlAgentInput:
    """Input fields required for one SQL agent rollout."""

    uid: str
    rollout_id: str
    question: str
    db_id: str
    schema_prompt: str
    gold_sql: str | None = None


class SqlAgentLoop:
    """Deterministic SQL execution-rewrite loop for local tests and rollout preparation."""

    def __init__(self, max_turns: int = 3, sqlite_tool: SQLiteTool | None = None) -> None:
        self.max_turns = max_turns
        self.sqlite_tool = sqlite_tool or SQLiteTool()

    def run_with_responses(
        self,
        sample: SqlAgentInput,
        model_responses,
        sqlite_path: str | Path,
    ) -> AgentTrajectory:
        """Run deterministic SQL rewrite semantics from pre-generated model responses."""

        response_iter = iter(model_responses)

        def next_response(_: list[AgentTurn]) -> str | None:
            try:
                return next(response_iter)
            except StopIteration:
                return None

        return self._run_core(sample, next_response, sqlite_path)

    def run(
        self,
        sample: SqlAgentInput,
        model_client: ModelClient,
        sqlite_path: str | Path,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> AgentTrajectory:
        """Run an interactive SQL agent rollout with a model client."""

        def next_response(turns: list[AgentTurn]) -> ModelResponse | None:
            return model_client.generate(
                ModelRequest(
                    turns=tuple(turns),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            )

        return self._run_core(sample, next_response, sqlite_path)

    def _build_request_turn(
        self,
        sample: SqlAgentInput,
        *,
        previous_failed_sql: str | None,
        previous_error: str | None,
    ) -> AgentTurn:
        return AgentTurn(
            role="user",
            content=build_agent_prompt(
                sample.question,
                sample.schema_prompt,
                previous_failed_sql=previous_failed_sql,
                previous_error=previous_error,
            ),
            metadata={"db_id": sample.db_id},
        )

    def _run_core(
        self,
        sample: SqlAgentInput,
        next_response: Callable[[list[AgentTurn]], str | ModelResponse | None],
        sqlite_path: str | Path,
    ) -> AgentTrajectory:
        turns: list[AgentTurn] = []
        last_candidate_sql: str | None = None
        previous_failed_sql: str | None = None
        previous_error: str | None = None
        final_sql: str | None = None
        final_sql_source = "none"
        num_execute_calls = 0
        num_parse_errors = 0
        ran_out_of_turns = False
        reward: float | None = None

        for turn_index in range(self.max_turns):
            request_turn = self._build_request_turn(
                sample,
                previous_failed_sql=previous_failed_sql,
                previous_error=previous_error,
            )
            raw_response = next_response([request_turn])
            if raw_response is None:
                break
            if isinstance(raw_response, ModelResponse):
                response = raw_response.content
                assistant_metadata = {
                    "turn_index": turn_index,
                    "prompt_ids": raw_response.prompt_ids,
                    "response_ids": raw_response.response_ids,
                    "prompt_text": raw_response.prompt_text,
                    "response_text": raw_response.response_text,
                }
            else:
                response = raw_response
                assistant_metadata = {"turn_index": turn_index}

            turns.append(request_turn)
            turns.append(AgentTurn(role="assistant", content=response, metadata=assistant_metadata))
            candidate_sql = extract_sql_candidate(response)
            if candidate_sql is None:
                num_parse_errors += 1
                previous_failed_sql = None
                previous_error = "No SQL query found. Return only one read-only SQLite SELECT query."
                turns.append(
                    AgentTurn(
                        role="tool",
                        content=previous_error,
                        metadata={"ok": False, "error": "no_sql"},
                    )
                )
                continue

            last_candidate_sql = candidate_sql
            num_execute_calls += 1
            execution = self.sqlite_tool.execute(sqlite_path, candidate_sql)
            feedback = _format_execution_feedback(execution.ok, execution.rows, execution.error)
            candidate_reward = (
                spider_execution_reward(candidate_sql, sample.gold_sql, sqlite_path)
                if sample.gold_sql and execution.ok
                else None
            )
            turns.append(
                AgentTurn(
                    role="tool",
                    content=feedback,
                    metadata={
                        "ok": execution.ok,
                        "sql": candidate_sql,
                        "error": execution.error,
                        "elapsed_seconds": execution.elapsed_seconds,
                        "safety_reason": execution.safety_reason,
                        "reward": candidate_reward,
                    },
                )
            )
            if execution.ok:
                final_sql = candidate_sql
                final_sql_source = "executed_successfully"
                reward = candidate_reward
                break
            previous_failed_sql = candidate_sql
            previous_error = feedback

        else:
            ran_out_of_turns = True

        if final_sql and sample.gold_sql and reward is None:
            reward = spider_execution_reward(final_sql, sample.gold_sql, sqlite_path)
        elif not final_sql:
            reward = 0.0

        return AgentTrajectory(
            uid=sample.uid,
            rollout_id=sample.rollout_id,
            turns=turns,
            final_sql=final_sql,
            final_sql_source=final_sql_source,
            reward=reward,
            metadata={
                "ran_out_of_turns": ran_out_of_turns,
                "num_execute_calls": num_execute_calls,
                "num_parse_errors": num_parse_errors,
                "no_parseable_sql": last_candidate_sql is None,
                "max_turns": self.max_turns,
            },
        )
