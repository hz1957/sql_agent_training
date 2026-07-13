"""Local SQL agent rollout loop used by dry runs and unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sql_agent_training.agent.actions import extract_sql_candidate
from sql_agent_training.agent.model_client import ModelClient, ModelRequest, ModelResponse
from sql_agent_training.agent.prompts import (
    build_check_query_prompt,
    build_rewrite_query_prompt,
    build_write_query_prompt,
)
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


def _checker_verdict(feedback: str) -> bool | None:
    """Return the checker decision when it uses the expected terminal phrase."""

    normalized = feedback.upper()
    correct_index = normalized.rfind("THE QUERY IS CORRECT.")
    incorrect_index = normalized.rfind("THE QUERY IS INCORRECT.")
    if correct_index < 0 and incorrect_index < 0:
        return None
    return correct_index > incorrect_index


def _default_checker_feedback(execution_ok: bool) -> str:
    """Fallback checker response for deterministic scripted SQL-only rollouts."""

    if execution_ok:
        return "No execution failure was observed.\nTHE QUERY IS CORRECT."
    return "The query failed to execute.\nTHE QUERY IS INCORRECT."


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
        *,
        checker_responses=None,
    ) -> AgentTrajectory:
        """Run deterministic SQL rewrite semantics from pre-generated model responses.

        `model_responses` feeds only trainable writer/rewriter calls. Checker
        calls use `checker_responses` when provided, otherwise a deterministic
        execution-based checker keeps existing dry runs compact.
        """

        response_iter = iter(model_responses)
        checker_iter = iter(checker_responses) if checker_responses is not None else None

        def next_response(_: list[AgentTurn], agent_step: str) -> str | None:
            if agent_step == "check_query":
                if checker_iter is None:
                    return None
                try:
                    return next(checker_iter)
                except StopIteration:
                    return None
            try:
                return next(response_iter)
            except StopIteration:
                return None

        return self._run_core(sample, next_response, sqlite_path, default_check_when_missing=True)

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

        def next_response(turns: list[AgentTurn], agent_step: str) -> ModelResponse | None:
            del agent_step
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

    def _build_sql_request_turn(
        self,
        sample: SqlAgentInput,
        *,
        agent_step: str,
        previous_sql: str | None,
        previous_execution: str | None,
        feedback: str | None,
    ) -> AgentTurn:
        if agent_step == "write_query":
            content = build_write_query_prompt(sample.question, sample.schema_prompt)
        else:
            content = build_rewrite_query_prompt(
                sample.question,
                sample.schema_prompt,
                previous_sql=previous_sql or "",
                previous_execution=previous_execution or "",
                feedback=feedback or "",
            )
        return AgentTurn(
            role="user",
            content=content,
            metadata={"db_id": sample.db_id, "agent_step": agent_step},
        )

    def _build_check_turn(self, sample: SqlAgentInput, *, query: str, execution: str) -> AgentTurn:
        return AgentTurn(
            role="user",
            content=build_check_query_prompt(sample.question, sample.schema_prompt, query, execution),
            metadata={"db_id": sample.db_id, "agent_step": "check_query"},
        )

    def _run_core(
        self,
        sample: SqlAgentInput,
        next_response: Callable[[list[AgentTurn], str], str | ModelResponse | None],
        sqlite_path: str | Path,
        *,
        default_check_when_missing: bool = False,
    ) -> AgentTrajectory:
        turns: list[AgentTurn] = []
        last_candidate_sql: str | None = None
        previous_sql: str | None = None
        previous_execution: str | None = None
        previous_feedback: str | None = None
        final_sql: str | None = None
        final_sql_source = "none"
        num_execute_calls = 0
        num_parse_errors = 0
        num_check_calls = 0
        ran_out_of_turns = False
        reward: float | None = None

        for turn_index in range(self.max_turns):
            agent_step = "write_query" if turn_index == 0 else "rewrite_query"
            request_turn = self._build_sql_request_turn(
                sample,
                agent_step=agent_step,
                previous_sql=previous_sql,
                previous_execution=previous_execution,
                feedback=previous_feedback,
            )
            raw_response = next_response([request_turn], agent_step)
            if raw_response is None:
                break
            if isinstance(raw_response, ModelResponse):
                response = raw_response.content
                assistant_metadata = {
                    "agent_step": agent_step,
                    "trainable": True,
                    "turn_index": turn_index,
                    "prompt_ids": raw_response.prompt_ids,
                    "response_ids": raw_response.response_ids,
                    "prompt_text": raw_response.prompt_text,
                    "response_text": raw_response.response_text,
                }
            else:
                response = raw_response
                assistant_metadata = {"agent_step": agent_step, "trainable": True, "turn_index": turn_index}

            turns.append(request_turn)
            turns.append(AgentTurn(role="assistant", content=response, metadata=assistant_metadata))
            candidate_sql = extract_sql_candidate(response)
            if candidate_sql is None:
                num_parse_errors += 1
                previous_sql = None
                previous_execution = "No SQL query found. Return only one read-only SQLite SELECT query."
                previous_feedback = previous_execution
                turns.append(
                    AgentTurn(
                        role="tool",
                        content=previous_execution,
                        metadata={"ok": False, "error": "no_sql", "agent_step": "execute_query"},
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
                        "agent_step": "execute_query",
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

            check_turn = self._build_check_turn(sample, query=candidate_sql, execution=feedback)
            raw_check = next_response([check_turn], "check_query")
            if raw_check is None and default_check_when_missing:
                raw_check = _default_checker_feedback(execution.ok)
            if raw_check is None:
                if execution.ok:
                    break
                previous_sql = candidate_sql
                previous_execution = feedback
                previous_feedback = feedback
                continue

            if isinstance(raw_check, ModelResponse):
                check_response = raw_check.content
                check_metadata = {
                    "agent_step": "check_query",
                    "trainable": False,
                    "turn_index": turn_index,
                    "prompt_ids": raw_check.prompt_ids,
                    "response_ids": raw_check.response_ids,
                    "prompt_text": raw_check.prompt_text,
                    "response_text": raw_check.response_text,
                    "query": candidate_sql,
                    "execution_ok": execution.ok,
                }
            else:
                check_response = raw_check
                check_metadata = {
                    "agent_step": "check_query",
                    "trainable": False,
                    "turn_index": turn_index,
                    "query": candidate_sql,
                    "execution_ok": execution.ok,
                }
            num_check_calls += 1
            turns.append(check_turn)
            turns.append(AgentTurn(role="assistant", content=check_response, metadata=check_metadata))

            verdict = _checker_verdict(check_response)
            if verdict is True and execution.ok:
                final_sql = candidate_sql
                final_sql_source = "checker_approved"
                reward = candidate_reward
                break

            previous_sql = candidate_sql
            previous_execution = feedback
            previous_feedback = check_response

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
                "num_check_calls": num_check_calls,
                "num_parse_errors": num_parse_errors,
                "no_parseable_sql": last_candidate_sql is None,
                "max_turns": self.max_turns,
            },
        )
