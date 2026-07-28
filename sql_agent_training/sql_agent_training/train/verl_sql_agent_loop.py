"""verl AgentLoop implementation for Spider SQL-agent GRPO.

The loop keeps token-in/token-out integrity for model-generated tokens: decoded
text is used only for SQL parsing and reward computation, while the original
generated token ids are returned to verl for training.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from sql_agent_training.agent.actions import extract_sql_candidate
from sql_agent_training.agent.prompts import build_check_query_prompt, build_rewrite_query_prompt
from sql_agent_training.agent.sql_agent_loop import _checker_verdict, _format_execution_feedback
from sql_agent_training.env.sqlite_tool import SQLiteTool
from sql_agent_training.reward.spider_reward import spider_execution_reward

logger = logging.getLogger(__name__)

REWARD_SCHEME_FINAL_SHARED = "final_shared"
REWARD_SCHEME_CHAIN_FINAL = "chain_final"
VALID_REWARD_SCHEMES = {REWARD_SCHEME_FINAL_SHARED, REWARD_SCHEME_CHAIN_FINAL}
TRANSITION_SELECTION_ROUND_ROBIN = "round_robin"
TRANSITION_SELECTION_FINAL = "final"
VALID_TRANSITION_SELECTIONS = {TRANSITION_SELECTION_ROUND_ROBIN, TRANSITION_SELECTION_FINAL}
_TRANSITION_COUNTERS: dict[str, int] = {}

try:  # pragma: no cover - exercised on the verl runtime.
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
    from verl.utils.profiler import simple_timer
    from verl.utils.rollout_trace import rollout_trace_op
except ImportError:  # pragma: no cover - local tests run without verl installed.

    class AgentLoopBase:  # type: ignore[no-redef]
        pass

    AgentLoopOutput = None  # type: ignore[assignment]

    def register(_: str):  # type: ignore[no-redef]
        def decorator(cls):
            return cls

        return decorator

    def rollout_trace_op(func):  # type: ignore[no-redef]
        return func

    @contextmanager
    def simple_timer(name: str, metrics: dict[str, Any]):  # type: ignore[no-redef]
        start = time.perf_counter()
        try:
            yield
        finally:
            metrics[name] = time.perf_counter() - start


def _to_python(value: Any) -> Any:
    """Convert numpy/pyarrow scalar wrappers to regular Python values."""

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    value = _to_python(value)
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): _to_python(item) for key, item in value.items()}
    raise TypeError(f"Expected mapping-like extra_info, got {type(value)!r}")


def _cfg_get(config: Any, path: tuple[str, ...], default: Any) -> Any:
    current = config
    for key in path:
        if current is None:
            return default
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue
        if not hasattr(current, key):
            return default
        current = getattr(current, key)
    return current


def _cfg_or_env(config: Any, path: tuple[str, ...], env_name: str, default: Any) -> Any:
    env_value = os.environ.get(env_name)
    if env_value is not None and env_value != "":
        return env_value
    return _cfg_get(config, path, default)


def _normalize_reward_scheme(value: Any) -> str:
    scheme = str(value or REWARD_SCHEME_FINAL_SHARED).strip().lower()
    if scheme not in VALID_REWARD_SCHEMES:
        raise ValueError(f"Unknown SQL-agent reward scheme: {scheme}. Expected one of {sorted(VALID_REWARD_SCHEMES)}")
    return scheme


def _normalize_reward_gamma(value: Any) -> float:
    gamma = float(value)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"SQL-agent reward gamma must be between 0 and 1, got {gamma}.")
    return gamma


def _normalize_transition_selection(value: Any) -> str:
    selection = str(value or TRANSITION_SELECTION_ROUND_ROBIN).strip().lower()
    if selection not in VALID_TRANSITION_SELECTIONS:
        raise ValueError(
            f"Unknown SQL-agent transition selection: {selection}. "
            f"Expected one of {sorted(VALID_TRANSITION_SELECTIONS)}"
        )
    return selection


def _compute_transition_rewards(
    *,
    final_execution_reward: float,
    reward_scheme: str,
    reward_gamma: float,
    num_transitions: int,
) -> list[tuple[float, int]]:
    """Assign final execution reward to SQL-action transitions.

    `final_shared` preserves the old complete-trajectory behavior. `chain_final`
    matches the local GRPO transition splitter: the final SQL action gets the
    final reward, and earlier SQL actions receive gamma-discounted credit by
    distance from that final action.
    """

    if num_transitions <= 0:
        return []
    final_reward = float(final_execution_reward)
    if reward_scheme == REWARD_SCHEME_FINAL_SHARED:
        return [(final_reward, 0)]
    if reward_scheme == REWARD_SCHEME_CHAIN_FINAL:
        last_index = num_transitions - 1
        return [
            (final_reward * (reward_gamma ** (last_index - index)), last_index - index)
            for index in range(num_transitions)
        ]
    raise ValueError(f"Unknown SQL-agent reward scheme: {reward_scheme}")


def _select_transition_index(
    *,
    uid: str,
    num_transitions: int,
    selection: str,
) -> int:
    if num_transitions <= 0:
        raise ValueError("num_transitions must be positive")
    if selection == TRANSITION_SELECTION_FINAL:
        return num_transitions - 1
    if selection == TRANSITION_SELECTION_ROUND_ROBIN:
        cursor = _TRANSITION_COUNTERS.get(uid, 0)
        _TRANSITION_COUNTERS[uid] = cursor + 1
        return cursor % num_transitions
    raise ValueError(f"Unknown SQL-agent transition selection: {selection}")


def _raw_prompt_content(raw_prompt: Any) -> str | None:
    raw_prompt = _to_python(raw_prompt)
    if isinstance(raw_prompt, tuple):
        raw_prompt = list(raw_prompt)
    if not isinstance(raw_prompt, list) or not raw_prompt:
        return None
    first = _to_python(raw_prompt[0])
    if isinstance(first, dict) and first.get("content") is not None:
        return str(first["content"])
    return None


def _sample_fields(kwargs: dict[str, Any]) -> dict[str, str]:
    extra_info = _as_dict(kwargs.get("extra_info"))
    raw_prompt = _raw_prompt_content(kwargs.get("raw_prompt"))
    uid = str(extra_info.get("uid") or kwargs.get("index") or uuid4().hex)
    fields = {
        "uid": uid,
        "question": str(extra_info.get("question") or ""),
        "db_id": str(extra_info.get("db_id") or ""),
        "schema_prompt": str(extra_info.get("schema_prompt") or ""),
        "gold_sql": str(extra_info.get("gold_sql") or ""),
        "sqlite_path": str(extra_info.get("sqlite_path") or ""),
        "initial_prompt": str(raw_prompt or ""),
    }
    if not fields["initial_prompt"] and fields["question"] and fields["schema_prompt"]:
        # The parquet writer always provides raw_prompt through verl, but this
        # fallback keeps unit tests and direct calls ergonomic.
        from sql_agent_training.agent.prompts import build_write_query_prompt

        fields["initial_prompt"] = build_write_query_prompt(fields["question"], fields["schema_prompt"])
    required_keys = ("question", "schema_prompt", "gold_sql", "sqlite_path", "initial_prompt")
    missing = [key for key in required_keys if not fields[key]]
    if missing:
        raise ValueError(f"Missing required SQL-agent verl fields: {', '.join(missing)}")
    return fields


@dataclass
class _SqlActionTransition:
    turn_index: int
    agent_step: str
    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float] | None
    response_text: str
    candidate_sql: str | None = None
    tool_ok: bool = False
    tool_error: str | None = None
    final_execution_reward: float = 0.0
    transition_reward: float = 0.0
    reward_discount_power: int = 0


def _rollout_extra_fields(
    *,
    final_sql: str | None,
    final_sql_source: str,
    num_execute_calls: int,
    num_check_calls: int,
    num_parse_errors: int,
    final_execution_reward: float,
    transition_reward: float,
    reward_scheme: str,
    reward_gamma: float,
    reward_discount_power: int | None,
    final_success_turn_index: int | None,
    selected_transition_index: int | None,
    num_sql_transitions: int,
    transition_selection: str,
    selected_transition_turn_index: int | None,
    selected_transition_agent_step: str | None,
    selected_transition_tool_ok: bool | None,
) -> dict[str, Any]:
    """Return only rollout-owned fields so verl can safely union batches."""

    return {
        "final_sql": final_sql,
        "final_sql_source": final_sql_source,
        "num_execute_calls": num_execute_calls,
        "num_check_calls": num_check_calls,
        "num_parse_errors": num_parse_errors,
        "final_execution_reward": final_execution_reward,
        "trajectory_reward": final_execution_reward,
        "transition_reward": transition_reward,
        "reward_scheme": reward_scheme,
        "reward_gamma": reward_gamma,
        "reward_discount_power": reward_discount_power,
        "final_success_turn_index": final_success_turn_index,
        "selected_transition_index": selected_transition_index,
        "num_sql_transitions": num_sql_transitions,
        "transition_selection": transition_selection,
        "selected_transition_turn_index": selected_transition_turn_index,
        "selected_transition_agent_step": selected_transition_agent_step,
        "selected_transition_tool_ok": selected_transition_tool_ok,
    }


@register("sql_agent")
class SpiderSqlAgentLoop(AgentLoopBase):
    """Run SQL write/check/rewrite rollouts inside verl's async AgentLoop."""

    def __init__(self, *args, **kwargs) -> None:
        if AgentLoopOutput is None:  # pragma: no cover - dependency guard.
            raise RuntimeError("Install verl to instantiate SpiderSqlAgentLoop.")
        super().__init__(*args, **kwargs)
        self.prompt_length = int(_cfg_get(self.rollout_config, ("prompt_length",), 2048))
        self.response_length = int(_cfg_get(self.rollout_config, ("response_length",), 2048))
        self.max_turns = int(_cfg_get(self.rollout_config, ("multi_turn", "max_assistant_turns"), 3))
        self.reward_scheme = _normalize_reward_scheme(
            _cfg_or_env(
                self.rollout_config,
                ("agent", "reward_scheme"),
                "SQL_AGENT_REWARD_SCHEME",
                REWARD_SCHEME_FINAL_SHARED,
            )
        )
        self.reward_gamma = _normalize_reward_gamma(
            _cfg_or_env(self.rollout_config, ("agent", "reward_gamma"), "SQL_AGENT_REWARD_GAMMA", 0.9)
        )
        self.transition_selection = _normalize_transition_selection(
            _cfg_or_env(
                self.rollout_config,
                ("agent", "transition_selection"),
                "SQL_AGENT_TRANSITION_SELECTION",
                TRANSITION_SELECTION_ROUND_ROBIN,
            )
        )
        self.sqlite_tool = SQLiteTool()

    async def _encode_user_prompt(self, content: str, *, remove_system_prompt: bool) -> list[int]:
        return await self.apply_chat_template(
            [{"role": "user", "content": content}],
            remove_system_prompt=remove_system_prompt,
        )

    def _decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=True)).strip()

    def _append_tokens(
        self,
        *,
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
        token_ids: list[int],
        mask_value: int,
        log_probs: list[float] | None = None,
    ) -> list[float] | None:
        if not token_ids:
            return response_logprobs
        before_len = len(response_ids)
        available = self.response_length - before_len
        if available <= 0:
            return response_logprobs
        trimmed = token_ids[:available]
        response_ids.extend(trimmed)
        response_mask.extend([mask_value] * len(trimmed))

        if response_logprobs is not None or log_probs is not None:
            if response_logprobs is None:
                response_logprobs = [0.0] * before_len
            values = list(log_probs or [])[: len(trimmed)]
            if len(values) < len(trimmed):
                values.extend([0.0] * (len(trimmed) - len(values)))
            response_logprobs.extend(float(value) for value in values)
        return response_logprobs

    def _remaining_generation_budget(self, response_ids: list[int]) -> int:
        return max(0, self.response_length - len(response_ids))

    def _sampling_params_for_remaining_budget(
        self,
        sampling_params: dict[str, Any],
        response_ids: list[int],
    ) -> dict[str, Any]:
        remaining = self._remaining_generation_budget(response_ids)
        params = dict(sampling_params)
        if "max_tokens" in params:
            params["max_tokens"] = min(int(params["max_tokens"]), remaining)
        elif "max_new_tokens" in params:
            params["max_new_tokens"] = min(int(params["max_new_tokens"]), remaining)
        else:
            params["max_tokens"] = remaining
        return params

    async def _generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        response_ids: list[int],
        priority: int,
        metrics: dict[str, Any],
    ) -> Any:
        params = self._sampling_params_for_remaining_budget(sampling_params, response_ids)
        if int(params.get("max_tokens") or params.get("max_new_tokens") or 0) <= 0:
            return None

        local_metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", local_metrics):
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=params,
                priority=priority,
            )
        metrics["generate_sequences"] = metrics.get("generate_sequences", 0.0) + float(
            local_metrics.get("generate_sequences", 0.0)
        )
        preempted = getattr(output, "num_preempted", None)
        if preempted is not None:
            metrics["num_preempted"] = max(int(metrics.get("num_preempted", -1)), int(preempted))
        return output

    async def _run_sqlite(self, sqlite_path: str | Path, sql: str) -> Any:
        loop = getattr(self, "loop", None) or asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.sqlite_tool.execute(sqlite_path, sql))

    async def _score_sql(self, sql: str, gold_sql: str, sqlite_path: str | Path) -> float:
        loop = getattr(self, "loop", None) or asyncio.get_running_loop()
        return float(await loop.run_in_executor(None, lambda: spider_execution_reward(sql, gold_sql, sqlite_path)))

    async def _append_user_prompt(
        self,
        *,
        content: str,
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float] | None,
    ) -> list[float] | None:
        token_ids = await self._encode_user_prompt(content, remove_system_prompt=True)
        turn_separator = list(getattr(self, "turn_separator", []) or [])
        if turn_separator and response_ids[-len(turn_separator) :] == turn_separator:
            turn_separator = []
        needed = len(turn_separator) + len(token_ids)
        if needed > self._remaining_generation_budget(response_ids):
            logger.warning("Skipping SQL-agent prompt because response_length budget is exhausted.")
            return response_logprobs
        response_logprobs = self._append_tokens(
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            token_ids=turn_separator,
            mask_value=0,
        )
        return self._append_tokens(
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            token_ids=token_ids,
            mask_value=0,
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> Any:
        rollout_start = time.perf_counter()
        priority = int(_to_python(priority) or 0)
        fields = _sample_fields(kwargs)
        sqlite_path = fields["sqlite_path"]
        gold_sql = fields["gold_sql"]
        prompt_ids = await self._encode_user_prompt(fields["initial_prompt"], remove_system_prompt=False)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] | None = None
        metrics: dict[str, Any] = {"tool_calls": 0.0, "compute_score": 0.0, "num_preempted": -1}

        previous_sql: str | None = None
        previous_execution: str | None = None
        previous_feedback: str | None = None
        final_sql: str | None = None
        final_sql_source = "none"
        final_success_turn_index: int | None = None
        last_executable_sql: str | None = None
        last_executable_turn_index: int | None = None
        reward: float | None = None
        final_execution_reward = 0.0
        reward_discount_power: int | None = None
        ran_out_of_turns = False
        num_execute_calls = 0
        num_check_calls = 0
        num_parse_errors = 0
        num_turns = 1
        request_id = f"{fields['uid']}-{priority}-{uuid4().hex}"
        sql_transitions: list[_SqlActionTransition] = []

        for turn_index in range(self.max_turns):
            if turn_index > 0:
                rewrite_prompt = build_rewrite_query_prompt(
                    fields["question"],
                    fields["schema_prompt"],
                    previous_sql=previous_sql or "",
                    previous_execution=previous_execution or "",
                    feedback=previous_feedback or "",
                )
                before_len = len(response_ids)
                response_logprobs = await self._append_user_prompt(
                    content=rewrite_prompt,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                )
                if len(response_ids) == before_len:
                    break
                num_turns += 1

            action_prompt_ids = (prompt_ids + response_ids)[-self.prompt_length :]
            output = await self._generate(
                request_id=request_id,
                prompt_ids=prompt_ids + response_ids,
                sampling_params=sampling_params,
                response_ids=response_ids,
                priority=priority,
                metrics=metrics,
            )
            if output is None:
                break
            generated_ids = list(getattr(output, "token_ids", output))
            generated_text = self._decode(generated_ids)
            generated_logprobs_raw = getattr(output, "log_probs", None)
            generated_logprobs = None
            if generated_logprobs_raw is not None:
                generated_logprobs = [float(value) for value in list(generated_logprobs_raw)[: len(generated_ids)]]
                if len(generated_logprobs) < len(generated_ids):
                    generated_logprobs.extend([0.0] * (len(generated_ids) - len(generated_logprobs)))
            sql_transitions.append(
                _SqlActionTransition(
                    turn_index=turn_index,
                    agent_step="write_query" if turn_index == 0 else "rewrite_query",
                    prompt_ids=action_prompt_ids,
                    response_ids=generated_ids,
                    response_logprobs=generated_logprobs,
                    response_text=generated_text,
                )
            )
            response_logprobs = self._append_tokens(
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                token_ids=generated_ids,
                mask_value=1,
                log_probs=getattr(output, "log_probs", None),
            )
            num_turns += 1

            candidate_sql = extract_sql_candidate(generated_text)
            if candidate_sql is None:
                num_parse_errors += 1
                previous_sql = None
                previous_execution = "No SQL query found. Return only one read-only SQLite SELECT query."
                previous_feedback = previous_execution
                continue
            sql_transitions[-1].candidate_sql = candidate_sql

            num_execute_calls += 1
            tool_metrics: dict[str, Any] = {}
            with simple_timer("tool_calls", tool_metrics):
                execution = await self._run_sqlite(sqlite_path, candidate_sql)
            metrics["tool_calls"] = metrics.get("tool_calls", 0.0) + float(tool_metrics.get("tool_calls", 0.0))
            feedback = _format_execution_feedback(execution.ok, execution.rows, execution.error)
            sql_transitions[-1].tool_ok = bool(execution.ok)
            sql_transitions[-1].tool_error = execution.error
            if execution.ok:
                last_executable_sql = candidate_sql
                last_executable_turn_index = turn_index

            check_prompt = build_check_query_prompt(
                fields["question"],
                fields["schema_prompt"],
                candidate_sql,
                feedback,
            )
            before_len = len(response_ids)
            response_logprobs = await self._append_user_prompt(
                content=check_prompt,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
            )
            if len(response_ids) == before_len:
                if execution.ok:
                    final_sql = candidate_sql
                    final_sql_source = "executed_successfully"
                    final_success_turn_index = turn_index
                    break
                previous_sql = candidate_sql
                previous_execution = feedback
                previous_feedback = feedback
                continue
            num_turns += 1

            check_output = await self._generate(
                request_id=request_id,
                prompt_ids=prompt_ids + response_ids,
                sampling_params=sampling_params,
                response_ids=response_ids,
                priority=priority,
                metrics=metrics,
            )
            if check_output is None:
                if execution.ok:
                    final_sql = candidate_sql
                    final_sql_source = "executed_successfully"
                    final_success_turn_index = turn_index
                    break
                previous_sql = candidate_sql
                previous_execution = feedback
                previous_feedback = feedback
                continue

            check_ids = list(getattr(check_output, "token_ids", check_output))
            check_text = self._decode(check_ids)
            response_logprobs = self._append_tokens(
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                token_ids=check_ids,
                mask_value=0,
                log_probs=getattr(check_output, "log_probs", None),
            )
            num_check_calls += 1
            num_turns += 1

            verdict = _checker_verdict(check_text)
            if verdict is True and execution.ok:
                final_sql = candidate_sql
                final_sql_source = "checker_approved"
                final_success_turn_index = turn_index
                break

            previous_sql = candidate_sql
            previous_execution = feedback
            previous_feedback = check_text
        else:
            ran_out_of_turns = True

        if (
            final_sql is None
            and ran_out_of_turns
            and last_executable_sql is not None
            and last_executable_turn_index == self.max_turns - 1
        ):
            final_sql = last_executable_sql
            final_sql_source = "ran_out_of_turns"
            final_success_turn_index = last_executable_turn_index

        if final_sql:
            score_metrics = {}
            with simple_timer("compute_score", score_metrics):
                final_execution_reward = await self._score_sql(final_sql, gold_sql, sqlite_path)
            metrics["compute_score"] = metrics.get("compute_score", 0.0) + float(
                score_metrics.get("compute_score", 0.0)
            )
        transition_rewards = _compute_transition_rewards(
            final_execution_reward=final_execution_reward,
            reward_scheme=self.reward_scheme,
            reward_gamma=self.reward_gamma,
            num_transitions=len(sql_transitions),
        )
        for transition, (transition_reward, discount_power) in zip(
            sql_transitions, transition_rewards, strict=False
        ):
            transition.final_execution_reward = final_execution_reward
            transition.transition_reward = transition_reward
            transition.reward_discount_power = discount_power

        selected_transition_index: int | None = None
        selected_transition: _SqlActionTransition | None = None
        full_response_tokens = len(response_ids)
        full_trainable_tokens = int(sum(int(value) for value in response_mask))
        if self.reward_scheme == REWARD_SCHEME_CHAIN_FINAL and sql_transitions:
            selected_transition_index = _select_transition_index(
                uid=fields["uid"],
                num_transitions=len(sql_transitions),
                selection=self.transition_selection,
            )
            selected_transition = sql_transitions[selected_transition_index]
            prompt_ids = selected_transition.prompt_ids[-self.prompt_length :]
            response_ids = selected_transition.response_ids[: self.response_length]
            response_mask = [1] * len(response_ids)
            response_logprobs = (
                selected_transition.response_logprobs[: self.response_length]
                if selected_transition.response_logprobs is not None
                else None
            )
            reward = selected_transition.transition_reward
            reward_discount_power = selected_transition.reward_discount_power
        else:
            reward = final_execution_reward
            reward_discount_power = 0 if final_execution_reward > 0.0 else None
            response_ids = response_ids[: self.response_length]
            response_mask = response_mask[: self.response_length]
            if response_logprobs is not None:
                response_logprobs = response_logprobs[: self.response_length]

        rollout_time_sec = max(time.perf_counter() - rollout_start, 1e-9)
        prompt_tokens = len(prompt_ids)
        response_tokens = len(response_ids)
        trainable_tokens = int(sum(int(value) for value in response_mask))
        total_tokens = prompt_tokens + response_tokens
        metrics.update(
            {
                "rollout_time_sec": rollout_time_sec,
                "generate_time_sec": float(metrics.get("generate_sequences", 0.0)),
                "tool_time_sec": float(metrics.get("tool_calls", 0.0)),
                "reward_time_sec": float(metrics.get("compute_score", 0.0)),
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "trainable_tokens": trainable_tokens,
                "total_tokens": total_tokens,
                "full_response_tokens": full_response_tokens,
                "full_trainable_tokens": full_trainable_tokens,
                "tokens_per_sec_total": total_tokens / rollout_time_sec,
                "tokens_per_sec_trainable": trainable_tokens / rollout_time_sec,
                "trajectories_per_sec": 1.0 / rollout_time_sec,
                "num_turns": num_turns,
                "num_execute_calls": num_execute_calls,
                "num_check_calls": num_check_calls,
                "num_parse_errors": num_parse_errors,
                "final_execution_reward": final_execution_reward,
                "transition_reward": float(reward),
                "reward_scheme": self.reward_scheme,
                "reward_gamma": self.reward_gamma,
                "reward_discount_power": -1 if reward_discount_power is None else reward_discount_power,
                "final_success_turn_index": -1 if final_success_turn_index is None else final_success_turn_index,
                "selected_transition_index": -1 if selected_transition_index is None else selected_transition_index,
                "selected_transition_turn_index": -1
                if selected_transition is None
                else selected_transition.turn_index,
                "selected_transition_tool_ok": -1
                if selected_transition is None
                else int(selected_transition.tool_ok),
                "selected_transition_agent_step": "full_trajectory"
                if selected_transition is None
                else selected_transition.agent_step,
                "num_sql_transitions": len(sql_transitions),
                "transition_selection": self.transition_selection,
            }
        )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=float(reward),
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=_rollout_extra_fields(
                final_sql=final_sql,
                final_sql_source=final_sql_source,
                num_execute_calls=num_execute_calls,
                num_check_calls=num_check_calls,
                num_parse_errors=num_parse_errors,
                final_execution_reward=final_execution_reward,
                transition_reward=float(reward),
                reward_scheme=self.reward_scheme,
                reward_gamma=self.reward_gamma,
                reward_discount_power=reward_discount_power,
                final_success_turn_index=final_success_turn_index,
                selected_transition_index=selected_transition_index,
                num_sql_transitions=len(sql_transitions),
                transition_selection=self.transition_selection,
                selected_transition_turn_index=None if selected_transition is None else selected_transition.turn_index,
                selected_transition_agent_step=None if selected_transition is None else selected_transition.agent_step,
                selected_transition_tool_ok=None if selected_transition is None else selected_transition.tool_ok,
            ),
        )
