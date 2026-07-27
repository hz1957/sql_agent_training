"""verl AgentLoop implementation for Spider SQL-agent GRPO.

The loop keeps token-in/token-out integrity for model-generated tokens: decoded
text is used only for SQL parsing and reward computation, while the original
generated token ids are returned to verl for training.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from sql_agent_training.agent.actions import extract_sql_candidate
from sql_agent_training.agent.prompts import build_check_query_prompt, build_rewrite_query_prompt
from sql_agent_training.agent.sql_agent_loop import _checker_verdict, _format_execution_feedback
from sql_agent_training.env.sqlite_tool import SQLiteTool
from sql_agent_training.reward.spider_reward import spider_execution_reward

logger = logging.getLogger(__name__)

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


@register("sql_agent")
class SpiderSqlAgentLoop(AgentLoopBase):
    """Run SQL write/check/rewrite rollouts inside verl's async AgentLoop."""

    def __init__(self, *args, **kwargs) -> None:
        if AgentLoopOutput is None:  # pragma: no cover - dependency guard.
            raise RuntimeError("Install verl to instantiate SpiderSqlAgentLoop.")
        super().__init__(*args, **kwargs)
        self.response_length = int(_cfg_get(self.rollout_config, ("response_length",), 2048))
        self.max_turns = int(_cfg_get(self.rollout_config, ("multi_turn", "max_assistant_turns"), 3))
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
        reward: float | None = None
        num_execute_calls = 0
        num_check_calls = 0
        num_parse_errors = 0
        num_turns = 1
        request_id = f"{fields['uid']}-{priority}-{uuid4().hex}"

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

            num_execute_calls += 1
            tool_metrics: dict[str, Any] = {}
            with simple_timer("tool_calls", tool_metrics):
                execution = await self._run_sqlite(sqlite_path, candidate_sql)
            metrics["tool_calls"] = metrics.get("tool_calls", 0.0) + float(tool_metrics.get("tool_calls", 0.0))
            feedback = _format_execution_feedback(execution.ok, execution.rows, execution.error)
            candidate_reward = None
            if execution.ok:
                score_metrics: dict[str, Any] = {}
                with simple_timer("compute_score", score_metrics):
                    candidate_reward = await self._score_sql(candidate_sql, gold_sql, sqlite_path)
                metrics["compute_score"] = metrics.get("compute_score", 0.0) + float(
                    score_metrics.get("compute_score", 0.0)
                )
                final_sql = candidate_sql
                final_sql_source = "executed_successfully"
                reward = candidate_reward

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
                reward = candidate_reward
                break

            previous_sql = candidate_sql
            previous_execution = feedback
            previous_feedback = check_text

        if final_sql and reward is None:
            score_metrics = {}
            with simple_timer("compute_score", score_metrics):
                reward = await self._score_sql(final_sql, gold_sql, sqlite_path)
            metrics["compute_score"] = metrics.get("compute_score", 0.0) + float(
                score_metrics.get("compute_score", 0.0)
            )
        elif not final_sql:
            reward = 0.0

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs is not None else None,
            reward_score=float(reward),
            num_turns=num_turns,
            metrics=metrics,
            extra_fields={
                "uid": fields["uid"],
                "db_id": fields["db_id"],
                "final_sql": final_sql,
                "final_sql_source": final_sql_source,
                "num_execute_calls": num_execute_calls,
                "num_check_calls": num_check_calls,
                "num_parse_errors": num_parse_errors,
            },
        )
