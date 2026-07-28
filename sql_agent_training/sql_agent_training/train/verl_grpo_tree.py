"""Tree-structured GRPO support for verl SQL-agent rollouts.

S3/S4 use one training row per tree node: the prompt is the node's parent
state, the response is one sampled SQL action, and the group id is the parent
state id. The custom worker keeps verl's repeated-batch shape fixed by filling
unused tree slots with fully masked dummy rows.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from sql_agent_training.agent.actions import extract_sql_candidate
from sql_agent_training.agent.prompts import build_check_query_prompt, build_rewrite_query_prompt
from sql_agent_training.agent.sql_agent_loop import _checker_verdict, _format_execution_feedback
from sql_agent_training.train.verl_sql_agent_loop import (
    AgentLoopOutput,
    SpiderSqlAgentLoop,
    _cfg_get,
    _env_float,
    _sample_fields,
    _to_python,
    register,
    simple_timer,
)

TREE_ADV_ESTIMATOR_NAME = "grpo_tree"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _cfg_agent_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    agent_config = getattr(config, "agent", None)
    if agent_config is None and isinstance(config, dict):
        agent_config = config.get("agent")
    if isinstance(agent_config, dict):
        return agent_config.get(key, default)
    if agent_config is not None and hasattr(agent_config, key):
        return getattr(agent_config, key)
    return default


def tree_slot_count(*, branch_n: int, beam_size: int, max_turns: int) -> int:
    """Return the fixed number of repeated slots required by bounded S3."""

    if branch_n <= 0:
        raise ValueError(f"branch_n must be positive, got {branch_n}.")
    if beam_size <= 0:
        raise ValueError(f"beam_size must be positive, got {beam_size}.")
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}.")
    return branch_n + max(0, max_turns - 1) * beam_size * branch_n


def root_state_id(root_uid: str) -> str:
    return f"{root_uid}::root"


def child_state_id(
    *,
    root_uid: str,
    next_turn_index: int,
    sql: str | None,
    execution_feedback: str,
    checker_feedback: str | None,
) -> str:
    """Build a stable id for a child state without exposing gold SQL."""

    payload = {
        "root_uid": root_uid,
        "next_turn_index": next_turn_index,
        "sql": sql or "",
        "execution_feedback": execution_feedback,
        "checker_feedback": checker_feedback or "",
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    return f"{root_uid}::t{next_turn_index}::{digest}"


@dataclass
class TreeParentState:
    """A decision state whose children form one GRPO group."""

    state_id: str
    root_uid: str
    turn_index: int
    prompt_content: str
    prompt_ids: list[int]
    source_node_id: str | None = None


@dataclass
class TreeNode:
    """One sampled SQL action from a parent state."""

    node_id: str
    parent_state_id: str
    root_uid: str
    turn_index: int
    prompt_ids: list[int]
    response_ids: list[int]
    response_text: str
    sql: str | None
    execution_ok: bool
    execution_feedback: str
    checker_feedback: str | None
    checker_verdict: bool | None
    correct: bool
    child_state: TreeParentState | None = None
    children: list["TreeNode"] = field(default_factory=list)
    value: float = 0.0

    @property
    def trainable(self) -> bool:
        return bool(self.response_ids)


def proxy_score(node: TreeNode) -> float:
    """Score failed frontier nodes using only agent-visible signals."""

    score = 0.0
    if node.sql:
        score += 0.2
    else:
        score -= 2.0
    if node.execution_ok:
        score += 1.0
        if "row_count=0" not in node.execution_feedback:
            score += 0.2
    else:
        score -= 0.5
    if node.checker_verdict is True:
        score += 0.3
    elif node.checker_verdict is False:
        score -= 0.1
    return score


def select_frontier_nodes(
    frontier: list[TreeNode],
    *,
    beam_size: int,
    rng: random.Random,
    tau: float,
    epsilon_random: float,
) -> list[TreeNode]:
    """Select a bounded failed frontier without using final gold reward."""

    if len(frontier) <= beam_size:
        return list(frontier)
    if beam_size <= 0:
        return []
    epsilon_random = min(max(float(epsilon_random), 0.0), 1.0)
    if rng.random() < epsilon_random:
        return rng.sample(frontier, beam_size)

    tau = max(float(tau), 1e-6)
    remaining = list(frontier)
    selected: list[TreeNode] = []
    for _ in range(min(beam_size, len(remaining))):
        scores = [proxy_score(node) / tau for node in remaining]
        max_score = max(scores)
        weights = [math.exp(score - max_score) for score in scores]
        total = sum(weights)
        threshold = rng.random() * total
        cumulative = 0.0
        chosen_index = len(remaining) - 1
        for index, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= threshold:
                chosen_index = index
                break
        selected.append(remaining.pop(chosen_index))
    return selected


def backup_tree_values(nodes: list[TreeNode], *, gamma: float, executable_fallback_beta: float = 0.0) -> None:
    """Assign tree values with final-reward mean backup and optional executable fallback."""

    def value_for(node: TreeNode) -> float:
        if node.correct:
            node.value = 1.0
        elif node.children:
            backed_up_value = float(gamma) * sum(value_for(child) for child in node.children) / len(node.children)
            executable_value = float(executable_fallback_beta) if node.execution_ok else 0.0
            node.value = max(backed_up_value, executable_value)
        elif node.execution_ok:
            node.value = float(executable_fallback_beta)
        else:
            node.value = 0.0
        return node.value

    for node in nodes:
        value_for(node)


def compute_grpo_tree_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | None = None,
    config: Any = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GRPO advantages grouped by tree parent state id."""

    del config
    if index is None:
        index = np.array([str(row) for row in range(response_mask.size(0))], dtype=object)
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        valid_rows = response_mask.sum(dim=-1) > 0
        groups: dict[str, list[torch.Tensor]] = {}
        for row in range(response_mask.size(0)):
            if not bool(valid_rows[row].item()):
                continue
            key = str(index[row])
            groups.setdefault(key, []).append(scores[row])

        stats: dict[str, tuple[torch.Tensor, torch.Tensor, bool]] = {}
        for key, values in groups.items():
            stacked = torch.stack(values)
            if len(values) < 2:
                stats[key] = (stacked.mean(), torch.zeros_like(stacked.mean()), False)
                continue
            mean = stacked.mean()
            std = stacked.std(unbiased=False)
            stats[key] = (mean, std, bool((std > epsilon).item()))

        advantages = torch.zeros_like(token_level_rewards, dtype=torch.float32)
        for row in range(response_mask.size(0)):
            if not bool(valid_rows[row].item()):
                continue
            mean, std, usable = stats[str(index[row])]
            if not usable:
                continue
            if norm_adv_by_std_in_grpo:
                scalar = (scores[row] - mean) / (std + epsilon)
            else:
                scalar = scores[row] - mean
            advantages[row] = scalar.to(dtype=advantages.dtype) * response_mask[row].to(dtype=advantages.dtype)
        return advantages, advantages


def patch_verl_compute_advantage() -> bool:
    """Patch verl's driver-side advantage dispatcher for S3 tree groups."""

    try:
        from verl.trainer.ppo import ray_trainer
    except ImportError:
        return False

    original = getattr(ray_trainer, "compute_advantage", None)
    if original is None or getattr(original, "_sql_agent_tree_patched", False):
        return False

    def compute_advantage_with_tree_groups(
        data,
        adv_estimator,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config=None,
    ):
        if str(adv_estimator) != TREE_ADV_ESTIMATOR_NAME:
            return original(
                data,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )
        if "tree_group_id" not in data.non_tensor_batch:
            raise ValueError("S3 grpo_tree requires non_tensor_batch['tree_group_id'].")
        advantages, returns = compute_grpo_tree_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["tree_group_id"],
            config=config,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    compute_advantage_with_tree_groups._sql_agent_tree_patched = True  # type: ignore[attr-defined]
    setattr(ray_trainer, "compute_advantage", compute_advantage_with_tree_groups)
    return True


try:  # pragma: no cover - exercised in the verl runtime.
    import ray
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopManager as _VerlAgentLoopManager,
        AgentLoopWorker as _VerlAgentLoopWorker,
        DictConfigWrap,
    )
except ImportError:  # pragma: no cover - unit tests can import without verl.
    ray = None  # type: ignore[assignment]
    _VerlAgentLoopManager = None  # type: ignore[assignment]
    _VerlAgentLoopWorker = None  # type: ignore[assignment]
    DictConfigWrap = None  # type: ignore[assignment]


@register("sql_agent_tree")
class TreeSqlAgentLoop(SpiderSqlAgentLoop):
    """Generate a bounded SQL tree and expose nodes as prompt/SQL pairs."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        agent_config = getattr(self.rollout_config, "agent", None)
        self.branch_n = int(_cfg_agent_get(self.rollout_config, "tree_branch_n", _env_int("GRPO_TREE_BRANCH_N", 4)))
        self.beam_size = int(_cfg_agent_get(self.rollout_config, "tree_beam_size", _env_int("GRPO_TREE_BEAM_SIZE", 4)))
        self.beam_tau = float(
            _cfg_agent_get(self.rollout_config, "tree_beam_tau", _env_float("GRPO_TREE_BEAM_TAU", 1.0))
        )
        self.beam_epsilon_random = float(
            _cfg_agent_get(
                self.rollout_config,
                "tree_beam_epsilon_random",
                _env_float("GRPO_TREE_BEAM_EPSILON_RANDOM", 0.1),
            )
        )
        self.prune_on_gold_reward = bool(
            _cfg_agent_get(
                self.rollout_config,
                "tree_prune_on_gold_reward",
                _env_bool("GRPO_TREE_PRUNE_ON_GOLD_REWARD", True),
            )
        )
        del agent_config

    async def _parent_state_for_node(
        self,
        *,
        fields: dict[str, str],
        root_uid: str,
        node: TreeNode,
    ) -> TreeParentState | None:
        if (node.correct and self.prune_on_gold_reward) or node.turn_index + 1 >= self.max_turns:
            return None
        prompt_content = build_rewrite_query_prompt(
            fields["question"],
            fields["schema_prompt"],
            previous_sql=node.sql or "",
            previous_execution=node.execution_feedback,
            feedback=node.checker_feedback or node.execution_feedback,
        )
        return TreeParentState(
            state_id=child_state_id(
                root_uid=root_uid,
                next_turn_index=node.turn_index + 1,
                sql=node.sql,
                execution_feedback=node.execution_feedback,
                checker_feedback=node.checker_feedback,
            ),
            root_uid=root_uid,
            turn_index=node.turn_index + 1,
            prompt_content=prompt_content,
            prompt_ids=await self._encode_user_prompt(prompt_content, remove_system_prompt=False),
            source_node_id=node.node_id,
        )

    async def _generate_tree_child(
        self,
        *,
        parent: TreeParentState,
        child_index: int,
        fields: dict[str, str],
        root_uid: str,
        sampling_params: dict[str, Any],
        priority: int,
        metrics: dict[str, Any],
    ) -> TreeNode:
        request_id = f"{root_uid}-{parent.state_id}-{child_index}-{uuid4().hex}"
        output = await self._generate(
            request_id=request_id,
            prompt_ids=parent.prompt_ids,
            sampling_params=sampling_params,
            response_ids=[],
            priority=priority,
            metrics=metrics,
        )
        generated_ids = [] if output is None else list(getattr(output, "token_ids", output))
        generated_text = self._decode(generated_ids) if generated_ids else ""
        candidate_sql = extract_sql_candidate(generated_text)

        execution_ok = False
        execution_feedback = "No SQL query found. Return only one read-only SQLite SELECT query."
        checker_feedback: str | None = execution_feedback
        checker_verdict: bool | None = False
        correct = False

        if candidate_sql is not None:
            tool_metrics: dict[str, Any] = {}
            with simple_timer("tool_calls", tool_metrics):
                execution = await self._run_sqlite(fields["sqlite_path"], candidate_sql)
            metrics["tool_calls"] = metrics.get("tool_calls", 0.0) + float(tool_metrics.get("tool_calls", 0.0))
            execution_ok = bool(execution.ok)
            execution_feedback = _format_execution_feedback(execution.ok, execution.rows, execution.error)
            score_metrics: dict[str, Any] = {}
            with simple_timer("compute_score", score_metrics):
                correct = bool(await self._score_sql(candidate_sql, fields["gold_sql"], fields["sqlite_path"]) > 0.0)
            metrics["compute_score"] = metrics.get("compute_score", 0.0) + float(
                score_metrics.get("compute_score", 0.0)
            )

            check_prompt = build_check_query_prompt(
                fields["question"],
                fields["schema_prompt"],
                candidate_sql,
                execution_feedback,
            )
            check_prompt_ids = await self._encode_user_prompt(check_prompt, remove_system_prompt=False)
            check_output = await self._generate(
                request_id=f"{request_id}-check",
                prompt_ids=check_prompt_ids,
                sampling_params=sampling_params,
                response_ids=[],
                priority=priority,
                metrics=metrics,
            )
            check_ids = [] if check_output is None else list(getattr(check_output, "token_ids", check_output))
            checker_feedback = self._decode(check_ids) if check_ids else execution_feedback
            checker_verdict = _checker_verdict(checker_feedback)

        node = TreeNode(
            node_id=f"{parent.state_id}::child{child_index}",
            parent_state_id=parent.state_id,
            root_uid=root_uid,
            turn_index=parent.turn_index,
            prompt_ids=parent.prompt_ids,
            response_ids=generated_ids,
            response_text=generated_text,
            sql=candidate_sql,
            execution_ok=execution_ok,
            execution_feedback=execution_feedback,
            checker_feedback=checker_feedback,
            checker_verdict=checker_verdict,
            correct=correct,
        )
        node.child_state = await self._parent_state_for_node(fields=fields, root_uid=root_uid, node=node)
        return node

    async def _expand_parent(
        self,
        *,
        parent: TreeParentState,
        fields: dict[str, str],
        root_uid: str,
        sampling_params: dict[str, Any],
        priority: int,
        metrics: dict[str, Any],
    ) -> list[TreeNode]:
        tasks = [
            asyncio.create_task(
                self._generate_tree_child(
                    parent=parent,
                    child_index=child_index,
                    fields=fields,
                    root_uid=root_uid,
                    sampling_params=sampling_params,
                    priority=priority,
                    metrics=metrics,
                )
            )
            for child_index in range(self.branch_n)
        ]
        children = await asyncio.gather(*tasks)
        return list(children)

    def _node_to_output(self, node: TreeNode, metrics: dict[str, Any]) -> Any:
        if AgentLoopOutput is None:  # pragma: no cover - dependency guard.
            raise RuntimeError("Install verl to build AgentLoopOutput.")
        return AgentLoopOutput(
            prompt_ids=node.prompt_ids,
            response_ids=node.response_ids,
            response_mask=[1] * len(node.response_ids),
            response_logprobs=None,
            reward_score=float(node.value),
            num_turns=node.turn_index + 1,
            metrics=dict(metrics),
            extra_fields={
                "tree_group_id": node.parent_state_id,
                "tree_root_uid": node.root_uid,
                "tree_turn_index": node.turn_index,
                "tree_is_dummy": False,
                "tree_node_id": node.node_id,
                "tree_sql": node.sql,
                "tree_execution_ok": node.execution_ok,
                "tree_correct": node.correct,
                "tree_value": float(node.value),
            },
        )

    def _dummy_output(self, *, root_uid: str, prompt_ids: list[int]) -> Any:
        if AgentLoopOutput is None:  # pragma: no cover - dependency guard.
            raise RuntimeError("Install verl to build AgentLoopOutput.")
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=[],
            response_mask=[],
            response_logprobs=None,
            reward_score=0.0,
            num_turns=0,
            metrics={
                "rollout_time_sec": 0.0,
                "generate_time_sec": 0.0,
                "tool_time_sec": 0.0,
                "reward_time_sec": 0.0,
                "prompt_tokens": len(prompt_ids),
                "response_tokens": 0,
                "trainable_tokens": 0,
                "total_tokens": len(prompt_ids),
                "tokens_per_sec_total": 0.0,
                "tokens_per_sec_trainable": 0.0,
                "trajectories_per_sec": 0.0,
                "num_execute_calls": 0,
                "num_check_calls": 0,
                "num_parse_errors": 0,
            },
            extra_fields={
                "tree_group_id": f"{root_uid}::dummy",
                "tree_root_uid": root_uid,
                "tree_turn_index": -1,
                "tree_is_dummy": True,
                "tree_node_id": None,
                "tree_sql": None,
                "tree_execution_ok": False,
                "tree_correct": False,
                "tree_value": 0.0,
            },
        )

    async def run_tree(
        self,
        sampling_params: dict[str, Any],
        *,
        root_uid: str,
        slot_count: int,
        priority: int = 0,
        **kwargs,
    ) -> list[Any]:
        rollout_start = time.perf_counter()
        fields = _sample_fields(kwargs)
        fields["uid"] = root_uid
        root_prompt_ids = await self._encode_user_prompt(fields["initial_prompt"], remove_system_prompt=False)
        root = TreeParentState(
            state_id=root_state_id(root_uid),
            root_uid=root_uid,
            turn_index=0,
            prompt_content=fields["initial_prompt"],
            prompt_ids=root_prompt_ids,
        )

        rng_seed = int(hashlib.sha1(root_uid.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(rng_seed)
        metrics: dict[str, Any] = {"generate_sequences": 0.0, "tool_calls": 0.0, "compute_score": 0.0}
        all_nodes: list[TreeNode] = []
        selected_parents = [root]

        for depth in range(self.max_turns):
            expanded_nodes: list[TreeNode] = []
            for parent in selected_parents:
                children = await self._expand_parent(
                    parent=parent,
                    fields=fields,
                    root_uid=root_uid,
                    sampling_params=sampling_params,
                    priority=priority,
                    metrics=metrics,
                )
                expanded_nodes.extend(children)
                if parent.source_node_id:
                    for node in all_nodes:
                        if node.node_id == parent.source_node_id:
                            node.children = children
                            break
            all_nodes.extend(expanded_nodes)
            if depth + 1 >= self.max_turns:
                break
            frontier = [node for node in expanded_nodes if not node.correct and node.child_state is not None]
            selected_nodes = select_frontier_nodes(
                frontier,
                beam_size=self.beam_size,
                rng=rng,
                tau=self.beam_tau,
                epsilon_random=self.beam_epsilon_random,
            )
            selected_parents = [node.child_state for node in selected_nodes if node.child_state is not None]
            if not selected_parents:
                break

        root_nodes = [node for node in all_nodes if node.turn_index == 0]
        executable_fallback_beta = self.executable_fallback_beta if self.reward_scheme == "tree_executable" else 0.0
        backup_tree_values(
            root_nodes,
            gamma=self.reward_gamma,
            executable_fallback_beta=executable_fallback_beta,
        )
        rollout_time_sec = max(time.perf_counter() - rollout_start, 1e-9)
        metrics.update(
            {
                "rollout_time_sec": rollout_time_sec,
                "generate_time_sec": float(metrics.get("generate_sequences", 0.0)),
                "tool_time_sec": float(metrics.get("tool_calls", 0.0)),
                "reward_time_sec": float(metrics.get("compute_score", 0.0)),
                "prompt_tokens": sum(len(node.prompt_ids) for node in all_nodes),
                "response_tokens": sum(len(node.response_ids) for node in all_nodes),
                "trainable_tokens": sum(len(node.response_ids) for node in all_nodes),
                "total_tokens": sum(len(node.prompt_ids) + len(node.response_ids) for node in all_nodes),
                "tokens_per_sec_total": (
                    sum(len(node.prompt_ids) + len(node.response_ids) for node in all_nodes) / rollout_time_sec
                ),
                "tokens_per_sec_trainable": (
                    sum(len(node.response_ids) for node in all_nodes) / rollout_time_sec
                ),
                "trajectories_per_sec": 1.0 / rollout_time_sec,
                "tree_nodes": len(all_nodes),
                "tree_slot_count": slot_count,
                "tree_dummy_count": max(0, slot_count - len(all_nodes)),
            }
        )
        if len(all_nodes) > slot_count:
            raise ValueError(f"S3 tree produced {len(all_nodes)} nodes, exceeding slot_count={slot_count}.")
        outputs = [self._node_to_output(node, metrics) for node in all_nodes]
        outputs.extend(
            self._dummy_output(root_uid=root_uid, prompt_ids=root_prompt_ids)
            for _ in range(slot_count - len(outputs))
        )
        return outputs

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> Any:
        slot_count = tree_slot_count(branch_n=self.branch_n, beam_size=self.beam_size, max_turns=self.max_turns)
        outputs = await self.run_tree(
            sampling_params,
            root_uid=str(uuid4()),
            slot_count=slot_count,
            priority=priority,
            **kwargs,
        )
        return outputs[0]


if _VerlAgentLoopWorker is not None:

    class TreeRewardAgentLoopWorker(_VerlAgentLoopWorker):
        """Map repeated S3 slots to fixed tree node outputs."""

        def _tree_slot_count(self) -> int:
            branch_n = int(_cfg_agent_get(self.rollout_config, "tree_branch_n", _env_int("GRPO_TREE_BRANCH_N", 4)))
            beam_size = int(_cfg_agent_get(self.rollout_config, "tree_beam_size", _env_int("GRPO_TREE_BEAM_SIZE", 4)))
            max_turns = int(getattr(self.rollout_config.multi_turn, "max_assistant_turns", _env_int("MAX_TURNS", 3)))
            return tree_slot_count(branch_n=branch_n, beam_size=beam_size, max_turns=max_turns)

        async def generate_sequences(self, batch):  # type: ignore[no-untyped-def]
            config = self.rollout_config
            sampling_params = dict(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=1.0,
                logprobs=config.calculate_log_probs,
            )
            if batch.meta_info.get("validate", False):
                sampling_params["top_p"] = config.val_kwargs.top_p
                sampling_params["top_k"] = config.val_kwargs.top_k
                sampling_params["temperature"] = config.val_kwargs.temperature

            slot_count = self._tree_slot_count()
            if len(batch) % slot_count != 0:
                raise ValueError(
                    f"S3 tree rollout requires each root sample to have {slot_count} repeated slots; "
                    f"got worker batch size {len(batch)}."
                )

            tree_loop = TreeSqlAgentLoop(
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
            )

            internal_outputs = []
            for start in range(0, len(batch), slot_count):
                root_kwargs = {key: value[start] for key, value in batch.non_tensor_batch.items()}
                root_uid = str(_to_python(batch.non_tensor_batch.get("uid", np.array([uuid4().hex]))[start]))
                uid_values = batch.non_tensor_batch.get("uid", [root_uid] * len(batch))
                repeated_uids = [
                    str(_to_python(value))
                    for value in uid_values[start : start + slot_count]
                ]
                if any(uid != root_uid for uid in repeated_uids):
                    raise ValueError("S3 tree rollout expects contiguous repeated slots with the same uid.")
                outputs = await tree_loop.run_tree(
                    sampling_params,
                    root_uid=root_uid,
                    slot_count=slot_count,
                    priority=start,
                    **root_kwargs,
                )
                for offset, output in enumerate(outputs):
                    slot_kwargs = {key: value[start + offset] for key, value in batch.non_tensor_batch.items()}
                    internal_outputs.append(await self._agent_loop_postprocess(output, **slot_kwargs))

            return self._postprocess(internal_outputs, input_non_tensor_batch=batch.non_tensor_batch)


    class TreeRewardAgentLoopManager(_VerlAgentLoopManager):
        """Use the S3 tree worker while keeping verl's rollout manager API."""

        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            super().__init__(*args, **kwargs)
            self.agent_loop_workers_class = ray.remote(TreeRewardAgentLoopWorker)

else:

    class TreeRewardAgentLoopWorker:  # pragma: no cover - dependency fallback.
        pass


    class TreeRewardAgentLoopManager:  # pragma: no cover - dependency fallback.
        pass
