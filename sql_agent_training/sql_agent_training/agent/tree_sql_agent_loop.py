"""Gold-free tree search loop for SQL agent evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sql_agent_training.agent.actions import extract_sql_candidate
from sql_agent_training.agent.model_client import ModelClient, ModelRequest, ModelResponse
from sql_agent_training.agent.sql_agent_loop import (
    SqlAgentInput,
    SqlAgentLoop,
    _checker_verdict,
    _format_execution_feedback,
    _irrecoverable_sqlite_error_reason,
)
from sql_agent_training.agent.trace_format import AgentTrajectory, AgentTurn
from sql_agent_training.env.sqlite_tool import SQLiteTool
from sql_agent_training.reward.spider_reward import spider_execution_reward


def tree_eval_slot_count(*, branch_n: int, beam_size: int, max_turns: int) -> int:
    """Return the maximum number of generated SQL candidates for bounded tree eval."""

    if branch_n <= 0:
        raise ValueError(f"branch_n must be positive, got {branch_n}.")
    if beam_size <= 0:
        raise ValueError(f"beam_size must be positive, got {beam_size}.")
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}.")
    return branch_n + max(0, max_turns - 1) * beam_size * branch_n


def _root_state_id(root_uid: str) -> str:
    return f"{root_uid}::root"


def _child_state_id(
    *,
    root_uid: str,
    next_turn_index: int,
    sql: str | None,
    execution_feedback: str,
    checker_feedback: str | None,
) -> str:
    payload = {
        "root_uid": root_uid,
        "next_turn_index": next_turn_index,
        "sql": sql or "",
        "execution_feedback": execution_feedback,
        "checker_feedback": checker_feedback or "",
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    return f"{root_uid}::t{next_turn_index}::{digest}"


def _seed_for_rollout(seed: int, rollout_id: str) -> int:
    payload = f"{seed}\x1f{rollout_id}"
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8], 16)


@dataclass(frozen=True)
class TreeEvalParentState:
    """A gold-free decision state selected for tree expansion."""

    state_id: str
    turn_index: int
    previous_sql: str | None = None
    previous_execution: str | None = None
    previous_feedback: str | None = None
    source_node_id: str | None = None


@dataclass
class TreeEvalNode:
    """One generated SQL candidate in a tree eval rollout."""

    node_id: str
    parent_state_id: str
    turn_index: int
    branch_index: int
    sequence_index: int
    response_text: str
    sql: str | None
    execution_ok: bool
    execution_feedback: str
    checker_feedback: str | None
    checker_verdict: bool | None
    prune_reason: str | None = None
    check_called: bool = False
    child_state: TreeEvalParentState | None = None
    children: list["TreeEvalNode"] = field(default_factory=list)

    @property
    def is_terminal_candidate(self) -> bool:
        """Whether this node is a checker-approved executable leaf."""

        return bool(self.sql and self.execution_ok and self.checker_verdict is True)

    @property
    def is_proxy_pruned_leaf(self) -> bool:
        return self.is_terminal_candidate or self.prune_reason is not None


def tree_eval_proxy_score(node: TreeEvalNode) -> float:
    """Score nodes using only signals visible at inference time."""

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
    if node.prune_reason is not None:
        score -= 2.0
    if node.checker_verdict is True:
        score += 0.3
    elif node.checker_verdict is False:
        score -= 0.1
    return score


def select_tree_eval_frontier(
    frontier: list[TreeEvalNode],
    *,
    beam_size: int,
    rng: random.Random,
    tau: float,
    epsilon_random: float,
) -> list[TreeEvalNode]:
    """Select failed frontier nodes for the next rewrite depth without gold labels."""

    if len(frontier) <= beam_size:
        return list(frontier)
    if beam_size <= 0:
        return []
    epsilon_random = min(max(float(epsilon_random), 0.0), 1.0)
    if rng.random() < epsilon_random:
        return rng.sample(frontier, beam_size)

    tau = max(float(tau), 1e-6)
    remaining = list(frontier)
    selected: list[TreeEvalNode] = []
    for _ in range(min(beam_size, len(remaining))):
        scores = [tree_eval_proxy_score(node) / tau for node in remaining]
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


class TreeSqlAgentEvalLoop(SqlAgentLoop):
    """Run bounded tree inference without using gold SQL until final scoring."""

    def __init__(
        self,
        *,
        max_turns: int = 3,
        branch_n: int = 4,
        beam_size: int = 2,
        beam_tau: float = 1.0,
        beam_epsilon_random: float = 0.0,
        seed: int = 0,
        sqlite_tool: SQLiteTool | None = None,
    ) -> None:
        super().__init__(max_turns=max_turns, sqlite_tool=sqlite_tool)
        tree_eval_slot_count(branch_n=branch_n, beam_size=beam_size, max_turns=max_turns)
        self.branch_n = branch_n
        self.beam_size = beam_size
        self.beam_tau = beam_tau
        self.beam_epsilon_random = beam_epsilon_random
        self.seed = seed

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
        """Run gold-free tree search and return one final SQL for eval scoring."""

        turns: list[AgentTurn] = []
        all_nodes: list[TreeEvalNode] = []
        selected_parents = [TreeEvalParentState(state_id=_root_state_id(sample.rollout_id), turn_index=0)]
        rng = random.Random(_seed_for_rollout(self.seed, sample.rollout_id))
        num_execute_calls = 0
        num_check_calls = 0
        num_parse_errors = 0
        selected_frontier_nodes = 0
        reached_max_turns = False

        for depth in range(self.max_turns):
            expanded_nodes: list[TreeEvalNode] = []
            for parent in selected_parents:
                parent_children: list[TreeEvalNode] = []
                for branch_index in range(self.branch_n):
                    node = self._generate_child(
                        sample,
                        parent=parent,
                        branch_index=branch_index,
                        sequence_index=len(all_nodes) + len(expanded_nodes) + len(parent_children),
                        model_client=model_client,
                        sqlite_path=sqlite_path,
                        turns=turns,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    parent_children.append(node)
                    num_execute_calls += int(node.sql is not None)
                    num_check_calls += int(node.check_called)
                    num_parse_errors += int(node.sql is None)

                if parent.source_node_id is not None:
                    for existing in all_nodes:
                        if existing.node_id == parent.source_node_id:
                            existing.children = parent_children
                            break
                expanded_nodes.extend(parent_children)

            all_nodes.extend(expanded_nodes)
            reached_max_turns = depth + 1 >= self.max_turns
            if reached_max_turns:
                break

            frontier = [
                node for node in expanded_nodes if not node.is_terminal_candidate and node.child_state is not None
            ]
            selected_nodes = select_tree_eval_frontier(
                frontier,
                beam_size=self.beam_size,
                rng=rng,
                tau=self.beam_tau,
                epsilon_random=self.beam_epsilon_random,
            )
            selected_frontier_nodes += len(selected_nodes)
            selected_parents = [node.child_state for node in selected_nodes if node.child_state is not None]
            if not selected_parents:
                break

        final_node = self._select_final_node(all_nodes)
        terminal_candidates = [node for node in all_nodes if node.is_terminal_candidate]
        final_sql = final_node.sql if final_node is not None else None
        if final_node is None:
            final_sql_source = "none"
        elif final_node.is_terminal_candidate:
            final_sql_source = "tree_checker_approved"
        else:
            final_sql_source = "tree_executable_fallback"

        reward = None
        if final_sql and sample.gold_sql:
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
                "inference_mode": "tree",
                "ran_out_of_turns": bool(not terminal_candidates and reached_max_turns),
                "num_execute_calls": num_execute_calls,
                "num_check_calls": num_check_calls,
                "num_parse_errors": num_parse_errors,
                "no_parseable_sql": all(node.sql is None for node in all_nodes),
                "max_turns": self.max_turns,
                "tree_branch_n": self.branch_n,
                "tree_beam_size": self.beam_size,
                "tree_beam_tau": self.beam_tau,
                "tree_beam_epsilon_random": self.beam_epsilon_random,
                "tree_seed": self.seed,
                "tree_nodes": len(all_nodes),
                "tree_slot_count": tree_eval_slot_count(
                    branch_n=self.branch_n,
                    beam_size=self.beam_size,
                    max_turns=self.max_turns,
                ),
                "tree_terminal_candidates": len(terminal_candidates),
                "tree_rule_pruned_candidates": sum(1 for node in all_nodes if node.prune_reason is not None),
                "tree_selected_frontier_nodes": selected_frontier_nodes,
                "tree_final_node_id": final_node.node_id if final_node is not None else None,
                "tree_final_turn_index": final_node.turn_index if final_node is not None else None,
                "tree_final_proxy_score": tree_eval_proxy_score(final_node) if final_node is not None else None,
            },
        )

    def _generate_child(
        self,
        sample: SqlAgentInput,
        *,
        parent: TreeEvalParentState,
        branch_index: int,
        sequence_index: int,
        model_client: ModelClient,
        sqlite_path: str | Path,
        turns: list[AgentTurn],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
    ) -> TreeEvalNode:
        agent_step = "write_query" if parent.turn_index == 0 else "rewrite_query"
        node_id = f"{parent.state_id}::child{branch_index}"
        request_turn = self._build_sql_request_turn(
            sample,
            agent_step=agent_step,
            previous_sql=parent.previous_sql,
            previous_execution=parent.previous_execution,
            feedback=parent.previous_feedback,
        )
        request_turn.metadata.update(
            {
                "tree_node_id": node_id,
                "tree_parent_state_id": parent.state_id,
                "tree_branch_index": branch_index,
            }
        )
        raw_response = model_client.generate(
            ModelRequest(
                turns=(request_turn,),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        )
        response = raw_response.content
        turns.append(request_turn)
        turns.append(
            AgentTurn(
                role="assistant",
                content=response,
                metadata=self._response_metadata(
                    raw_response,
                    agent_step=agent_step,
                    trainable=True,
                    turn_index=parent.turn_index,
                    node_id=node_id,
                    parent_state_id=parent.state_id,
                    branch_index=branch_index,
                ),
            )
        )

        candidate_sql = extract_sql_candidate(response)
        execution_ok = False
        execution_feedback = "No SQL query found. Return only one read-only SQLite SELECT query."
        checker_feedback: str | None = execution_feedback
        checker_verdict: bool | None = False
        prune_reason: str | None = None
        check_called = False

        if candidate_sql is None:
            turns.append(
                AgentTurn(
                    role="tool",
                    content=execution_feedback,
                    metadata={
                        "agent_step": "execute_query",
                        "ok": False,
                        "error": "no_sql",
                        "sql": None,
                        "tree_node_id": node_id,
                        "tree_parent_state_id": parent.state_id,
                        "tree_branch_index": branch_index,
                    },
                )
            )
        else:
            execution = self.sqlite_tool.execute(sqlite_path, candidate_sql)
            execution_ok = bool(execution.ok)
            execution_feedback = _format_execution_feedback(execution.ok, execution.rows, execution.error)
            if not execution_ok:
                prune_reason = _irrecoverable_sqlite_error_reason(execution.error, sample.schema_prompt)
            turns.append(
                AgentTurn(
                    role="tool",
                    content=execution_feedback,
                    metadata={
                        "agent_step": "execute_query",
                        "ok": execution.ok,
                        "sql": candidate_sql,
                        "error": execution.error,
                        "elapsed_seconds": execution.elapsed_seconds,
                        "safety_reason": execution.safety_reason,
                        "reward": None,
                        "tree_node_id": node_id,
                        "tree_parent_state_id": parent.state_id,
                        "tree_branch_index": branch_index,
                    },
                )
            )

            if prune_reason is None:
                check_turn = self._build_check_turn(sample, query=candidate_sql, execution=execution_feedback)
                check_turn.metadata.update(
                    {
                        "tree_node_id": node_id,
                        "tree_parent_state_id": parent.state_id,
                        "tree_branch_index": branch_index,
                    }
                )
                raw_check = model_client.generate(
                    ModelRequest(
                        turns=(check_turn,),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                )
                check_called = True
                checker_feedback = raw_check.content
                checker_verdict = _checker_verdict(checker_feedback)
                turns.append(check_turn)
                turns.append(
                    AgentTurn(
                        role="assistant",
                        content=checker_feedback,
                        metadata=self._response_metadata(
                            raw_check,
                            agent_step="check_query",
                            trainable=False,
                            turn_index=parent.turn_index,
                            node_id=node_id,
                            parent_state_id=parent.state_id,
                            branch_index=branch_index,
                            query=candidate_sql,
                            execution_ok=execution.ok,
                        ),
                    ),
                )
            else:
                checker_feedback = f"{execution_feedback}\nRule-pruned: {prune_reason}."

        node = TreeEvalNode(
            node_id=node_id,
            parent_state_id=parent.state_id,
            turn_index=parent.turn_index,
            branch_index=branch_index,
            sequence_index=sequence_index,
            response_text=response,
            sql=candidate_sql,
            execution_ok=execution_ok,
            execution_feedback=execution_feedback,
            checker_feedback=checker_feedback,
            checker_verdict=checker_verdict,
            prune_reason=prune_reason,
            check_called=check_called,
        )
        node.child_state = self._child_state_for_node(sample, node=node)
        return node

    def _child_state_for_node(self, sample: SqlAgentInput, *, node: TreeEvalNode) -> TreeEvalParentState | None:
        del sample
        if node.is_proxy_pruned_leaf or node.turn_index + 1 >= self.max_turns:
            return None
        return TreeEvalParentState(
            state_id=_child_state_id(
                root_uid=node.parent_state_id.split("::", 1)[0],
                next_turn_index=node.turn_index + 1,
                sql=node.sql,
                execution_feedback=node.execution_feedback,
                checker_feedback=node.checker_feedback,
            ),
            turn_index=node.turn_index + 1,
            previous_sql=node.sql,
            previous_execution=node.execution_feedback,
            previous_feedback=node.checker_feedback or node.execution_feedback,
            source_node_id=node.node_id,
        )

    @staticmethod
    def _response_metadata(
        response: ModelResponse,
        *,
        agent_step: str,
        trainable: bool,
        turn_index: int,
        node_id: str,
        parent_state_id: str,
        branch_index: int,
        **extra: Any,
    ) -> dict[str, Any]:
        metadata = {
            "agent_step": agent_step,
            "trainable": trainable,
            "turn_index": turn_index,
            "tree_node_id": node_id,
            "tree_parent_state_id": parent_state_id,
            "tree_branch_index": branch_index,
            "prompt_ids": response.prompt_ids,
            "response_ids": response.response_ids,
            "prompt_text": response.prompt_text,
            "response_text": response.response_text,
        }
        metadata.update(extra)
        return metadata

    @staticmethod
    def _select_final_node(nodes: list[TreeEvalNode]) -> TreeEvalNode | None:
        terminal_nodes = [node for node in nodes if node.is_terminal_candidate]
        if terminal_nodes:
            return max(terminal_nodes, key=_final_node_key)

        executable_nodes = [node for node in nodes if node.sql and node.execution_ok]
        if not executable_nodes:
            return None
        return max(executable_nodes, key=_final_node_key)


def _final_node_key(node: TreeEvalNode) -> tuple[int, float, int, int]:
    checker_rank = 2 if node.is_terminal_candidate else 1 if node.checker_verdict is not False else 0
    return (
        checker_rank,
        tree_eval_proxy_score(node),
        -node.turn_index,
        -node.sequence_index,
    )
