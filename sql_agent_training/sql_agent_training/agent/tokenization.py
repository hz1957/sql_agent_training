"""Tokenization contract for agent trajectories."""

from __future__ import annotations
from typing import Protocol

from sql_agent_training.agent.trace_format import AgentTrajectory, TokenizedTrajectory


class TextTokenizer(Protocol):
    """Minimal tokenizer protocol used before binding to a HF tokenizer."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""

    @property
    def pad_token_id(self) -> int:
        """Token id used for padding."""

    @property
    def eos_token_id(self) -> int | None:
        """Optional token id used to terminate generation."""


class WhitespaceTokenizer:
    """Deterministic toy tokenizer for tests and local dry runs."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        """Encode by whitespace with stable ids."""

        ids: list[int] = []
        for token in text.split():
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab) + 1
            ids.append(self._vocab[token])
        return ids or [0]

    @property
    def pad_token_id(self) -> int:
        """Return toy pad token id."""

        return 0

    @property
    def eos_token_id(self) -> int | None:
        """Return no EOS token for the toy tokenizer."""

        return None


class HuggingFaceTokenizer:
    """Adapter around a Hugging Face tokenizer."""

    def __init__(self, model_name_or_path: str, *, trust_remote_code: bool = True) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install transformers to use HuggingFaceTokenizer.") from exc

        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    def encode(self, text: str) -> list[int]:
        """Encode text without adding special tokens."""

        return list(self.tokenizer.encode(text, add_special_tokens=False)) or [self.pad_token_id]

    @property
    def pad_token_id(self) -> int:
        """Return a usable pad token id."""

        token_id = self.tokenizer.pad_token_id
        if token_id is not None:
            return int(token_id)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            return int(eos_token_id)
        return 0

    @property
    def eos_token_id(self) -> int | None:
        """Return the Hugging Face EOS token id if available."""

        token_id = self.tokenizer.eos_token_id
        return int(token_id) if token_id is not None else None


class ExistingHuggingFaceTokenizer:
    """Adapter around an already-loaded Hugging Face tokenizer."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        """Encode text without adding special tokens."""

        return list(self.tokenizer.encode(text, add_special_tokens=False)) or [self.pad_token_id]

    @property
    def pad_token_id(self) -> int:
        """Return a usable pad token id."""

        token_id = self.tokenizer.pad_token_id
        if token_id is not None:
            return int(token_id)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            return int(eos_token_id)
        return 0

    @property
    def eos_token_id(self) -> int | None:
        """Return the Hugging Face EOS token id if available."""

        token_id = self.tokenizer.eos_token_id
        return int(token_id) if token_id is not None else None


def load_tokenizer(kind: str, model_name_or_path: str | None = None) -> TextTokenizer:
    """Load a tokenizer by kind."""

    if kind == "whitespace":
        return WhitespaceTokenizer()
    if kind == "hf":
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required for hf tokenizer")
        return HuggingFaceTokenizer(model_name_or_path)
    raise ValueError(f"Unknown tokenizer kind: {kind}")


def _metadata_token_ids(value: object) -> list[int] | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, int) for item in value):
        return None
    return list(value)


def _metadata_reward(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _transition_rewards(
    *,
    final_reward: float,
    own_rewards: list[float],
    reward_mode: str,
    reward_gamma: float,
) -> list[float]:
    """Assign trainable action rewards while preserving final execution reward semantics."""

    if reward_mode == "final":
        return [final_reward] * len(own_rewards)
    if reward_mode != "discounted_final":
        raise ValueError(f"Unknown transition reward mode: {reward_mode}")
    if not 0.0 <= reward_gamma <= 1.0:
        raise ValueError("transition reward gamma must be between 0 and 1")

    last_index = len(own_rewards) - 1
    rewards: list[float] = []
    for index, own_reward in enumerate(own_rewards):
        discounted_final = final_reward * (reward_gamma ** (last_index - index))
        rewards.append(max(own_reward, discounted_final))
    return rewards


def trajectory_to_tokenized_transitions(
    trajectory: AgentTrajectory,
    tokenizer: TextTokenizer,
    *,
    reward_mode: str = "final",
    reward_gamma: float = 0.4,
) -> list[TokenizedTrajectory]:
    """Convert each assistant SQL action into an independent GRPO sample.

    Each transition uses the compact user prompt immediately preceding the
    assistant action as the prompt. Only assistant tokens are trainable; tool
    observations are used for credit assignment metadata rather than being
    concatenated into the training response.
    """

    if not trajectory.turns:
        raise ValueError("trajectory must contain at least one turn")

    transition_rows = []
    current_prompt: str | None = None
    action_index = 0
    for index, turn in enumerate(trajectory.turns):
        if turn.role == "user":
            current_prompt = f"{turn.role}: {turn.content}"
            continue
        if turn.role != "assistant":
            continue
        if turn.metadata.get("trainable") is False or turn.metadata.get("agent_step") == "check_query":
            continue
        if current_prompt is None:
            raise ValueError("assistant turn must have a preceding user prompt")

        tool_turn = trajectory.turns[index + 1] if index + 1 < len(trajectory.turns) else None
        tool_metadata = tool_turn.metadata if tool_turn is not None and tool_turn.role == "tool" else {}
        prompt_ids = _metadata_token_ids(turn.metadata.get("prompt_ids")) or tokenizer.encode(current_prompt)
        response_text = str(turn.metadata.get("response_text") or f"{turn.role}: {turn.content}")
        response_ids = _metadata_token_ids(turn.metadata.get("response_ids")) or tokenizer.encode(response_text)
        turn_index = int(turn.metadata.get("turn_index", action_index))
        prompt_text = str(turn.metadata.get("prompt_text") or current_prompt)
        own_reward = _metadata_reward(tool_metadata.get("reward"))
        transition_rows.append(
            {
                "turn_index": turn_index,
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "own_reward": own_reward,
                "tool_metadata": tool_metadata,
                "turn_metadata": turn.metadata,
            }
        )
        action_index += 1

    if not transition_rows:
        raise ValueError("trajectory must contain at least one assistant action")

    final_reward = float(trajectory.reward or 0.0)
    assigned_rewards = _transition_rewards(
        final_reward=final_reward,
        own_rewards=[row["own_reward"] for row in transition_rows],
        reward_mode=reward_mode,
        reward_gamma=reward_gamma,
    )

    transitions: list[TokenizedTrajectory] = []
    for row, assigned_reward in zip(transition_rows, assigned_rewards, strict=True):
        turn_metadata = row["turn_metadata"]
        tool_metadata = row["tool_metadata"]
        transitions.append(
            TokenizedTrajectory(
                uid=trajectory.uid,
                rollout_id=f"{trajectory.rollout_id}:turn{row['turn_index']}",
                prompt_ids=row["prompt_ids"],
                response_ids=row["response_ids"],
                response_mask=[1] * len(row["response_ids"]),
                reward=assigned_reward,
                prompt_text=row["prompt_text"],
                response_text=row["response_text"],
                group_id=trajectory.uid,
                metadata={
                    **trajectory.metadata,
                    "parent_rollout_id": trajectory.rollout_id,
                    "agent_step": turn_metadata.get("agent_step", "write_query"),
                    "turn_index": row["turn_index"],
                    "final_sql": trajectory.final_sql,
                    "final_sql_source": trajectory.final_sql_source,
                    "used_model_token_ids": bool(
                        _metadata_token_ids(turn_metadata.get("prompt_ids"))
                        and _metadata_token_ids(turn_metadata.get("response_ids"))
                    ),
                    "tool_ok": bool(tool_metadata.get("ok", False)),
                    "tool_error": tool_metadata.get("error"),
                    "tool_reward": tool_metadata.get("reward"),
                    "own_sql_reward": row["own_reward"],
                    "trajectory_reward": final_reward,
                    "transition_reward_mode": reward_mode,
                    "transition_reward_gamma": reward_gamma,
                },
            )
        )
    return transitions
