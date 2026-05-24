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


def load_tokenizer(kind: str, model_name_or_path: str | None = None) -> TextTokenizer:
    """Load a tokenizer by kind."""

    if kind == "whitespace":
        return WhitespaceTokenizer()
    if kind == "hf":
        if not model_name_or_path:
            raise ValueError("model_name_or_path is required for hf tokenizer")
        return HuggingFaceTokenizer(model_name_or_path)
    raise ValueError(f"Unknown tokenizer kind: {kind}")


def trajectory_to_tokenized(trajectory: AgentTrajectory, tokenizer: TextTokenizer) -> TokenizedTrajectory:
    """Convert an AgentTrajectory to the internal tokenized GRPO contract.

    The first user turn is treated as prompt. Assistant turns receive mask 1.
    Tool/environment turns receive mask 0.
    """

    if not trajectory.turns:
        raise ValueError("trajectory must contain at least one turn")

    prompt_turns = []
    response_turns = []
    seen_response = False
    for turn in trajectory.turns:
        if not seen_response and turn.role == "user":
            prompt_turns.append(turn)
        else:
            seen_response = True
            response_turns.append(turn)

    prompt_text = "\n".join(f"{turn.role}: {turn.content}" for turn in prompt_turns)
    prompt_ids = tokenizer.encode(prompt_text)
    response_ids: list[int] = []
    response_mask: list[int] = []
    for turn in response_turns:
        ids = tokenizer.encode(f"{turn.role}: {turn.content}")
        response_ids.extend(ids)
        response_mask.extend([1 if turn.role == "assistant" else 0] * len(ids))

    return TokenizedTrajectory(
        uid=trajectory.uid,
        rollout_id=trajectory.rollout_id,
        prompt_ids=prompt_ids,
        response_ids=response_ids or [0],
        response_mask=response_mask or [0],
        reward=float(trajectory.reward or 0.0),
        metadata=trajectory.metadata,
    )
