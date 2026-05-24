"""Model client abstraction for SQL agent rollouts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sql_agent_training.agent.trace_format import AgentTurn


@dataclass(frozen=True)
class ModelRequest:
    """A request sent to the policy model."""

    turns: Sequence[AgentTurn]
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class ModelResponse:
    """A model response."""

    content: str
    prompt_ids: list[int] | None = None
    response_ids: list[int] | None = None


class ModelClient(Protocol):
    """Protocol implemented by local, vLLM, SGLang, or VERL model clients."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate the next assistant message."""


class ScriptedModelClient:
    """Deterministic model client for local tests and dry runs."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Return the next scripted response."""

        del request
        if self.calls >= len(self._responses):
            return ModelResponse(content="")
        response = self._responses[self.calls]
        self.calls += 1
        return ModelResponse(content=response)
