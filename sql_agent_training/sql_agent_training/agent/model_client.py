"""Model client abstraction for SQL agent rollouts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

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
    """Protocol implemented by local or hosted model clients."""

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


class HuggingFaceModelClient:
    """Local Hugging Face causal LM client for SQL agent rollouts."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "auto",
        trust_remote_code: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install torch and transformers to use HuggingFaceModelClient.") from exc

        self.torch = torch
        self.model_name_or_path = model_name_or_path
        self.device = self._resolve_device(device)
        self.default_max_new_tokens = max_new_tokens
        self.default_temperature = temperature
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if self.torch.cuda.is_available() else "cpu"

    def _format_prompt(self, turns: Sequence[AgentTurn]) -> str:
        messages = []
        for turn in turns:
            if turn.role == "tool":
                messages.append({"role": "user", "content": f"Tool observation:\n{turn.content}"})
            elif turn.role in {"system", "user", "assistant"}:
                messages.append({"role": turn.role, "content": turn.content})
            else:
                messages.append({"role": "user", "content": f"{turn.role}: {turn.content}"})

        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            return str(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        lines = [f"{message['role']}: {message['content']}" for message in messages]
        lines.append("assistant:")
        return "\n".join(lines)

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate the next assistant message with the local HF model."""

        prompt = self._format_prompt(request.turns)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_ids = list(inputs["input_ids"][0].tolist())
        max_new_tokens = request.max_tokens or self.default_max_new_tokens
        temperature = self.default_temperature if request.temperature is None else request.temperature
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        response_ids = list(output_ids[0][len(prompt_ids) :].tolist())
        content = str(self.tokenizer.decode(response_ids, skip_special_tokens=True)).strip()
        return ModelResponse(content=content, prompt_ids=prompt_ids, response_ids=response_ids)
