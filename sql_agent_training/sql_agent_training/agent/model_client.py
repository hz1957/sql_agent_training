"""Model client abstraction for SQL agent rollouts."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sql_agent_training.agent.trace_format import AgentTurn


@dataclass(frozen=True)
class ModelRequest:
    """A request sent to the policy model."""

    turns: Sequence[AgentTurn]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


@dataclass(frozen=True)
class ModelResponse:
    """A model response."""

    content: str
    prompt_ids: list[int] | None = None
    response_ids: list[int] | None = None
    prompt_text: str | None = None
    response_text: str | None = None


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


def format_openai_messages(turns: Sequence[AgentTurn]) -> list[dict[str, str]]:
    """Convert agent turns to messages accepted by OpenAI-compatible chat APIs."""

    messages: list[dict[str, str]] = []
    for turn in turns:
        if turn.role == "tool":
            messages.append({"role": "user", "content": f"Tool observation:\n{turn.content}"})
        elif turn.role in {"system", "user", "assistant"}:
            messages.append({"role": turn.role, "content": turn.content})
        else:
            messages.append({"role": "user", "content": f"{turn.role}: {turn.content}"})
    return messages


def format_hf_prompt(tokenizer: Any, turns: Sequence[AgentTurn]) -> str:
    """Format agent turns with a Hugging Face chat template when available."""

    messages = format_openai_messages(turns)

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    lines = [f"{message['role']}: {message['content']}" for message in messages]
    lines.append("assistant:")
    return "\n".join(lines)


def _resolve_device(torch: Any, device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _post_json(
    url: str, payload: dict[str, Any], *, api_key: str | None, timeout_seconds: float
) -> dict[str, Any] | str:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body}") from exc
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _adapter_config_path(path: str | Path) -> Path:
    return Path(path) / "adapter_config.json"


def _resolve_adapter_base_model_path(
    adapter_config_path: str | Path,
    *,
    fallback_base_model_name_or_path: str | None = None,
) -> str:
    config = json.loads(Path(adapter_config_path).read_text(encoding="utf-8"))
    base_model_name_or_path = str(config["base_model_name_or_path"])
    if fallback_base_model_name_or_path and not Path(base_model_name_or_path).exists():
        return fallback_base_model_name_or_path
    return base_model_name_or_path


def _resolve_torch_dtype(torch: Any, value: str | None) -> Any | None:
    if value is None:
        return None
    requested = str(value).lower()
    if requested in {"auto", "none"}:
        return requested if requested == "auto" else None
    if requested in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if requested in {"float16", "fp16", "half"}:
        return torch.float16
    if requested in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unknown torch dtype: {value}")


def _load_causal_lm(
    AutoModelForCausalLM: Any,
    model_name_or_path: str,
    *,
    trust_remote_code: bool,
    torch_dtype: Any | None,
    fallback_base_model_name_or_path: str | None = None,
) -> Any:
    load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    adapter_config = _adapter_config_path(model_name_or_path)
    if not adapter_config.exists():
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the train extra with PEFT to evaluate LoRA adapter checkpoints.") from exc

    base_model_path = _resolve_adapter_base_model_path(
        adapter_config,
        fallback_base_model_name_or_path=fallback_base_model_name_or_path,
    )
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
    return PeftModel.from_pretrained(base_model, model_name_or_path)


class HuggingFaceInMemoryModelClient:
    """Hugging Face client backed by an already-loaded policy model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install torch to use HuggingFaceInMemoryModelClient.") from exc

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.default_max_new_tokens = max_new_tokens
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.default_top_k = top_k
        _ensure_pad_token(self.tokenizer)

    def _format_prompt(self, turns: Sequence[AgentTurn]) -> str:
        return format_hf_prompt(self.tokenizer, turns)

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate the next assistant message with an in-memory HF model."""

        prompt = self._format_prompt(request.turns)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_ids = list(inputs["input_ids"][0].tolist())
        max_new_tokens = request.max_tokens or self.default_max_new_tokens
        temperature = self.default_temperature if request.temperature is None else request.temperature
        top_p = self.default_top_p if request.top_p is None else request.top_p
        top_k = self.default_top_k if request.top_k is None else request.top_k
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
            if top_p is not None:
                generate_kwargs["top_p"] = top_p
            if top_k is not None:
                generate_kwargs["top_k"] = top_k
        else:
            generate_kwargs["do_sample"] = False

        self.model.eval()
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        response_ids = list(output_ids[0][len(prompt_ids) :].tolist())
        content = str(self.tokenizer.decode(response_ids, skip_special_tokens=True)).strip()
        return ModelResponse(
            content=content,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            prompt_text=prompt,
            response_text=content,
        )


class HuggingFaceModelClient(HuggingFaceInMemoryModelClient):
    """Local Hugging Face causal LM client for SQL agent rollouts."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        tokenizer_name_or_path: str | None = None,
        base_model_name_or_path: str | None = None,
        device: str = "auto",
        trust_remote_code: bool = True,
        torch_dtype: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install torch and transformers to use HuggingFaceModelClient.") from exc

        self.torch = torch
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        resolved_device = _resolve_device(torch, device)
        tokenizer: Any = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        model: Any = _load_causal_lm(
            AutoModelForCausalLM,
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=_resolve_torch_dtype(torch, torch_dtype),
            fallback_base_model_name_or_path=base_model_name_or_path,
        ).to(resolved_device)
        super().__init__(
            model,
            tokenizer,
            device=resolved_device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )


class VllmOpenAIModelClient:
    """OpenAI-compatible vLLM completions client for rollout generation."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        tokenizer: Any | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.default_max_new_tokens = max_new_tokens
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.default_top_k = top_k
        if self.tokenizer is not None:
            _ensure_pad_token(self.tokenizer)

    def _format_prompt(self, turns: Sequence[AgentTurn]) -> str:
        if self.tokenizer is not None:
            return format_hf_prompt(self.tokenizer, turns)
        lines = [f"{turn.role}: {turn.content}" for turn in turns]
        lines.append("assistant:")
        return "\n".join(lines)

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate the next assistant message through vLLM's OpenAI-compatible API."""

        prompt = self._format_prompt(request.turns)
        max_new_tokens = request.max_tokens or self.default_max_new_tokens
        temperature = self.default_temperature if request.temperature is None else request.temperature
        top_p = self.default_top_p if request.top_p is None else request.top_p
        top_k = self.default_top_k if request.top_k is None else request.top_k
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k

        response = _post_json(
            _join_url(self.base_url, "completions"),
            payload,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected vLLM completions response: {response!r}")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"vLLM completions response did not contain choices: {response!r}")
        text = str(choices[0].get("text", "")).strip()
        prompt_ids = None
        response_ids = None
        if self.tokenizer is not None:
            prompt_ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
            response_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        return ModelResponse(
            content=text,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            prompt_text=prompt,
            response_text=text,
        )

    def load_lora_adapter(self, *, lora_name: str, lora_path: str | Path, load_inplace: bool = True) -> None:
        """Load or replace a LoRA adapter in a trusted vLLM server."""

        payload: dict[str, Any] = {
            "lora_name": lora_name,
            "lora_path": str(lora_path),
        }
        if load_inplace:
            payload["load_inplace"] = True
        _post_json(
            _join_url(self.base_url, "load_lora_adapter"),
            payload,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )


class OpenAIChatModelClient:
    """Client for tokenizer-free, OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.default_max_new_tokens = max_new_tokens
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "http 408",
                "http 409",
                "http 425",
                "http 429",
                "http 5",
                "timed out",
                "temporarily unavailable",
                "connection reset",
                "remote end closed connection",
            )
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any] | str:
        for attempt in range(self.max_retries + 1):
            try:
                return _post_json(
                    _join_url(self.base_url, "chat/completions"),
                    payload,
                    api_key=self.api_key,
                    timeout_seconds=self.timeout_seconds,
                )
            except (RuntimeError, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt >= self.max_retries or not self._is_retryable(exc):
                    raise
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise RuntimeError("Chat request retry loop ended unexpectedly")

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate the next assistant message through a hosted chat API."""

        messages = format_openai_messages(request.turns)
        max_new_tokens = request.max_tokens or self.default_max_new_tokens
        temperature = self.default_temperature if request.temperature is None else request.temperature
        top_p = self.default_top_p if request.top_p is None else request.top_p
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if top_p is not None:
            payload["top_p"] = top_p

        response = self._post(payload)
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected chat-completions response: {response!r}")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Chat-completions response did not contain choices: {response!r}")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise RuntimeError(f"Chat-completions choice did not contain a message: {choice!r}")
        content = message.get("content")
        if content is None:
            content = message.get("reasoning_content", "")
        text = str(content).strip()
        return ModelResponse(
            content=text,
            prompt_text=json.dumps(messages, ensure_ascii=False),
            response_text=text,
        )
