import json
import sys
import types
from pathlib import Path

import pytest

from sql_agent_training.agent.model_client import (
    HuggingFaceModelClient,
    ModelRequest,
    OpenAIChatModelClient,
    VllmOpenAIModelClient,
    _resolve_adapter_base_model_path,
)
from sql_agent_training.agent.trace_format import AgentTurn


class DummyTokenizer:
    pad_token_id = None
    eos_token = "<eos>"
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(text)]


class DummyModel:
    def __init__(self) -> None:
        self.device = None

    def to(self, device):
        self.device = device
        return self


def test_huggingface_model_client_loads_lora_adapter_with_fallback_base(
    monkeypatch,
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    fallback_base = tmp_path / "base"
    fallback_base.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(tmp_path / "missing_base")}),
        encoding="utf-8",
    )

    calls = []
    transformers = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name_or_path, trust_remote_code=True):
            calls.append(("tokenizer", model_name_or_path, trust_remote_code))
            return DummyTokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_name_or_path, **kwargs):
            calls.append(("base_model", model_name_or_path, kwargs))
            return DummyModel()

    class PeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path):
            calls.append(("adapter", adapter_path, model))
            return model

    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=PeftModel))

    client = HuggingFaceModelClient(
        str(adapter_dir),
        tokenizer_name_or_path=str(adapter_dir),
        base_model_name_or_path=str(fallback_base),
        device="cpu",
        torch_dtype="auto",
    )

    assert client.model.device == "cpu"
    assert calls[0] == ("tokenizer", str(adapter_dir), True)
    assert calls[1] == (
        "base_model",
        str(fallback_base),
        {"trust_remote_code": True, "torch_dtype": "auto"},
    )
    assert calls[2][0:2] == ("adapter", str(adapter_dir))


def test_resolve_adapter_base_model_path_uses_existing_recorded_path(tmp_path: Path) -> None:
    base_model = tmp_path / "base"
    base_model.mkdir()
    adapter_config = tmp_path / "adapter_config.json"
    adapter_config.write_text(json.dumps({"base_model_name_or_path": str(base_model)}), encoding="utf-8")

    assert _resolve_adapter_base_model_path(
        adapter_config,
        fallback_base_model_name_or_path=str(tmp_path / "fallback"),
    ) == str(base_model)


def test_vllm_openai_client_generates_and_loads_lora(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class FakeResponse:
        def __init__(self, body: str) -> None:
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def read(self) -> bytes:
            return self.body.encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append(
            {
                "url": request.full_url,
                "payload": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        if request.full_url.endswith("/v1/completions"):
            return FakeResponse(json.dumps({"choices": [{"text": "SELECT 1"}]}))
        return FakeResponse("Success")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = VllmOpenAIModelClient(
        base_url="http://127.0.0.1:8000/v1",
        model_name="current_policy",
        tokenizer=DummyTokenizer(),
        timeout_seconds=12.0,
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
    )

    response = client.generate(ModelRequest(turns=[AgentTurn(role="user", content="Write SQL")]))
    client.load_lora_adapter(lora_name="current_policy", lora_path=tmp_path / "adapter", load_inplace=True)

    assert response.content == "SELECT 1"
    assert response.prompt_ids == [len("user: Write SQL\nassistant:")]
    assert response.response_ids == [len("SELECT 1")]
    assert calls[0]["url"] == "http://127.0.0.1:8000/v1/completions"
    assert calls[0]["payload"]["model"] == "current_policy"
    assert calls[0]["payload"]["temperature"] == 0.7
    assert calls[0]["payload"]["top_k"] == 40
    assert calls[1]["url"] == "http://127.0.0.1:8000/v1/load_lora_adapter"
    assert calls[1]["payload"] == {
        "lora_name": "current_policy",
        "lora_path": str(tmp_path / "adapter"),
        "load_inplace": True,
    }


def test_openai_chat_client_generates_without_tokenizer(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "SELECT 1"}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append(
            {
                "url": request.full_url,
                "payload": json.loads(request.data.decode("utf-8")),
                "authorization": request.get_header("Authorization"),
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAIChatModelClient(
        base_url="https://example.test/v1",
        model_name="deepseek-chat",
        api_key="secret",
        timeout_seconds=15.0,
        max_new_tokens=128,
    )

    response = client.generate(
        ModelRequest(
            turns=[
                AgentTurn(role="user", content="Write SQL"),
                AgentTurn(role="tool", content="syntax error"),
            ]
        )
    )

    assert response.content == "SELECT 1"
    assert response.prompt_ids is None
    assert calls[0]["url"] == "https://example.test/v1/chat/completions"
    assert calls[0]["authorization"] == "Bearer secret"
    assert calls[0]["timeout"] == 15.0
    assert calls[0]["payload"]["model"] == "deepseek-chat"
    assert calls[0]["payload"]["messages"] == [
        {"role": "user", "content": "Write SQL"},
        {"role": "user", "content": "Tool observation:\nsyntax error"},
    ]


def test_openai_chat_client_rejects_non_header_api_key() -> None:
    client = OpenAIChatModelClient(
        base_url="https://example.test/v1",
        model_name="deepseek-chat",
        api_key="你的key",
        max_retries=0,
    )

    with pytest.raises(ValueError, match="API key contains"):
        client.generate(ModelRequest(turns=[AgentTurn(role="user", content="Check SQL")]))
