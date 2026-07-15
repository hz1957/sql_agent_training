import json
import sys
import types
from pathlib import Path

from sql_agent_training.agent.model_client import HuggingFaceModelClient, _resolve_adapter_base_model_path


class DummyTokenizer:
    pad_token_id = None
    eos_token = "<eos>"
    eos_token_id = 0


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
