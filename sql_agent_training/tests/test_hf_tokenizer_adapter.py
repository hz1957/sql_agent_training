import sys
import types

from sql_agent_training.agent.tokenization import HuggingFaceTokenizer, load_tokenizer


class DummyTokenizer:
    pad_token_id = None
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [len(part) for part in text.split()]


def test_load_whitespace_tokenizer() -> None:
    tokenizer = load_tokenizer("whitespace")

    assert tokenizer.encode("hello world") == [1, 2]


def test_huggingface_tokenizer_adapter_with_stub(monkeypatch) -> None:
    transformers = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name_or_path, trust_remote_code=True):
            assert model_name_or_path == "dummy-model"
            assert trust_remote_code is True
            return DummyTokenizer()

    transformers.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    tokenizer = HuggingFaceTokenizer("dummy-model")

    assert tokenizer.encode("aa bbb") == [2, 3]
    assert tokenizer.encode("") == [99]
    assert tokenizer.pad_token_id == 99
