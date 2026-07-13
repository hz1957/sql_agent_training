"""SFT generation and evaluation utilities."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from sql_agent_training.agent.tokenization import HuggingFaceTokenizer
from sql_agent_training.data.schema import load_tables_json
from sql_agent_training.data.sft_formatter import format_sft_record
from sql_agent_training.data.spider_dataset import SpiderExample, expected_sqlite_path, load_spider_file
from sql_agent_training.env.sqlite_tool import SQLiteTool
from sql_agent_training.reward.spider_reward import spider_execution_reward


class SqlGenerator(Protocol):
    """Protocol for SQL generation backends."""

    def generate_sql(self, prompt: str) -> str:
        """Generate SQL from a prompt."""


class TransformersSqlGenerator:
    """Transformers-backed SQL generator."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        tokenizer_name_or_path: str | None = None,
        base_model_name_or_path: str | None = None,
        max_input_tokens: int = 1024,
        max_new_tokens: int = 256,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the train extra to run generation: pip install -e '.[train]'") from exc

        self.torch = torch
        self.tokenizer = HuggingFaceTokenizer(tokenizer_name_or_path or model_name_or_path).tokenizer
        self.model = _load_model(model_name_or_path, fallback_base_model_name_or_path=base_model_name_or_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    def generate_sql(self, prompt: str) -> str:
        """Generate SQL from a prompt."""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        if self.torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}

        # Collect all token IDs that should terminate generation.
        # Qwen2.5 uses <|im_end|> (151645) as the EOS during SFT but the
        # checkpoint's generation_config may store a different default; always
        # set both eos_token_id and pad_token_id explicitly so generation stops
        # as soon as the model emits <|im_end|> or <|endoftext|>.
        stop_ids: list[int] = []
        for token_id in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
            if token_id is not None and token_id not in stop_ids:
                stop_ids.append(token_id)
        # Also stop on <|im_end|> if the tokenizer knows it (Qwen2.5 family).
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != self.tokenizer.unk_token_id and im_end_id not in stop_ids:
            stop_ids.append(im_end_id)
        eos_token_id: int | list[int] | None = stop_ids if len(stop_ids) > 1 else (stop_ids[0] if stop_ids else None)
        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return normalize_generated_sql(self.tokenizer.decode(generated, skip_special_tokens=True))


def normalize_generated_sql(text: str) -> str:
    """Return a single SQL statement from generated text."""

    stripped = text.strip()
    fence = re.fullmatch(r"```(?:sql)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    for prefix in ("FINAL:", "Final:", "final:", "SQL:", "Sql:", "sql:"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
            break
    match = re.search(r"\b(select|with)\b", stripped, flags=re.IGNORECASE)
    if match:
        stripped = stripped[match.start() :].strip()
    if "```" in stripped:
        stripped = stripped.split("```", 1)[0].strip()
    if ";" in stripped:
        stripped = stripped.split(";", 1)[0].strip()
    return stripped


def _adapter_config_path(path: str | Path) -> Path:
    return Path(path) / "adapter_config.json"


def _has_adapter_files(path: str | Path) -> bool:
    return _adapter_config_path(path).exists()


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


def _load_model(model_path: str | Path, *, fallback_base_model_name_or_path: str | None = None) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the train extra to run generation: pip install -e '.[train]'") from exc

    adapter_config = _adapter_config_path(model_path)
    if adapter_config.exists():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Install the train extra with PEFT to evaluate LoRA checkpoints.") from exc

        base_model_path = _resolve_adapter_base_model_path(
            adapter_config,
            fallback_base_model_name_or_path=fallback_base_model_name_or_path,
        )
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype="auto")
        return PeftModel.from_pretrained(base_model, model_path)

    return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto")


@dataclass(frozen=True)
class PredictionResult:
    """One generated prediction row."""

    uid: str
    db_id: str
    question: str
    gold_sql: str
    prediction: str
    reward: float | None = None


def write_predictions_jsonl(rows: list[PredictionResult], output_path: str | Path) -> int:
    """Write predictions to JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")
    return len(rows)


def evaluate_predictions(rows: list[PredictionResult], data_dir: str | Path) -> dict[str, float | int]:
    """Evaluate prediction rows with Spider execution reward."""

    scored = []
    executable = 0
    sql_tool = SQLiteTool()
    for row in rows:
        db_path = expected_sqlite_path(data_dir, row.db_id)
        if sql_tool.execute(db_path, row.prediction).ok:
            executable += 1
        scored.append(spider_execution_reward(row.prediction, row.gold_sql, db_path))
    total = len(scored)
    return {
        "total": total,
        "executable_rate": executable / total if total else 0.0,
        "execution_accuracy": sum(scored) / total if total else 0.0,
    }


def _has_model_files(path: str | Path) -> bool:
    root = Path(path)
    return any((root / name).exists() for name in ("model.safetensors", "pytorch_model.bin")) or _has_adapter_files(
        root
    )


def _has_tokenizer_files(path: str | Path) -> bool:
    root = Path(path)
    return any((root / name).exists() for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"))


def _checkpoint_sort_key(path: Path) -> tuple[int, int | str]:
    checkpoint_match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if checkpoint_match:
        return (1, int(checkpoint_match.group(1)))
    timestamp_match = re.fullmatch(r"\d{8}_\d{6}(?:_\d{2})?", path.name)
    if timestamp_match:
        return (2, path.name)
    return (0, path.name)


def _latest_model_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    candidates = [child for child in path.iterdir() if child.is_dir() and _has_model_files(child)]
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_sort_key)


def _resolve_model_and_tokenizer(
    config: dict,
    *,
    checkpoint: str | None,
    tokenizer_path: str | None,
) -> tuple[str, str]:
    model_config = config.get("model", {})
    output_config = config.get("output", {})
    tokenizer_config = config.get("tokenizer", {})

    model_path = Path(checkpoint or output_config.get("checkpoint_dir") or model_config["path"])
    if not _has_model_files(model_path):
        nested_model_dir = _latest_model_dir(model_path)
        if nested_model_dir is not None:
            model_path = nested_model_dir

    resolved_tokenizer = tokenizer_path or model_config.get("tokenizer_path") or tokenizer_config.get("path")
    if not resolved_tokenizer:
        resolved_tokenizer = str(model_path) if _has_tokenizer_files(model_path) else str(model_config["path"])

    return str(model_path), str(resolved_tokenizer)


def generate_predictions(
    examples: list[SpiderExample],
    tables_index: dict,
    generator: SqlGenerator | None = None,
    *,
    dry_run_gold: bool = False,
) -> list[PredictionResult]:
    """Generate SQL predictions for examples."""

    if generator is None and not dry_run_gold:
        raise ValueError("generator is required unless dry_run_gold is enabled")

    rows: list[PredictionResult] = []
    for example in examples:
        record = format_sft_record(example, tables_index)
        if dry_run_gold:
            prediction = example.gold_sql
        else:
            assert generator is not None
            prediction = generator.generate_sql(record["prompt"])
        rows.append(
            PredictionResult(
                uid=example.uid,
                db_id=example.db_id,
                question=example.question,
                gold_sql=example.gold_sql,
                prediction=prediction,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and evaluate SFT SQL predictions.")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument(
        "--checkpoint", default=None, help="Model checkpoint to evaluate. Defaults to output.checkpoint_dir."
    )
    parser.add_argument(
        "--tokenizer", default=None, help="Tokenizer path. Defaults to checkpoint tokenizer or model.path."
    )
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples for local smoke tests.")
    parser.add_argument("--dry-run-gold", action="store_true", help="Emit gold SQL as predictions for plumbing tests.")
    parser.add_argument("--output", default=None, help="Override predictions JSONL path.")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_dir = Path(config["data"]["data_dir"])
    split_file = config["data"]["validation_file"] if args.split == "validation" else config["data"]["train_file"]
    examples = load_spider_file(data_dir / split_file)
    if args.limit is not None:
        examples = examples[: args.limit]
    tables_index = load_tables_json(data_dir / "tables.json")

    if args.dry_run_gold:
        generator: SqlGenerator | None = None
        model_path = None
    else:
        training = config.get("training", {})
        model_path, tokenizer_path = _resolve_model_and_tokenizer(
            config,
            checkpoint=args.checkpoint,
            tokenizer_path=args.tokenizer,
        )
        base_model_name_or_path = config.get("model", {}).get("path")
        generator = TransformersSqlGenerator(
            model_path,
            tokenizer_name_or_path=tokenizer_path,
            base_model_name_or_path=str(base_model_name_or_path) if base_model_name_or_path else None,
            max_input_tokens=int(training.get("max_prompt_length", 1024)),
            max_new_tokens=int(training.get("max_response_length", 256)),
        )

    # Default: write eval results into <checkpoint>/eval/ so each run is self-contained.
    # Explicit --output or the legacy config key override this.
    _eval_base = Path(model_path) / "eval" if model_path else Path("artifacts/eval/sft")
    output_path = Path(
        args.output or config.get("output", {}).get("predictions_jsonl") or _eval_base / "predictions.jsonl"
    )

    rows = generate_predictions(examples, tables_index, generator, dry_run_gold=args.dry_run_gold)
    write_predictions_jsonl(rows, output_path)
    metrics = evaluate_predictions(rows, data_dir)
    print(json.dumps({"predictions": str(output_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
