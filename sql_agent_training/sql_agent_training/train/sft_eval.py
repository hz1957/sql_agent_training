"""SFT generation and evaluation utilities."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
        max_input_tokens: int = 1024,
        max_new_tokens: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the train extra to run generation: pip install -e '.[train]'") from exc

        self.torch = torch
        self.tokenizer = HuggingFaceTokenizer(tokenizer_name_or_path or model_name_or_path).tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
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
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


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
    return any((root / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))


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
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint to evaluate. Defaults to output.checkpoint_dir.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to checkpoint tokenizer or model.path.")
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
        generator = TransformersSqlGenerator(
            model_path,
            tokenizer_name_or_path=tokenizer_path,
            max_input_tokens=int(training.get("max_prompt_length", 1024)),
            max_new_tokens=int(training.get("max_response_length", 256)),
        )

    # Default: write eval results into <checkpoint>/eval/ so each run is self-contained.
    # Explicit --output or the legacy config key override this.
    _eval_base = Path(model_path) / "eval" if model_path else Path("artifacts/eval/sft")
    output_path = Path(
        args.output
        or config.get("output", {}).get("predictions_jsonl")
        or _eval_base / "predictions.jsonl"
    )

    rows = generate_predictions(examples, tables_index, generator, dry_run_gold=args.dry_run_gold)
    write_predictions_jsonl(rows, output_path)
    metrics = evaluate_predictions(rows, data_dir)
    print(json.dumps({"predictions": str(output_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
