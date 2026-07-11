"""SFT entrypoint for formatting Spider records and running Transformers training."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import yaml

from sql_agent_training.agent.tokenization import load_tokenizer
from sql_agent_training.data.schema import load_tables_json
from sql_agent_training.data.sft_formatter import write_sft_jsonl
from sql_agent_training.data.spider_dataset import load_spider_file
from sql_agent_training.train.sft_dataset import (
    SftDataCollator,
    SftTorchDataset,
    load_sft_jsonl,
    tokenize_sft_records,
)

_MODEL_WEIGHT_PATTERNS = (
    "model.safetensors",
    "pytorch_model.bin",
    "model-*.safetensors",
    "pytorch_model-*.bin",
)
_TOKENIZER_FILE_NAMES = ("tokenizer.json", "tokenizer_config.json", "vocab.json")


def _prepare_sft_jsonl(config: dict) -> Path:
    data_dir = Path(config["data"]["data_dir"])
    train_file = data_dir / config["data"]["train_file"]
    tables_file = data_dir / "tables.json"
    output_file = Path(config["output"]["sft_jsonl"])

    examples = load_spider_file(train_file)
    tables_index = load_tables_json(tables_file)
    count = write_sft_jsonl(examples, tables_index, output_file)
    print(f"Wrote {count} SFT records to {output_file}")
    return output_file


def _build_tokenized_dataset(config: dict, sft_jsonl: Path):
    tokenizer_kind = config.get("tokenizer", {}).get("kind", "whitespace")
    tokenizer = load_tokenizer(tokenizer_kind, config["model"]["path"] if tokenizer_kind == "hf" else None)
    records = load_sft_jsonl(sft_jsonl)
    max_train_samples = config.get("training", {}).get("max_train_samples")
    if max_train_samples is not None:
        records = records[: int(max_train_samples)]
    tokenized = tokenize_sft_records(
        records,
        tokenizer,
        max_prompt_length=int(config["training"]["max_prompt_length"]),
        max_response_length=int(config["training"]["max_response_length"]),
    )
    print(f"Tokenized {len(tokenized)} SFT records")
    return tokenizer, SftTorchDataset(tokenized)


def _model_weight_files(path: str | Path) -> list[Path]:
    root = Path(path)
    files: list[Path] = []
    for pattern in _MODEL_WEIGHT_PATTERNS:
        files.extend(sorted(root.glob(pattern)))
    return [file for file in files if file.is_file()]


def _has_tokenizer_files(path: str | Path) -> bool:
    root = Path(path)
    return any((root / name).is_file() for name in _TOKENIZER_FILE_NAMES)


def _assert_checkpoint_complete(path: str | Path, *, requires_tokenizer: bool) -> None:
    root = Path(path)
    weight_files = _model_weight_files(root)
    if not weight_files:
        expected = ", ".join(_MODEL_WEIGHT_PATTERNS)
        raise RuntimeError(
            f"SFT checkpoint at {root} is incomplete: no model weights found. "
            f"Expected one of: {expected}. Check disk space/quota and whether saving was interrupted."
        )
    if requires_tokenizer and not _has_tokenizer_files(root):
        expected = ", ".join(_TOKENIZER_FILE_NAMES)
        raise RuntimeError(
            f"SFT checkpoint at {root} is incomplete: no tokenizer files found. "
            f"Expected one of: {expected}."
        )


def _trainer_tokenizer_kwargs(trainer_cls, tokenizer) -> dict:
    hf_tokenizer = getattr(tokenizer, "tokenizer", None)
    if hf_tokenizer is None:
        return {}

    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": hf_tokenizer}
    if "tokenizer" in parameters:
        return {"tokenizer": hf_tokenizer}
    return {}


def _save_final_checkpoint(trainer, tokenizer, checkpoint_dir: str | Path) -> None:
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving final SFT checkpoint to {checkpoint_path}")
    try:
        trainer.save_model(str(checkpoint_path))
        hf_tokenizer = getattr(tokenizer, "tokenizer", None)
        if hf_tokenizer is not None:
            hf_tokenizer.save_pretrained(str(checkpoint_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save final SFT checkpoint to {checkpoint_path}. "
            "Check disk space/quota and filesystem permissions."
        ) from exc

    _assert_checkpoint_complete(checkpoint_path, requires_tokenizer=getattr(tokenizer, "tokenizer", None) is not None)
    print(f"Saved final SFT checkpoint to {checkpoint_path}")


def _run_transformers_training(config: dict, tokenizer, dataset: SftTorchDataset) -> None:
    try:
        from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install the train extra to run SFT: pip install -e '.[train]'") from exc

    model = AutoModelForCausalLM.from_pretrained(config["model"]["path"], trust_remote_code=True)
    collator = SftDataCollator(pad_token_id=tokenizer.pad_token_id)
    training = config["training"]
    args = TrainingArguments(
        output_dir=config["output"]["checkpoint_dir"],
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        logging_steps=int(training.get("logging_steps", 10)),
        save_steps=int(training.get("save_steps", 100)),
        max_steps=int(training["max_steps"]) if training.get("max_steps") is not None else -1,
        fp16=bool(training.get("fp16", False)),
        bf16=bool(training.get("bf16", False)),
        report_to=training.get("report_to", "none"),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        **_trainer_tokenizer_kwargs(Trainer, tokenizer),
    )
    trainer.train()
    _save_final_checkpoint(trainer, tokenizer, config["output"]["checkpoint_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run Spider SFT.")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Format data only; do not train.")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_file = _prepare_sft_jsonl(config)
    tokenizer, dataset = _build_tokenized_dataset(config, output_file)

    if not args.dry_run:
        _run_transformers_training(config, tokenizer, dataset)


if __name__ == "__main__":
    main()
