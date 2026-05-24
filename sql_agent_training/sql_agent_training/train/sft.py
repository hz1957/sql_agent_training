"""SFT entrypoint for formatting Spider records and running Transformers training."""

from __future__ import annotations

import argparse
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
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    trainer.train()
    trainer.save_model(config["output"]["checkpoint_dir"])
    if hasattr(tokenizer, "tokenizer"):
        tokenizer.tokenizer.save_pretrained(config["output"]["checkpoint_dir"])


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
