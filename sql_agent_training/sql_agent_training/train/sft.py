"""SFT entrypoint for formatting Spider records and running Transformers training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
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


def _new_checkpoint_dir(base_dir: str | Path) -> Path:
    """Create a new timestamped SFT checkpoint directory path."""

    root = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    counter = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{counter:02d}"
        counter += 1
    return candidate


def _new_final_checkpoint_dir(config: dict) -> Path:
    """Return a fresh directory that should contain the eval-ready SFT model."""

    return _new_checkpoint_dir(config["output"]["checkpoint_dir"])


def _trainer_output_dir(config: dict, checkpoint_dir: str | Path) -> Path:
    """Return the private Hugging Face Trainer state/checkpoint directory."""

    output = config.get("output", {})
    if output.get("trainer_checkpoint_dir"):
        return Path(output["trainer_checkpoint_dir"])
    return Path(checkpoint_dir) / "trainer_checkpoints"


def _write_run_config(config: dict, checkpoint_dir: str | Path) -> Path:
    path = Path(checkpoint_dir) / "run_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _run_transformers_training(config: dict, tokenizer, dataset: SftTorchDataset) -> dict[str, str]:
    try:
        from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install the train extra to run SFT: pip install -e '.[train]'") from exc

    model = AutoModelForCausalLM.from_pretrained(config["model"]["path"], trust_remote_code=True)
    collator = SftDataCollator(pad_token_id=tokenizer.pad_token_id)
    training = config["training"]
    save_strategy = str(training.get("save_strategy", "no"))
    final_checkpoint_dir = _new_final_checkpoint_dir(config)
    final_checkpoint_dir.mkdir(parents=True, exist_ok=False)
    run_config_path = _write_run_config(config, final_checkpoint_dir)
    args = TrainingArguments(
        output_dir=str(_trainer_output_dir(config, final_checkpoint_dir)),
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        logging_steps=int(training.get("logging_steps", 10)),
        save_strategy=save_strategy,
        save_steps=int(training.get("save_steps", 100)),
        max_steps=int(training["max_steps"]) if training.get("max_steps") is not None else -1,
        fp16=bool(training.get("fp16", False)),
        bf16=bool(training.get("bf16", False)),
        report_to=training.get("report_to", "none"),
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    trainer.train()
    trainer.save_model(str(final_checkpoint_dir))
    if hasattr(tokenizer, "tokenizer"):
        tokenizer.tokenizer.save_pretrained(str(final_checkpoint_dir))
    return {
        "checkpoint_dir": str(final_checkpoint_dir),
        "trainer_checkpoint_dir": str(_trainer_output_dir(config, final_checkpoint_dir)),
        "run_config": str(run_config_path),
    }


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
        summary = _run_transformers_training(config, tokenizer, dataset)
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
