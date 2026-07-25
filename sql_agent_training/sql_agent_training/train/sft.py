"""SFT entrypoint for formatting Spider records and running Transformers training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from sql_agent_training.agent.tokenization import load_tokenizer
from sql_agent_training.data.schema import load_tables_json
from sql_agent_training.data.sft_formatter import format_sft_record, write_sft_jsonl
from sql_agent_training.data.spider_dataset import load_spider_file
from sql_agent_training.train.distributed import DistributedContext, barrier, broadcast_object, init_distributed
from sql_agent_training.train.eval_sampling import select_eval_examples
from sql_agent_training.train.sft_dataset import (
    IGNORE_INDEX,
    SftDataCollator,
    SftRecord,
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


def _build_tokenized_eval_dataset(config: dict, tokenizer) -> SftTorchDataset:
    """Build the validation split used for periodic Trainer eval metrics."""

    data = config["data"]
    data_dir = Path(data["data_dir"])
    validation_file = data.get("validation_file")
    if not validation_file:
        raise ValueError("data.validation_file is required when training.eval_strategy is not 'no'")

    examples = load_spider_file(data_dir / validation_file)
    eval_config = config.get("eval", {})
    examples = select_eval_examples(
        examples,
        sample_size=eval_config.get("sample_size"),
        sample_seed=int(eval_config.get("sample_seed", 0)),
    )
    tables_index = load_tables_json(data_dir / "tables.json")
    records = []
    for example in examples:
        row = format_sft_record(example, tables_index)
        records.append(
            SftRecord(
                uid=str(row["uid"]),
                db_id=str(row["db_id"]),
                prompt=str(row["prompt"]),
                completion=str(row["completion"]),
            )
        )
    tokenized = tokenize_sft_records(
        records,
        tokenizer,
        max_prompt_length=int(config["training"]["max_prompt_length"]),
        max_response_length=int(config["training"]["max_response_length"]),
    )
    print(f"Tokenized {len(tokenized)} validation SFT records")
    return SftTorchDataset(tokenized)


def _build_train_metric_dataset(config: dict, dataset: SftTorchDataset) -> SftTorchDataset:
    """Build a fixed train subset used only for periodic token-accuracy monitoring."""

    eval_config = config.get("eval", {})
    sample_size = eval_config.get("train_sample_size", eval_config.get("sample_size"))
    records = select_eval_examples(
        list(dataset.records),
        sample_size=sample_size,
        sample_seed=int(eval_config.get("sample_seed", 0)),
    )
    print(f"Selected {len(records)} train SFT records for periodic metrics")
    return SftTorchDataset(records)


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
    """Return the Hugging Face Trainer output directory for this run."""

    output = config.get("output", {})
    if output.get("trainer_output_dir"):
        return Path(output["trainer_output_dir"])
    if output.get("trainer_checkpoint_dir"):
        return Path(output["trainer_checkpoint_dir"])
    return Path(checkpoint_dir)


def _normalize_save_strategy(value) -> str:
    """Normalize YAML-loaded save_strategy values for Hugging Face TrainingArguments."""

    if value is None or value is False:
        return "no"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "none"}:
            return "no"
        if normalized in {"no", "steps", "epoch", "best"}:
            return normalized
    raise ValueError("training.save_strategy must be one of 'no', 'steps', 'epoch', or 'best'; " f"got {value!r}")


def _deepspeed_config(training: dict) -> str | dict | None:
    """Return a Trainer-compatible DeepSpeed config value from training settings."""

    value = training.get("deepspeed")
    if value is None or value is False:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, dict):
        return value
    raise ValueError("training.deepspeed must be a config path, inline config dict, false, or null")


def _eval_strategy(training: dict) -> str:
    """Return the normalized Hugging Face Trainer eval strategy."""

    value = str(training.get("eval_strategy", "no")).strip().lower()
    if value in {"false", "none"}:
        return "no"
    if value not in {"no", "steps", "epoch"}:
        raise ValueError("training.eval_strategy must be one of 'no', 'steps', or 'epoch'")
    return value


def _optional_positive_int(training: dict, key: str) -> int | None:
    """Return an optional positive integer training setting."""

    value = training.get(key)
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"training.{key} must be positive when set")
    return parsed


def _preprocess_logits_for_metrics(logits: Any, labels: Any) -> Any:
    """Convert full-vocab logits to token predictions before metric gathering."""

    del labels
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def _compute_token_accuracy(eval_pred: Any) -> dict[str, float]:
    """Compute next-token accuracy on unmasked SFT completion labels."""

    import numpy as np

    predictions, labels = eval_pred
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)
    if predictions.shape[1] < 2 or labels.shape[1] < 2:
        return {"token_accuracy": 0.0}
    shifted_predictions = predictions[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != IGNORE_INDEX
    total = int(mask.sum())
    if total == 0:
        return {"token_accuracy": 0.0}
    correct = int((shifted_predictions[mask] == shifted_labels[mask]).sum())
    return {"token_accuracy": correct / total}


def _write_run_config(config: dict, checkpoint_dir: str | Path, *, is_main_process: bool = True) -> Path:
    path = Path(checkpoint_dir) / "run_config.yaml"
    if is_main_process:
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _lora_config_kwargs(config: dict) -> dict:
    """Return PEFT LoRA kwargs from config, filling stable defaults."""

    lora = config.get("lora", {})
    return {
        "r": int(lora.get("r", 16)),
        "lora_alpha": int(lora.get("alpha", lora.get("lora_alpha", 32))),
        "lora_dropout": float(lora.get("dropout", lora.get("lora_dropout", 0.05))),
        "bias": str(lora.get("bias", "none")),
        "target_modules": list(
            lora.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    }


def _apply_lora_if_enabled(model: Any, config: dict) -> Any:
    """Wrap the model with PEFT LoRA adapters when configured."""

    if not config.get("lora", {}).get("enabled", False):
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install the train extra with PEFT to run LoRA SFT: pip install -e '.[train]'") from exc

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **_lora_config_kwargs(config))
    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()
    return lora_model


def _model_load_kwargs(training: dict) -> dict[str, object]:
    """Return model loading kwargs that match the requested precision."""

    kwargs = {"trust_remote_code": True}
    if bool(training.get("bf16", False)):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is required for training
            raise RuntimeError("Install torch to run SFT training.") from exc
        kwargs["torch_dtype"] = torch.bfloat16
    elif bool(training.get("fp16", False)):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is required for training
            raise RuntimeError("Install torch to run SFT training.") from exc
        kwargs["torch_dtype"] = torch.float16
    return kwargs


def _run_transformers_training(
    config: dict,
    tokenizer,
    dataset: SftTorchDataset,
    *,
    context: DistributedContext | None = None,
) -> dict[str, str]:
    try:
        from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install the train extra to run SFT: pip install -e '.[train]'") from exc

    context = context or init_distributed("auto")
    training = config["training"]
    gradient_checkpointing = bool(training.get("gradient_checkpointing", False))
    save_strategy = _normalize_save_strategy(training.get("save_strategy", "no"))
    eval_strategy = _eval_strategy(training)
    run_config = {**config, "training": {**training, "save_strategy": save_strategy}}
    checkpoint_dir_value = str(_new_final_checkpoint_dir(config)) if context.is_main_process else None
    final_checkpoint_dir = Path(broadcast_object(checkpoint_dir_value, context))
    if context.is_main_process:
        final_checkpoint_dir.mkdir(parents=True, exist_ok=False)
    barrier(context)
    run_config_path = _write_run_config(
        run_config,
        final_checkpoint_dir,
        is_main_process=context.is_main_process,
    )
    args = TrainingArguments(
        output_dir=str(_trainer_output_dir(config, final_checkpoint_dir)),
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        logging_steps=int(training.get("logging_steps", 10)),
        save_strategy=save_strategy,
        save_steps=int(training.get("save_steps", 100)),
        eval_strategy=eval_strategy,
        eval_steps=_optional_positive_int(training, "eval_steps"),
        per_device_eval_batch_size=int(training.get("per_device_eval_batch_size", 1)),
        eval_accumulation_steps=_optional_positive_int(training, "eval_accumulation_steps"),
        max_steps=int(training["max_steps"]) if training.get("max_steps") is not None else -1,
        fp16=bool(training.get("fp16", False)),
        bf16=bool(training.get("bf16", False)),
        gradient_checkpointing=gradient_checkpointing,
        warmup_ratio=float(training.get("warmup_ratio", 0.0)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        report_to=training.get("report_to", "none"),
        deepspeed=_deepspeed_config(training),
    )
    model = AutoModelForCausalLM.from_pretrained(config["model"]["path"], **_model_load_kwargs(training))
    if bool(training.get("gradient_checkpointing", False)) and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model = _apply_lora_if_enabled(model, config)
    if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    collator = SftDataCollator(pad_token_id=tokenizer.pad_token_id)
    eval_dataset = (
        {
            "train": _build_train_metric_dataset(config, dataset),
            "validation": _build_tokenized_eval_dataset(config, tokenizer),
        }
        if eval_strategy != "no"
        else None
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=_compute_token_accuracy if eval_dataset is not None else None,
        preprocess_logits_for_metrics=_preprocess_logits_for_metrics if eval_dataset is not None else None,
    )
    trainer.train()
    # Free GPU memory before serializing weights to CPU RAM to avoid OOM during save.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    trainer.save_model(str(final_checkpoint_dir))
    if context.is_main_process and hasattr(tokenizer, "tokenizer"):
        tokenizer.tokenizer.save_pretrained(str(final_checkpoint_dir))
    barrier(context)
    return {
        "checkpoint_dir": str(final_checkpoint_dir),
        "trainer_output_dir": str(_trainer_output_dir(config, final_checkpoint_dir)),
        "run_config": str(run_config_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run Spider SFT.")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Format data only; do not train.")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    context = init_distributed("auto")
    if context.is_main_process:
        output_file = _prepare_sft_jsonl(config)
    else:
        output_file = Path(config["output"]["sft_jsonl"])
    barrier(context)
    tokenizer, dataset = _build_tokenized_dataset(config, output_file)

    if not args.dry_run:
        summary = _run_transformers_training(config, tokenizer, dataset, context=context)
        if context.is_main_process:
            print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
