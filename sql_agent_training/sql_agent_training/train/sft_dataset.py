"""Tokenized SFT dataset and collator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sql_agent_training.agent.tokenization import TextTokenizer

IGNORE_INDEX = -100


@dataclass(frozen=True)
class SftRecord:
    """One formatted SFT example."""

    uid: str
    db_id: str
    prompt: str
    completion: str


@dataclass(frozen=True)
class TokenizedSftRecord:
    """One tokenized causal LM SFT example."""

    uid: str
    db_id: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def load_sft_jsonl(path: str | Path) -> list[SftRecord]:
    """Load SFT JSONL records."""

    records: list[SftRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                SftRecord(
                    uid=str(row["uid"]),
                    db_id=str(row["db_id"]),
                    prompt=str(row["prompt"]),
                    completion=str(row["completion"]),
                )
            )
    return records


def tokenize_sft_record(
    record: SftRecord,
    tokenizer: TextTokenizer,
    *,
    max_prompt_length: int,
    max_response_length: int,
) -> TokenizedSftRecord:
    """Tokenize one SFT record with prompt labels masked out."""

    prompt_ids = tokenizer.encode(record.prompt)[-max_prompt_length:]
    completion_ids = tokenizer.encode(record.completion)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        completion_ids = completion_ids[: max(max_response_length - 1, 0)] + [eos_token_id]
    else:
        completion_ids = completion_ids[:max_response_length]
    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    return TokenizedSftRecord(
        uid=record.uid,
        db_id=record.db_id,
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
    )


def tokenize_sft_records(
    records: Iterable[SftRecord],
    tokenizer: TextTokenizer,
    *,
    max_prompt_length: int,
    max_response_length: int,
) -> list[TokenizedSftRecord]:
    """Tokenize a collection of SFT records."""

    return [
        tokenize_sft_record(
            record,
            tokenizer,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
        )
        for record in records
    ]


class SftTorchDataset:
    """Lazy torch-compatible wrapper around tokenized records."""

    def __init__(self, records: list[TokenizedSftRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "input_ids": record.input_ids,
            "attention_mask": record.attention_mask,
            "labels": record.labels,
        }


class SftDataCollator:
    """Pad SFT records for causal LM training."""

    def __init__(self, pad_token_id: int, label_pad_token_id: int = IGNORE_INDEX) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [self.label_pad_token_id] * pad_len)

        try:
            import torch
        except ImportError:
            return batch
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
