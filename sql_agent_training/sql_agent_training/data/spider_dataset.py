"""Spider dataset loading and verification utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SpiderExample:
    """One Spider text-to-SQL example."""

    uid: str
    db_id: str
    question: str
    gold_sql: str


def example_from_mapping(row: dict[str, Any], index: int = 0) -> SpiderExample:
    """Normalize a mapping from HF/JSON into SpiderExample."""

    db_id = str(row["db_id"])
    return SpiderExample(
        uid=str(row.get("uid") or row.get("id") or f"{db_id}:{index}"),
        db_id=db_id,
        question=str(row["question"]),
        gold_sql=str(row.get("query") or row.get("gold_sql") or row.get("sql")),
    )


def load_spider_json(path: str | Path) -> list[SpiderExample]:
    """Load Spider-style JSON records."""

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return [example_from_mapping(row, index) for index, row in enumerate(rows)]


def load_spider_jsonl(path: str | Path) -> list[SpiderExample]:
    """Load Spider-style JSONL records."""

    jsonl_path = Path(path)
    examples: list[SpiderExample] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            examples.append(example_from_mapping(json.loads(line), index))
    return examples


def load_spider_file(path: str | Path) -> list[SpiderExample]:
    """Load Spider examples from JSON or JSONL."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return load_spider_json(file_path)
    if suffix == ".jsonl":
        return load_spider_jsonl(file_path)
    raise ValueError(f"Unsupported Spider file extension: {file_path}")


def write_spider_json(examples: Iterable[SpiderExample], path: str | Path) -> int:
    """Write normalized Spider examples to JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"uid": example.uid, "db_id": example.db_id, "question": example.question, "query": example.gold_sql}
        for example in examples
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    return len(rows)


def load_hf_spider(split: str = "train") -> list[SpiderExample]:
    """Load xlangai/spider from Hugging Face datasets."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError("Install datasets to load xlangai/spider.") from exc

    dataset = load_dataset("xlangai/spider", split=split)
    return [example_from_mapping(row, index) for index, row in enumerate(dataset)]  # type: ignore[arg-type]


def expected_sqlite_path(data_dir: str | Path, db_id: str) -> Path:
    """Return expected Spider SQLite path for a db_id."""

    root = Path(data_dir)
    return root / "database" / db_id / f"{db_id}.sqlite"


def find_missing_databases(examples: Iterable[SpiderExample], data_dir: str | Path) -> list[str]:
    """Return sorted db_ids whose SQLite files are missing."""

    missing = {example.db_id for example in examples if not expected_sqlite_path(data_dir, example.db_id).exists()}
    return sorted(missing)


def verify_spider_assets(data_dir: str | Path, examples: Iterable[SpiderExample] | None = None) -> dict[str, Any]:
    """Verify tables.json and database files under a Spider data directory."""

    root = Path(data_dir)
    tables_path = root / "tables.json"
    database_dir = root / "database"
    result: dict[str, Any] = {
        "data_dir": str(root),
        "tables_json_exists": tables_path.exists(),
        "database_dir_exists": database_dir.exists(),
        "missing_db_ids": [],
    }
    if examples is not None:
        example_list = list(examples)
        result["num_examples"] = len(example_list)
        result["missing_db_ids"] = find_missing_databases(example_list, root)
    result["ok"] = result["tables_json_exists"] and result["database_dir_exists"] and not result["missing_db_ids"]
    return result
