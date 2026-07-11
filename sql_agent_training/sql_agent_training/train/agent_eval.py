"""Evaluate SQL-agent checkpoints on Spider splits."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from sql_agent_training.agent.model_client import HuggingFaceModelClient, ModelClient
from sql_agent_training.agent.sql_agent_loop import SqlAgentInput, SqlAgentLoop
from sql_agent_training.agent.trace_format import AgentTrajectory
from sql_agent_training.data.schema import build_schema_prompt, load_tables_json
from sql_agent_training.data.spider_dataset import SpiderExample, expected_sqlite_path, load_spider_file


@dataclass(frozen=True)
class AgentEvalResult:
    """One evaluated SQL-agent trajectory."""

    uid: str
    db_id: str
    question: str
    gold_sql: str
    final_sql: str | None
    reward: float
    executable: bool
    assistant_turns: int
    num_execute_calls: int
    num_parse_errors: int
    ran_out_of_turns: bool
    no_parseable_sql: bool
    turns: list[dict[str, Any]]


def _trajectory_to_result(example: SpiderExample, trajectory: AgentTrajectory) -> AgentEvalResult:
    metadata = trajectory.metadata
    assistant_turns = sum(1 for turn in trajectory.turns if turn.role == "assistant")

    def clean_turn(turn: Any) -> dict[str, Any]:
        row = asdict(turn)
        for key in ("prompt_ids", "response_ids", "prompt_text", "response_text"):
            row["metadata"].pop(key, None)
        return row

    return AgentEvalResult(
        uid=example.uid,
        db_id=example.db_id,
        question=example.question,
        gold_sql=example.gold_sql,
        final_sql=trajectory.final_sql,
        reward=float(trajectory.reward or 0.0),
        executable=trajectory.final_sql is not None,
        assistant_turns=assistant_turns,
        num_execute_calls=int(metadata.get("num_execute_calls", 0)),
        num_parse_errors=int(metadata.get("num_parse_errors", 0)),
        ran_out_of_turns=bool(metadata.get("ran_out_of_turns", False)),
        no_parseable_sql=bool(metadata.get("no_parseable_sql", False)),
        turns=[clean_turn(turn) for turn in trajectory.turns],
    )


def evaluate_agent(
    examples: list[SpiderExample],
    tables_index: dict[str, dict[str, Any]],
    data_dir: str | Path,
    *,
    model_client: ModelClient | None = None,
    dry_run_gold: bool = False,
    max_turns: int = 2,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> list[AgentEvalResult]:
    """Run the SQL agent over examples and collect per-example results."""

    if model_client is None and not dry_run_gold:
        raise ValueError("model_client is required unless dry_run_gold is enabled")

    loop = SqlAgentLoop(max_turns=max_turns)
    rows: list[AgentEvalResult] = []
    for index, example in enumerate(examples):
        sample = SqlAgentInput(
            uid=example.uid,
            rollout_id=f"{example.uid}:eval{index}",
            question=example.question,
            db_id=example.db_id,
            schema_prompt=build_schema_prompt(example.db_id, tables_index),
            gold_sql=example.gold_sql,
        )
        sqlite_path = expected_sqlite_path(data_dir, example.db_id)
        if dry_run_gold:
            trajectory = loop.run_with_responses(sample, [example.gold_sql], sqlite_path)
        else:
            assert model_client is not None
            trajectory = loop.run(
                sample,
                model_client,
                sqlite_path,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        rows.append(_trajectory_to_result(example, trajectory))
    return rows


def summarize_agent_eval(rows: list[AgentEvalResult]) -> dict[str, float | int]:
    """Aggregate agent evaluation rows into headline and diagnostic metrics."""

    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "execution_accuracy": 0.0,
            "executable_rate": 0.0,
            "avg_turns": 0.0,
            "avg_execute_calls": 0.0,
            "parse_error_rate": 0.0,
            "ran_out_of_turns_rate": 0.0,
            "no_parseable_sql_rate": 0.0,
            "rewrite_rate": 0.0,
            "rewrite_success_rate": 0.0,
        }

    rewrite_rows = [row for row in rows if row.assistant_turns > 1]
    return {
        "total": total,
        "execution_accuracy": sum(row.reward for row in rows) / total,
        "executable_rate": sum(1 for row in rows if row.executable) / total,
        "avg_turns": sum(row.assistant_turns for row in rows) / total,
        "avg_execute_calls": sum(row.num_execute_calls for row in rows) / total,
        "parse_error_rate": sum(1 for row in rows if row.num_parse_errors > 0) / total,
        "ran_out_of_turns_rate": sum(1 for row in rows if row.ran_out_of_turns) / total,
        "no_parseable_sql_rate": sum(1 for row in rows if row.no_parseable_sql) / total,
        "rewrite_rate": len(rewrite_rows) / total,
        "rewrite_success_rate": (
            sum(1 for row in rewrite_rows if row.reward == 1.0) / len(rewrite_rows) if rewrite_rows else 0.0
        ),
    }


def write_eval_outputs(
    rows: list[AgentEvalResult],
    metrics: dict[str, float | int],
    *,
    predictions_jsonl: str | Path,
    metrics_json: str | Path,
) -> None:
    """Write per-example predictions and aggregate metrics."""

    predictions_path = Path(predictions_jsonl)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    metrics_path = Path(metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def _split_file(config: dict[str, Any], split: str) -> str:
    data = config["data"]
    if split == "validation":
        return str(data.get("validation_file", "dev.json"))
    return str(data.get("train_file", "train_spider.json"))


def _has_tokenizer_files(path: str | Path) -> bool:
    root = Path(path)
    return any((root / name).exists() for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"))


def _load_model_client(config: dict[str, Any], checkpoint: str | None, tokenizer_path: str | None) -> HuggingFaceModelClient:
    model_config = config.get("model", {})
    tokenizer_config = config.get("tokenizer", {})
    rollout_config = config.get("rollout", {})
    training_config = config.get("training", {})
    model_path = checkpoint or str(model_config["path"])
    resolved_tokenizer = tokenizer_path or str(model_config.get("tokenizer_path") or tokenizer_config.get("path") or "")
    if not resolved_tokenizer:
        resolved_tokenizer = model_path if _has_tokenizer_files(model_path) else str(model_config["path"])

    return HuggingFaceModelClient(
        model_path,
        tokenizer_name_or_path=resolved_tokenizer,
        device=str(model_config.get("device", training_config.get("device", "auto"))),
        max_new_tokens=int(rollout_config.get("max_response_length", 256)),
        temperature=float(rollout_config.get("temperature", 0.0)),
        top_p=float(rollout_config["top_p"]) if rollout_config.get("top_p") is not None else None,
        top_k=int(rollout_config["top_k"]) if rollout_config.get("top_k") is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SQL-agent model on a Spider split.")
    parser.add_argument("--config", default="configs/agent_eval.yaml")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint to evaluate. Defaults to config model.path.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path. Defaults to checkpoint tokenizer or model.path.")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--predictions-jsonl", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--dry-run-gold", action="store_true", help="Evaluate by emitting gold SQL for plumbing tests.")
    args = parser.parse_args()

    config = _load_config(args.config)
    data_dir = Path(config["data"]["data_dir"])
    examples = load_spider_file(data_dir / _split_file(config, args.split))
    if args.limit is not None:
        examples = examples[: args.limit]
    tables_index = load_tables_json(data_dir / "tables.json")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.checkpoint) / "eval" if args.checkpoint else Path("artifacts/eval/agent")
    )
    predictions_jsonl = Path(args.predictions_jsonl or output_dir / "eval_predictions.jsonl")
    metrics_json = Path(args.metrics_json or output_dir / "eval_metrics.json")

    model_client = None if args.dry_run_gold else _load_model_client(config, args.checkpoint, args.tokenizer)
    rollout_config = config.get("rollout", {})
    rows = evaluate_agent(
        examples,
        tables_index,
        data_dir,
        model_client=model_client,
        dry_run_gold=args.dry_run_gold,
        max_turns=int(rollout_config.get("max_turns", 2)),
        max_tokens=int(rollout_config.get("max_response_length", 256)),
        temperature=float(rollout_config.get("temperature", 0.0)),
    )
    metrics = summarize_agent_eval(rows)
    write_eval_outputs(rows, metrics, predictions_jsonl=predictions_jsonl, metrics_json=metrics_json)
    print(
        json.dumps(
            {
                "predictions": str(predictions_jsonl),
                "metrics_json": str(metrics_json),
                **metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
