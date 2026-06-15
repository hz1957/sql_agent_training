"""Minimal local GRPO rollout preparation for the SQL agent.

This module intentionally stops at the smallest useful RL boundary:

1. Build SQL-agent rollouts.
2. Score them with execution reward.
3. Tokenize trajectories.
4. Group trajectories by Spider example id for GRPO-style advantage computation.

It does not launch distributed training. That keeps the code path readable while
preserving the core data shape that a GRPO trainer consumes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import yaml

from sql_agent_training.agent.model_client import HuggingFaceModelClient, ModelClient
from sql_agent_training.agent.sql_agent_loop import SqlAgentInput, SqlAgentLoop
from sql_agent_training.agent.tokenization import load_tokenizer, trajectory_to_tokenized
from sql_agent_training.agent.trace_format import TokenizedTrajectory
from sql_agent_training.data.schema import build_schema_prompt, load_tables_json
from sql_agent_training.data.spider_dataset import SpiderExample, expected_sqlite_path, load_spider_file
from sql_agent_training.train.grpo_batch import GrpoBatch, build_grpo_batch


def _rollout_n(config: dict[str, Any]) -> int:
    return int(config.get("rollout", {}).get("n", 2))


def _max_turns(config: dict[str, Any]) -> int:
    return int(config.get("rollout", {}).get("max_turns", 2))


def _trim_tokenized_trajectory(trajectory: TokenizedTrajectory, config: dict[str, Any]) -> TokenizedTrajectory:
    rollout = config.get("rollout", {})
    max_prompt_length = int(rollout.get("max_prompt_length", len(trajectory.prompt_ids)))
    max_response_length = int(rollout.get("max_response_length", len(trajectory.response_ids)))
    return TokenizedTrajectory(
        uid=trajectory.uid,
        rollout_id=trajectory.rollout_id,
        prompt_ids=trajectory.prompt_ids[-max_prompt_length:],
        response_ids=trajectory.response_ids[:max_response_length],
        response_mask=trajectory.response_mask[:max_response_length],
        reward=trajectory.reward,
        metadata=trajectory.metadata,
    )


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def _write_rollout_summary(batch: GrpoBatch, output_path: str | Path) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for trajectory in batch.trajectories:
            row = {
                "uid": trajectory.uid,
                "rollout_id": trajectory.rollout_id,
                "reward": trajectory.reward,
                "prompt_tokens": len(trajectory.prompt_ids),
                "response_tokens": len(trajectory.response_ids),
                "trainable_response_tokens": sum(trajectory.response_mask),
                "metadata": trajectory.metadata,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _run_one_example(
    *,
    example: SpiderExample,
    schema_prompt: str,
    sqlite_path: str | Path,
    rollout_index: int,
    loop: SqlAgentLoop,
    model_client: ModelClient | None = None,
    scripted_responses: list[str] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    sample = SqlAgentInput(
        uid=example.uid,
        rollout_id=f"{example.uid}:{rollout_index}",
        question=example.question,
        db_id=example.db_id,
        schema_prompt=schema_prompt,
        gold_sql=example.gold_sql,
    )
    if scripted_responses:
        response = scripted_responses[rollout_index % len(scripted_responses)]
        return loop.run_with_responses(sample, [response], sqlite_path)
    if model_client is None:
        return loop.run_with_responses(sample, [example.gold_sql], sqlite_path)
    return loop.run(sample, model_client, sqlite_path, max_tokens=max_tokens, temperature=temperature)


def _load_rollout_model_client(config: dict[str, Any]) -> ModelClient | None:
    model_config = config.get("model", {})
    backend = str(model_config.get("backend", "scripted"))
    if backend in {"scripted", "tiny"}:
        return None
    if backend != "hf":
        raise ValueError(f"Unknown rollout model backend: {backend}")

    model_path = model_config.get("path")
    if not model_path:
        raise ValueError("model.path is required when model.backend is hf")

    rollout_config = config.get("rollout", {})
    return HuggingFaceModelClient(
        str(model_path),
        device=str(model_config.get("device", config.get("training", {}).get("device", "auto"))),
        max_new_tokens=int(rollout_config.get("max_response_length", 256)),
        temperature=float(rollout_config.get("temperature", 0.0)),
    )


def _build_batch_from_examples(
    *,
    examples: list[SpiderExample],
    schema_prompts: dict[str, str],
    sqlite_paths: dict[str, str | Path],
    config: dict[str, Any],
) -> GrpoBatch:
    tokenizer_config = config.get("tokenizer", {})
    tokenizer_kind = tokenizer_config.get("kind", "whitespace")
    model_path = config.get("model", {}).get("path")
    tokenizer = load_tokenizer(tokenizer_kind, model_path if tokenizer_kind == "hf" else None)
    loop = SqlAgentLoop(max_turns=_max_turns(config))
    scripted_responses = config.get("rollout", {}).get("scripted_responses")
    model_client = None if scripted_responses else _load_rollout_model_client(config)
    max_tokens = int(config.get("rollout", {}).get("max_response_length", 256))
    temperature = config.get("rollout", {}).get("temperature")
    temperature = float(temperature) if temperature is not None else None

    tokenized = []
    for example in examples:
        schema_prompt = schema_prompts[example.db_id]
        sqlite_path = sqlite_paths[example.db_id]
        for rollout_index in range(_rollout_n(config)):
            trajectory = _run_one_example(
                example=example,
                schema_prompt=schema_prompt,
                sqlite_path=sqlite_path,
                rollout_index=rollout_index,
                loop=loop,
                model_client=model_client,
                scripted_responses=scripted_responses,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            tokenized.append(_trim_tokenized_trajectory(trajectory_to_tokenized(trajectory, tokenizer), config))

    return build_grpo_batch(tokenized, rollout_n=_rollout_n(config))


def _demo_examples() -> tuple[list[SpiderExample], dict[str, str], dict[str, Path], tempfile.TemporaryDirectory[str]]:
    temp_dir = tempfile.TemporaryDirectory()
    db_path = Path(temp_dir.name) / "music.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE Singer (Name TEXT)")
        conn.execute("INSERT INTO Singer VALUES ('Ada')")
        conn.execute("INSERT INTO Singer VALUES ('Grace')")
        conn.commit()
    finally:
        conn.close()

    example = SpiderExample(
        uid="demo:0",
        db_id="music",
        question="List singer names.",
        gold_sql="SELECT Name FROM Singer",
    )
    schema_prompts = {"music": "Database: music\n- Singer(Name)"}
    sqlite_paths = {"music": db_path}
    return [example], schema_prompts, sqlite_paths, temp_dir


def build_grpo_batch_from_config(config: dict[str, Any]) -> GrpoBatch:
    """Build a tokenized local GRPO batch from demo or Spider data."""

    if bool(config.get("dry_run", False)):
        examples, schema_prompts, sqlite_paths, temp_dir = _demo_examples()
        try:
            return _build_batch_from_examples(
                examples=examples,
                schema_prompts=schema_prompts,
                sqlite_paths=sqlite_paths,
                config=config,
            )
        finally:
            temp_dir.cleanup()

    data = config["data"]
    data_dir = Path(data["data_dir"])
    train_file = data["train_file"]
    examples = load_spider_file(data_dir / train_file)
    limit = config.get("rollout", {}).get("train_limit")
    if limit is not None:
        examples = examples[: int(limit)]

    tables_index = load_tables_json(data_dir / "tables.json")
    schema_prompts = {example.db_id: build_schema_prompt(example.db_id, tables_index) for example in examples}
    sqlite_paths = {example.db_id: expected_sqlite_path(data_dir, example.db_id) for example in examples}
    return _build_batch_from_examples(
        examples=examples,
        schema_prompts=schema_prompts,
        sqlite_paths=sqlite_paths,
        config=config,
    )


def run_grpo(config: dict[str, Any]) -> dict[str, Any]:
    """Run the minimal local GRPO rollout flow and return summary metrics."""

    batch = build_grpo_batch_from_config(config)
    output_path = config.get("output", {}).get("rollouts_jsonl", "artifacts/grpo/rollouts.jsonl")
    rows_written = _write_rollout_summary(batch, output_path)
    rewards = [trajectory.reward for trajectory in batch.trajectories]
    return {
        "groups": len(batch.groups),
        "trajectories": batch.num_trajectories,
        "rows_written": rows_written,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "rollouts_jsonl": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal local SQL-agent GRPO rollout preparation.")
    parser.add_argument("--config", default="configs/grpo.local_dryrun.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force the built-in demo instead of loading Spider data.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.dry_run:
        config["dry_run"] = True
    summary = run_grpo(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
