"""Generate verified Spider trajectories and final-action SFT records.

Example:
    uv run --no-sync python scripts/generate_sft_trajectories.py \
        --question-count 500 --rollouts-per-question 4 --target-correct 800
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_agent_training.agent.model_client import VllmOpenAIModelClient  # noqa: E402
from sql_agent_training.agent.sql_agent_loop import SqlAgentInput, SqlAgentLoop  # noqa: E402
from sql_agent_training.agent.trace_format import AgentTrajectory, AgentTurn  # noqa: E402
from sql_agent_training.data.schema import build_schema_prompt, load_tables_json  # noqa: E402
from sql_agent_training.data.spider_dataset import SpiderExample, expected_sqlite_path, load_spider_file  # noqa: E402


@dataclass(frozen=True)
class TrajectorySample:
    """One sampled Spider question with its rendered schema and database."""

    example: SpiderExample
    schema_prompt: str
    sqlite_path: Path


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_samples(
    data_dir: Path,
    *,
    split_file: str,
    question_count: int,
    seed: int,
) -> list[TrajectorySample]:
    examples = load_spider_file(data_dir / split_file)
    if question_count > len(examples):
        raise ValueError(f"Requested {question_count} questions, but {split_file} contains only {len(examples)}")

    selected = random.Random(seed).sample(examples, question_count)
    tables_index = load_tables_json(data_dir / "tables.json")
    schema_cache: dict[str, str] = {}
    samples: list[TrajectorySample] = []
    for example in selected:
        schema_prompt = schema_cache.setdefault(
            example.db_id,
            build_schema_prompt(example.db_id, tables_index),
        )
        sqlite_path = expected_sqlite_path(data_dir, example.db_id)
        if not sqlite_path.exists():
            raise FileNotFoundError(f"Missing SQLite database for {example.db_id}: {sqlite_path}")
        samples.append(
            TrajectorySample(
                example=example,
                schema_prompt=schema_prompt,
                sqlite_path=sqlite_path,
            )
        )
    return samples


def _clean_turn(turn: AgentTurn) -> dict[str, Any]:
    metadata = dict(turn.metadata or {})
    metadata.pop("prompt_ids", None)
    metadata.pop("response_ids", None)
    return {
        "role": turn.role,
        "content": turn.content,
        "metadata": metadata,
    }


def _trajectory_row(
    sample: TrajectorySample,
    trajectory: AgentTrajectory,
    *,
    rollout_index: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    turns = [_clean_turn(turn) for turn in trajectory.turns]
    sql_actions = [
        turn for turn in turns if turn["role"] == "assistant" and bool(turn["metadata"].get("trainable", False))
    ]
    return {
        "uid": sample.example.uid,
        "db_id": sample.example.db_id,
        "question": sample.example.question,
        "gold_sql": sample.example.gold_sql,
        "rollout_id": trajectory.rollout_id,
        "rollout_index": rollout_index,
        "reward": float(trajectory.reward or 0.0),
        "final_sql": trajectory.final_sql,
        "final_sql_source": trajectory.final_sql_source,
        "num_sql_actions": len(sql_actions),
        "is_rewrite_trajectory": len(sql_actions) > 1,
        "elapsed_seconds": elapsed_seconds,
        "metadata": dict(trajectory.metadata or {}),
        "turns": turns,
    }


def _trajectory_signature(row: dict[str, Any]) -> str:
    trace = [(turn["role"], turn["content"]) for turn in row["turns"]]
    return json.dumps([row["uid"], trace], ensure_ascii=False, sort_keys=True)


def _select_verified_rows(
    rows: list[dict[str, Any]],
    *,
    target_correct: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    correct = [row for row in rows if float(row["reward"]) == 1.0 and str(row.get("final_sql") or "").strip()]
    deduplicated: list[dict[str, Any]] = []
    signatures: set[str] = set()
    for row in correct:
        signature = _trajectory_signature(row)
        if signature in signatures:
            continue
        signatures.add(signature)
        deduplicated.append(row)

    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deduplicated:
        grouped[str(row["uid"])].append(row)
    for group in grouped.values():
        rng.shuffle(group)

    uids = sorted(grouped)
    rng.shuffle(uids)
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < target_correct:
        added = False
        for uid in uids:
            group = grouped[uid]
            if round_index >= len(group):
                continue
            selected.append(group[round_index])
            added = True
            if len(selected) == target_correct:
                break
        if not added:
            break
        round_index += 1
    return selected, len(deduplicated)


def _trajectory_to_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    actions = [
        turn for turn in row["turns"] if turn["role"] == "assistant" and bool(turn["metadata"].get("trainable", False))
    ]
    if not actions:
        raise ValueError(f"Verified trajectory has no trainable SQL action: {row['rollout_id']}")

    final_action = actions[-1]
    metadata = final_action["metadata"]
    prompt = str(metadata.get("prompt_text") or "").strip()
    if not prompt:
        final_index = row["turns"].index(final_action)
        preceding = next(
            (turn for turn in reversed(row["turns"][:final_index]) if turn["role"] == "user"),
            None,
        )
        if preceding is None:
            raise ValueError(f"Final SQL action has no prompt: {row['rollout_id']}")
        prompt = str(preceding["content"]).strip()

    turn_index = int(metadata.get("turn_index", len(actions) - 1))
    return {
        "uid": f"{row['rollout_id']}:turn{turn_index}",
        "db_id": row["db_id"],
        "prompt": prompt,
        "completion": str(row["final_sql"]).strip(),
        "source_uid": row["uid"],
        "source_rollout_id": row["rollout_id"],
        "agent_step": str(metadata.get("agent_step", "rewrite_query" if len(actions) > 1 else "write_query")),
        "turn_index": turn_index,
        "is_rewrite_trajectory": bool(row["is_rewrite_trajectory"]),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_rollout(
    client: VllmOpenAIModelClient,
    sample: TrajectorySample,
    *,
    rollout_index: int,
    max_turns: int,
    max_tokens_per_call: int,
    temperature: float,
    top_p: float,
    top_k: int,
    request_retries: int,
) -> dict[str, Any]:
    rollout_id = f"{sample.example.uid}:trajectory_sft:rollout={rollout_index}"
    agent_input = SqlAgentInput(
        uid=sample.example.uid,
        rollout_id=rollout_id,
        question=sample.example.question,
        db_id=sample.example.db_id,
        schema_prompt=sample.schema_prompt,
        gold_sql=sample.example.gold_sql,
    )

    for attempt in range(request_retries + 1):
        start = time.perf_counter()
        try:
            trajectory = SqlAgentLoop(max_turns=max_turns).run(
                agent_input,
                client,
                sample.sqlite_path,
                max_tokens=max_tokens_per_call,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            return _trajectory_row(
                sample,
                trajectory,
                rollout_index=rollout_index,
                elapsed_seconds=time.perf_counter() - start,
            )
        except Exception:
            if attempt == request_retries:
                raise
            time.sleep(min(2**attempt, 4))
    raise AssertionError("unreachable")


def generate(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = _resolve_project_path(args.data_dir)
    output_dir = _resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _load_samples(
        data_dir,
        split_file=args.split_file,
        question_count=args.question_count,
        seed=args.seed,
    )

    model_path = _resolve_project_path(args.model_path)
    tokenizer_path = _resolve_project_path(args.tokenizer_path or args.model_path)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to generate trajectory SFT data.") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    client = VllmOpenAIModelClient(
        base_url=args.base_url,
        model_name=args.model_name,
        tokenizer=tokenizer,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        max_new_tokens=args.max_tokens_per_call,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    jobs = [(sample, rollout_index) for sample in samples for rollout_index in range(args.rollouts_per_question)]
    print(
        f"Generating {len(jobs)} trajectories from {len(samples)} questions "
        f"with workers={args.workers} model={model_path}",
        flush=True,
    )
    start = time.perf_counter()

    def run_job(job: tuple[TrajectorySample, int]) -> dict[str, Any]:
        sample, rollout_index = job
        return _run_rollout(
            client,
            sample,
            rollout_index=rollout_index,
            max_turns=args.max_turns,
            max_tokens_per_call=args.max_tokens_per_call,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            request_retries=args.request_retries,
        )

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, row in enumerate(executor.map(run_job, jobs), start=1):
            rows.append(row)
            if completed % args.log_every == 0 or completed == len(jobs):
                correct = sum(float(item["reward"]) == 1.0 for item in rows)
                print(f"completed={completed}/{len(jobs)} correct_so_far={correct}", flush=True)

    selected, deduplicated_correct = _select_verified_rows(
        rows,
        target_correct=args.target_correct,
        seed=args.seed,
    )
    sft_rows = [_trajectory_to_sft_row(row) for row in selected]
    candidate_path = output_dir / "candidate_trajectories.jsonl"
    verified_path = output_dir / "verified_trajectories.jsonl"
    sft_path = output_dir / "trajectory_sft.jsonl"
    _write_jsonl(candidate_path, rows)
    _write_jsonl(verified_path, selected)
    _write_jsonl(sft_path, sft_rows)

    direct_count = sum(not bool(row["is_rewrite_trajectory"]) for row in selected)
    rewrite_count = len(selected) - direct_count
    summary = {
        "model_path": str(model_path),
        "model_name": args.model_name,
        "data_dir": str(data_dir),
        "split_file": args.split_file,
        "seed": args.seed,
        "question_count": len(samples),
        "rollouts_per_question": args.rollouts_per_question,
        "candidate_trajectories": len(rows),
        "correct_before_deduplication": sum(float(row["reward"]) == 1.0 for row in rows),
        "correct_after_deduplication": deduplicated_correct,
        "target_correct": args.target_correct,
        "selected_correct": len(selected),
        "selected_unique_questions": len({row["uid"] for row in selected}),
        "selected_direct_trajectories": direct_count,
        "selected_rewrite_trajectories": rewrite_count,
        "trajectory_sft_records": len(sft_rows),
        "temperature": args.temperature,
        "max_turns": args.max_turns,
        "elapsed_seconds": time.perf_counter() - start,
        "candidate_trajectories_jsonl": str(candidate_path),
        "verified_trajectories_jsonl": str(verified_path),
        "trajectory_sft_jsonl": str(sft_path),
        "target_met": len(selected) == args.target_correct,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--model-path", default="data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged")
    parser.add_argument("--model-name", default="qwen25-coder-14b-sft-merged")
    parser.add_argument("--tokenizer-path", default=None)

    parser.add_argument("--data-dir", default="data/spider")
    parser.add_argument("--split-file", default="train_spider.json")
    parser.add_argument("--question-count", type=int, default=500)
    parser.add_argument("--rollouts-per-question", type=int, default=4)
    parser.add_argument("--target-correct", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-tokens-per-call", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        default="artifacts/sft_trajectory/qwen25_coder_14b_sft_merged_q500_n4_target800_seed42",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "question_count",
        "rollouts_per_question",
        "target_correct",
        "max_turns",
        "max_tokens_per_call",
        "workers",
        "log_every",
    )
    for field in positive_fields:
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    candidates = args.question_count * args.rollouts_per_question
    if args.target_correct > candidates:
        raise ValueError(f"--target-correct cannot exceed the {candidates} candidate trajectories")
    if args.request_retries < 0:
        raise ValueError("--request-retries cannot be negative")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    summary = generate(args)
    if not summary["target_met"]:
        print(
            f"ERROR: target={summary['target_correct']} but only "
            f"{summary['selected_correct']} unique correct trajectories were available.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
