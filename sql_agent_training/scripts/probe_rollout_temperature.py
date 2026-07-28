"""Probe rollout temperature for the 14B SQL-agent policy without training.

The script runs the existing local SQL-agent loop over a fixed Spider sample set
for multiple temperatures. It reports group reward variance so GRPO temperature
can be chosen before launching expensive training runs.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_agent_training.agent.model_client import (  # noqa: E402
    HuggingFaceModelClient,
    ModelClient,
    VllmOpenAIModelClient,
)
from sql_agent_training.agent.sql_agent_loop import SqlAgentInput, SqlAgentLoop  # noqa: E402
from sql_agent_training.agent.trace_format import AgentTrajectory, AgentTurn  # noqa: E402
from sql_agent_training.data.schema import build_schema_prompt, load_tables_json  # noqa: E402
from sql_agent_training.data.spider_dataset import SpiderExample, expected_sqlite_path, load_spider_file  # noqa: E402


@dataclass(frozen=True)
class RolloutProbeSample:
    """A Spider sample with rendered schema and SQLite path."""

    example: SpiderExample
    schema_prompt: str
    sqlite_path: Path


@dataclass(frozen=True)
class TemperatureMetrics:
    """Aggregate metrics for one temperature."""

    temperature: float
    tasks: int
    trajectories: int
    rollout_n: int
    mean_reward: float
    reward_std: float
    reward_min: float
    reward_max: float
    nonzero_variance_groups: int
    zero_variance_groups: int
    nonzero_variance_group_ratio: float
    zero_variance_group_ratio: float
    reward_variance_mean: float
    sqlite_execution_success_rate: float
    executable_but_incorrect_rate: float
    invalid_sql_rate: float
    empty_sql_rate: float
    average_num_turns: float
    average_execute_calls: float
    average_check_calls: float
    average_parse_errors: float
    average_prompt_tokens: float | None
    average_response_tokens: float | None
    average_trainable_tokens: float | None
    elapsed_seconds: float
    rollouts_jsonl: str


class GoldSqlClient:
    """Sentinel backend that returns the gold SQL through run_with_responses."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return str(value)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _load_samples(
    *,
    data_dir: Path,
    split_file: str,
    limit: int,
    offset: int,
    seed: int,
    shuffle: bool,
) -> list[RolloutProbeSample]:
    examples = load_spider_file(data_dir / split_file)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(examples)
    examples = examples[offset : offset + limit]

    tables_index = load_tables_json(data_dir / "tables.json")
    schema_cache: dict[str, str] = {}
    samples = []
    for example in examples:
        schema = schema_cache.setdefault(example.db_id, build_schema_prompt(example.db_id, tables_index))
        sqlite_path = expected_sqlite_path(data_dir, example.db_id)
        if not sqlite_path.exists():
            raise FileNotFoundError(f"Missing SQLite database for {example.db_id}: {sqlite_path}")
        samples.append(RolloutProbeSample(example=example, schema_prompt=schema, sqlite_path=sqlite_path))
    return samples


def _maybe_load_tokenizer(tokenizer_path: str | None) -> Any | None:
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers or omit --tokenizer-path.") from exc
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _build_model_client(args: argparse.Namespace) -> ModelClient | GoldSqlClient:
    if args.backend == "gold":
        return GoldSqlClient()

    resolved_model_path = str(_resolve_project_path(args.model_path))
    resolved_tokenizer_path = str(_resolve_project_path(args.tokenizer_path or args.model_path))
    resolved_lora_path = str(_resolve_project_path(args.lora_path)) if args.lora_path else None
    tokenizer = _maybe_load_tokenizer(resolved_tokenizer_path) if args.backend == "vllm" else None

    if args.backend == "vllm":
        model_name = args.model_name
        if not model_name:
            model_name = args.lora_name if args.lora_path else args.model_path
        client = VllmOpenAIModelClient(
            base_url=args.base_url,
            model_name=str(model_name),
            tokenizer=tokenizer,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            max_new_tokens=args.max_tokens_per_call,
            temperature=args.temperatures[0],
            top_p=args.top_p,
            top_k=args.top_k,
        )
        if resolved_lora_path:
            client.load_lora_adapter(lora_name=args.lora_name, lora_path=resolved_lora_path)
        return client

    if args.backend == "hf":
        model_path = resolved_lora_path or resolved_model_path
        return HuggingFaceModelClient(
            str(model_path),
            tokenizer_name_or_path=resolved_tokenizer_path,
            base_model_name_or_path=resolved_model_path if resolved_lora_path else None,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_tokens_per_call,
            temperature=args.temperatures[0],
            top_p=args.top_p,
            top_k=args.top_k,
        )

    raise ValueError(f"Unsupported backend: {args.backend}")


def _agent_input(sample: RolloutProbeSample, rollout_id: str) -> SqlAgentInput:
    example = sample.example
    return SqlAgentInput(
        uid=example.uid,
        rollout_id=rollout_id,
        question=example.question,
        db_id=example.db_id,
        schema_prompt=sample.schema_prompt,
        gold_sql=example.gold_sql,
    )


def _assistant_turns(trajectory: AgentTrajectory) -> list[AgentTurn]:
    return [turn for turn in trajectory.turns if turn.role == "assistant"]


def _token_stats(trajectory: AgentTrajectory) -> dict[str, int | None]:
    prompt_tokens = 0
    response_tokens = 0
    trainable_tokens = 0
    saw_tokens = False
    for turn in _assistant_turns(trajectory):
        metadata = turn.metadata or {}
        prompt_ids = metadata.get("prompt_ids")
        response_ids = metadata.get("response_ids")
        if isinstance(prompt_ids, list):
            prompt_tokens += len(prompt_ids)
            saw_tokens = True
        if isinstance(response_ids, list):
            response_tokens += len(response_ids)
            saw_tokens = True
            if bool(metadata.get("trainable", False)):
                trainable_tokens += len(response_ids)
    if not saw_tokens:
        return {"prompt_tokens": None, "response_tokens": None, "trainable_tokens": None}
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "trainable_tokens": trainable_tokens,
    }


def _trajectory_row(
    *,
    sample: RolloutProbeSample,
    trajectory: AgentTrajectory,
    temperature: float,
    rollout_index: int,
    elapsed_seconds: float,
    include_turns: bool,
) -> dict[str, Any]:
    metadata = dict(trajectory.metadata or {})
    execute_turns = [turn for turn in trajectory.turns if turn.metadata.get("agent_step") == "execute_query"]
    assistant_turns = _assistant_turns(trajectory)
    sql_action_turns = [turn for turn in assistant_turns if bool(turn.metadata.get("trainable", False))]
    checker_turns = [turn for turn in assistant_turns if turn.metadata.get("agent_step") == "check_query"]
    any_execution_ok = any(bool(turn.metadata.get("ok")) for turn in execute_turns)
    no_parseable_sql = bool(metadata.get("no_parseable_sql", False))
    final_sql = trajectory.final_sql or ""
    reward = float(trajectory.reward or 0.0)
    row = {
        "temperature": temperature,
        "uid": sample.example.uid,
        "db_id": sample.example.db_id,
        "question": sample.example.question,
        "rollout_id": trajectory.rollout_id,
        "rollout_index": rollout_index,
        "reward": reward,
        "final_sql": final_sql,
        "final_sql_source": trajectory.final_sql_source,
        "any_execution_ok": any_execution_ok,
        "executable_but_incorrect": any_execution_ok and reward == 0.0,
        "invalid_sql": not any_execution_ok,
        "empty_sql": no_parseable_sql or not final_sql.strip(),
        "num_assistant_turns": len(assistant_turns),
        "num_sql_actions": len(sql_action_turns),
        "num_checker_actions": len(checker_turns),
        "elapsed_seconds": elapsed_seconds,
        **metadata,
        **_token_stats(trajectory),
    }
    if include_turns:
        row["turns"] = [
            {"role": turn.role, "content": turn.content, "metadata": turn.metadata} for turn in trajectory.turns
        ]
    return row


def _run_one_rollout(
    *,
    loop: SqlAgentLoop,
    client: ModelClient | GoldSqlClient,
    sample: RolloutProbeSample,
    temperature: float,
    rollout_index: int,
    max_tokens_per_call: int,
    top_p: float | None,
    top_k: int | None,
) -> tuple[AgentTrajectory, float]:
    rollout_id = f"{sample.example.uid}:temp={temperature}:rollout={rollout_index}"
    agent_input = _agent_input(sample, rollout_id)
    start = time.perf_counter()
    if isinstance(client, GoldSqlClient):
        trajectory = loop.run_with_responses(agent_input, [sample.example.gold_sql], sample.sqlite_path)
    else:
        trajectory = loop.run(
            agent_input,
            client,
            sample.sqlite_path,
            max_tokens=max_tokens_per_call,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    return trajectory, time.perf_counter() - start


def _variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.pvariance(values)


def _average_optional(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _summarize_temperature(
    *,
    temperature: float,
    rows: list[dict[str, Any]],
    tasks: int,
    rollout_n: int,
    elapsed_seconds: float,
    rollouts_jsonl: Path,
) -> TemperatureMetrics:
    rewards = [float(row["reward"]) for row in rows]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["uid"])].append(float(row["reward"]))

    variances = [_variance(values) for values in grouped.values()]
    nonzero_groups = sum(1 for value in variances if value > 0.0)
    zero_groups = len(variances) - nonzero_groups
    trajectories = len(rows)
    return TemperatureMetrics(
        temperature=temperature,
        tasks=tasks,
        trajectories=trajectories,
        rollout_n=rollout_n,
        mean_reward=float(sum(rewards) / len(rewards)) if rewards else 0.0,
        reward_std=float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
        reward_min=float(min(rewards)) if rewards else 0.0,
        reward_max=float(max(rewards)) if rewards else 0.0,
        nonzero_variance_groups=nonzero_groups,
        zero_variance_groups=zero_groups,
        nonzero_variance_group_ratio=nonzero_groups / len(variances) if variances else 0.0,
        zero_variance_group_ratio=zero_groups / len(variances) if variances else 0.0,
        reward_variance_mean=float(sum(variances) / len(variances)) if variances else 0.0,
        sqlite_execution_success_rate=sum(bool(row["any_execution_ok"]) for row in rows) / trajectories,
        executable_but_incorrect_rate=sum(bool(row["executable_but_incorrect"]) for row in rows) / trajectories,
        invalid_sql_rate=sum(bool(row["invalid_sql"]) for row in rows) / trajectories,
        empty_sql_rate=sum(bool(row["empty_sql"]) for row in rows) / trajectories,
        average_num_turns=sum(int(row.get("num_sql_actions", 0)) for row in rows) / trajectories,
        average_execute_calls=sum(int(row.get("num_execute_calls", 0)) for row in rows) / trajectories,
        average_check_calls=sum(int(row.get("num_check_calls", 0)) for row in rows) / trajectories,
        average_parse_errors=sum(int(row.get("num_parse_errors", 0)) for row in rows) / trajectories,
        average_prompt_tokens=_average_optional(rows, "prompt_tokens"),
        average_response_tokens=_average_optional(rows, "response_tokens"),
        average_trainable_tokens=_average_optional(rows, "trainable_tokens"),
        elapsed_seconds=elapsed_seconds,
        rollouts_jsonl=str(rollouts_jsonl),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _temperature_slug(value: float) -> str:
    return str(value).replace(".", "p")


def _write_summary_markdown(path: Path, metrics: list[TemperatureMetrics]) -> None:
    lines = [
        "# Rollout Temperature Probe",
        "",
        "| temperature | nonzero variance | mean reward | reward std | invalid SQL | empty SQL | exec-but-wrong | elapsed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in metrics:
        lines.append(
            "| "
            f"{item.temperature:g} | "
            f"{item.nonzero_variance_group_ratio:.3f} | "
            f"{item.mean_reward:.3f} | "
            f"{item.reward_std:.3f} | "
            f"{item.invalid_sql_rate:.3f} | "
            f"{item.empty_sql_rate:.3f} | "
            f"{item.executable_but_incorrect_rate:.3f} | "
            f"{item.elapsed_seconds:.1f}s |"
        )
    lines.extend(
        [
            "",
            "Selection rule: choose the lowest temperature with enough nonzero reward variance and without a sharp "
            "increase in invalid SQL, empty SQL, or executable-but-wrong trajectories.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_progress(message: str) -> None:
    print(message, flush=True)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = _resolve_project_path(args.data_dir)
    output_dir = _resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(
        data_dir=data_dir,
        split_file=args.split_file,
        limit=args.limit,
        offset=args.offset,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    client = _build_model_client(args)
    loop = SqlAgentLoop(max_turns=args.max_turns)
    all_metrics: list[TemperatureMetrics] = []

    for temperature in args.temperatures:
        _print_progress(f"temperature={temperature:g} tasks={len(samples)} rollout_n={args.rollout_n}")
        start = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for sample_index, sample in enumerate(samples, start=1):
            for rollout_index in range(args.rollout_n):
                trajectory, elapsed = _run_one_rollout(
                    loop=loop,
                    client=client,
                    sample=sample,
                    temperature=temperature,
                    rollout_index=rollout_index,
                    max_tokens_per_call=args.max_tokens_per_call,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                rows.append(
                    _trajectory_row(
                        sample=sample,
                        trajectory=trajectory,
                        temperature=temperature,
                        rollout_index=rollout_index,
                        elapsed_seconds=elapsed,
                        include_turns=args.include_turns,
                    )
                )
            if sample_index % args.log_every == 0 or sample_index == len(samples):
                _print_progress(f"  completed {sample_index}/{len(samples)} tasks at temperature={temperature:g}")

        rollouts_path = output_dir / f"temperature_{_temperature_slug(temperature)}_rollouts.jsonl"
        metrics_path = output_dir / f"temperature_{_temperature_slug(temperature)}_metrics.json"
        _write_jsonl(rollouts_path, rows)
        metrics = _summarize_temperature(
            temperature=temperature,
            rows=rows,
            tasks=len(samples),
            rollout_n=args.rollout_n,
            elapsed_seconds=time.perf_counter() - start,
            rollouts_jsonl=rollouts_path,
        )
        all_metrics.append(metrics)
        _write_json(metrics_path, asdict(metrics))
        _print_progress(
            "  "
            f"nonzero_variance={metrics.nonzero_variance_group_ratio:.3f} "
            f"mean_reward={metrics.mean_reward:.3f} "
            f"invalid_sql={metrics.invalid_sql_rate:.3f} "
            f"empty_sql={metrics.empty_sql_rate:.3f}"
        )

    summary = {
        "data_dir": str(data_dir),
        "split_file": args.split_file,
        "limit": args.limit,
        "offset": args.offset,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "backend": args.backend,
        "model_path": args.model_path,
        "model_name": args.model_name,
        "lora_path": args.lora_path,
        "rollout_n": args.rollout_n,
        "max_turns": args.max_turns,
        "max_tokens_per_call": args.max_tokens_per_call,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperatures": [asdict(item) for item in all_metrics],
        "reward_counts_by_temperature": {
            str(item.temperature): dict(
                Counter(
                    str(row["reward"])
                    for row in _read_jsonl(output_dir / f"temperature_{_temperature_slug(item.temperature)}_rollouts.jsonl")
                )
            )
            for item in all_metrics
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_summary_markdown(output_dir / "summary.md", all_metrics)
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["vllm", "hf", "gold"], default="vllm")
    parser.add_argument("--model-path", default="data/models/Qwen2.5-Coder-14B-Instruct-SFT-Merged")
    parser.add_argument("--model-name", default=None, help="OpenAI/vLLM model name. Defaults to model path or LoRA name.")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path for chat-template formatting and token stats.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--lora-path", default=None, help="LoRA adapter path. vLLM loads it dynamically; HF loads it via PEFT.")
    parser.add_argument("--lora-name", default="temperature_probe_lora")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--data-dir", default="data/spider")
    parser.add_argument("--split-file", default="train_spider.json")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-tokens-per-call", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)

    parser.add_argument("--output-dir", default="artifacts/rollout_temperature_probe")
    parser.add_argument("--include-turns", action="store_true", help="Store full prompt/response/tool text in JSONL.")
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.rollout_n <= 0:
        raise ValueError("--rollout-n must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    summary = run_probe(args)
    print(json.dumps({"summary_json": str(_resolve_project_path(args.output_dir) / "summary.json")}, indent=2))
    best = max(summary["temperatures"], key=lambda item: item["nonzero_variance_group_ratio"])
    print(
        "best_by_nonzero_variance="
        f"{best['temperature']} ratio={best['nonzero_variance_group_ratio']:.3f} "
        f"invalid_sql={best['invalid_sql_rate']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
