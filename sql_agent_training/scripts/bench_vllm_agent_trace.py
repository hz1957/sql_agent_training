#!/usr/bin/env python3
"""Benchmark complete SQL agent traces against OpenAI-compatible vLLM servers."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_agent_training.agent.model_client import VllmOpenAIModelClient  # noqa: E402
from sql_agent_training.agent.sql_agent_loop import SqlAgentInput, SqlAgentLoop  # noqa: E402


@dataclass(frozen=True)
class AgentTraceSpec:
    """Environment fields required to run one SQL agent trajectory."""

    uid: str
    question: str
    db_id: str
    schema_prompt: str
    gold_sql: str
    sqlite_path: Path


@dataclass(frozen=True)
class AgentTraceResult:
    """Per-trajectory timing, token, and behavior measurements."""

    request_id: str
    uid: str
    base_url: str
    ok: bool
    elapsed_s: float
    reward: float | None
    executable: bool
    model_calls: int
    sql_actions: int
    execute_calls: int
    parse_errors: int
    rewrite: bool
    ran_out_of_turns: bool
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    final_sql_source: str | None
    error: str | None = None


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_tokenizer(tokenizer_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(_resolve_path(tokenizer_path)),
        trust_remote_code=True,
        fix_mistral_regex=True,
    )


def _resolve_sqlite_path(raw_path: str, *, data_dir: Path, db_id: str) -> Path:
    candidates = [Path(raw_path)] if raw_path else []
    candidates.append(data_dir / "database" / db_id / f"{db_id}.sqlite")
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else ROOT / candidate
        if resolved.exists():
            return resolved
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing SQLite database for {db_id}; checked: {rendered}")


def _load_specs(
    *,
    dataset_parquet: Path,
    data_dir: Path,
    limit: int,
    seed: int,
    shuffle: bool,
) -> list[AgentTraceSpec]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pyarrow is required to read verl parquet samples.") from exc

    if not dataset_parquet.exists():
        raise FileNotFoundError(f"Missing dataset parquet: {dataset_parquet}")
    rows = pq.read_table(dataset_parquet, columns=["extra_info"]).to_pylist()
    if shuffle:
        random.Random(seed).shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows loaded from {dataset_parquet}")

    specs: list[AgentTraceSpec] = []
    for row_index, row in enumerate(rows):
        extra = row.get("extra_info") or {}
        db_id = str(extra.get("db_id") or "")
        required = {
            "question": str(extra.get("question") or ""),
            "schema_prompt": str(extra.get("schema_prompt") or ""),
            "gold_sql": str(extra.get("gold_sql") or ""),
        }
        missing = [key for key, value in required.items() if not value]
        if not db_id:
            missing.append("db_id")
        if missing:
            raise ValueError(f"Row {row_index} is missing agent fields: {', '.join(missing)}")
        specs.append(
            AgentTraceSpec(
                uid=str(extra.get("uid") or f"row-{row_index}"),
                question=required["question"],
                db_id=db_id,
                schema_prompt=required["schema_prompt"],
                gold_sql=required["gold_sql"],
                sqlite_path=_resolve_sqlite_path(
                    str(extra.get("sqlite_path") or ""),
                    data_dir=data_dir,
                    db_id=db_id,
                ),
            )
        )
    return specs


def _trajectory_token_counts(turns: list[Any]) -> tuple[int, int, int]:
    prompt_tokens = 0
    output_tokens = 0
    for turn in turns:
        if turn.role != "assistant":
            continue
        prompt_ids = turn.metadata.get("prompt_ids")
        response_ids = turn.metadata.get("response_ids")
        if isinstance(prompt_ids, list):
            prompt_tokens += len(prompt_ids)
        if isinstance(response_ids, list):
            output_tokens += len(response_ids)
    return prompt_tokens, output_tokens, prompt_tokens + output_tokens


def _run_trace(
    *,
    spec: AgentTraceSpec,
    request_index: int,
    base_url: str,
    client: VllmOpenAIModelClient,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> AgentTraceResult:
    request_id = f"{spec.uid}:agent:{request_index}"
    sample = SqlAgentInput(
        uid=spec.uid,
        rollout_id=request_id,
        question=spec.question,
        db_id=spec.db_id,
        schema_prompt=spec.schema_prompt,
        gold_sql=spec.gold_sql,
    )
    start = time.perf_counter()
    try:
        trajectory = SqlAgentLoop(max_turns=max_turns).run(
            sample,
            client,
            spec.sqlite_path,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        elapsed_s = time.perf_counter() - start
        assistant_turns = [turn for turn in trajectory.turns if turn.role == "assistant"]
        sql_actions = sum(bool(turn.metadata.get("trainable", False)) for turn in assistant_turns)
        prompt_tokens, output_tokens, total_tokens = _trajectory_token_counts(trajectory.turns)
        return AgentTraceResult(
            request_id=request_id,
            uid=spec.uid,
            base_url=base_url,
            ok=True,
            elapsed_s=elapsed_s,
            reward=float(trajectory.reward or 0.0),
            executable=trajectory.final_sql is not None,
            model_calls=len(assistant_turns),
            sql_actions=sql_actions,
            execute_calls=int(trajectory.metadata.get("num_execute_calls", 0)),
            parse_errors=int(trajectory.metadata.get("num_parse_errors", 0)),
            rewrite=sql_actions > 1,
            ran_out_of_turns=bool(trajectory.metadata.get("ran_out_of_turns", False)),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            final_sql_source=trajectory.final_sql_source,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark records per-trajectory failures.
        return AgentTraceResult(
            request_id=request_id,
            uid=spec.uid,
            base_url=base_url,
            ok=False,
            elapsed_s=time.perf_counter() - start,
            reward=None,
            executable=False,
            model_calls=0,
            sql_actions=0,
            execute_calls=0,
            parse_errors=0,
            rewrite=False,
            ran_out_of_turns=False,
            prompt_tokens=0,
            output_tokens=0,
            total_tokens=0,
            final_sql_source=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _gpu_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    by_gpu: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 5 or row[0].startswith("timestamp"):
                continue
            index = row[1].strip()
            try:
                memory_used_mb = float(row[2].strip())
                memory_total_mb = float(row[3].strip())
                gpu_utilization_pct = float(row[4].strip())
            except ValueError:
                continue
            item = by_gpu.setdefault(
                index,
                {"peak_memory_mb": 0.0, "memory_total_mb": memory_total_mb, "max_gpu_utilization_pct": 0.0},
            )
            item["peak_memory_mb"] = max(item["peak_memory_mb"], memory_used_mb)
            item["memory_total_mb"] = max(item["memory_total_mb"], memory_total_mb)
            item["max_gpu_utilization_pct"] = max(item["max_gpu_utilization_pct"], gpu_utilization_pct)
    if not by_gpu:
        return {}
    return {
        "gpu_monitor_csv": str(path),
        "peak_memory_mb_per_gpu": by_gpu,
        "peak_memory_mb_max": max(item["peak_memory_mb"] for item in by_gpu.values()),
        "peak_memory_mb_sum": sum(item["peak_memory_mb"] for item in by_gpu.values()),
    }


def _summarize(
    *,
    results: list[AgentTraceResult],
    wall_time_s: float,
    base_urls: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ok_results = [row for row in results if row.ok]
    latencies = [row.elapsed_s for row in ok_results]
    model_calls = sum(row.model_calls for row in ok_results)
    prompt_tokens = sum(row.prompt_tokens for row in ok_results)
    output_tokens = sum(row.output_tokens for row in ok_results)
    total_tokens = sum(row.total_tokens for row in ok_results)
    summary: dict[str, Any] = {
        "benchmark_kind": "agent_trace",
        "case_name": args.case_name,
        "model_name": args.model_name,
        "dataset_parquet": str(_resolve_path(args.dataset_parquet)),
        "data_dir": str(_resolve_path(args.data_dir)),
        "base_urls": base_urls,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "max_turns": args.max_turns,
        "max_tokens_per_call": args.max_tokens,
        "temperature": args.temperature,
        "trajectories_total": len(results),
        "trajectories_ok": len(ok_results),
        "trajectories_failed": len(results) - len(ok_results),
        "wall_time_s": wall_time_s,
        "trajectories_per_s": len(ok_results) / wall_time_s if wall_time_s > 0 else 0.0,
        "model_calls": model_calls,
        "model_calls_per_s": model_calls / wall_time_s if wall_time_s > 0 else 0.0,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "output_tokens_per_s": output_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        "total_tokens_per_s": total_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        "trajectory_latency_mean_s": statistics.fmean(latencies) if latencies else None,
        "trajectory_latency_p50_s": _percentile(latencies, 0.50),
        "trajectory_latency_p95_s": _percentile(latencies, 0.95),
        "trajectory_latency_p99_s": _percentile(latencies, 0.99),
        "avg_model_calls": statistics.fmean(row.model_calls for row in ok_results) if ok_results else None,
        "avg_sql_actions": statistics.fmean(row.sql_actions for row in ok_results) if ok_results else None,
        "avg_execute_calls": statistics.fmean(row.execute_calls for row in ok_results) if ok_results else None,
        "rewrite_rate": sum(row.rewrite for row in ok_results) / len(ok_results) if ok_results else None,
        "execution_accuracy": (
            sum(float(row.reward or 0.0) for row in ok_results) / len(results) if results else None
        ),
        "executable_rate": sum(row.executable for row in ok_results) / len(results) if results else None,
        "ran_out_of_turns_rate": (
            sum(row.ran_out_of_turns for row in ok_results) / len(ok_results) if ok_results else None
        ),
        "parse_error_rate": sum(row.parse_errors > 0 for row in ok_results) / len(ok_results) if ok_results else None,
    }
    summary.update(_gpu_summary(_resolve_path(args.gpu_monitor_csv) if args.gpu_monitor_csv else None))
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[AgentTraceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = _load_tokenizer(args.tokenizer_path)
    specs = _load_specs(
        dataset_parquet=_resolve_path(args.dataset_parquet),
        data_dir=_resolve_path(args.data_dir),
        limit=args.limit,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    base_urls = [url.rstrip("/") for url in args.base_urls]
    if not base_urls:
        raise ValueError("--base-urls must not be empty")
    clients = {
        base_url: VllmOpenAIModelClient(
            base_url=base_url,
            model_name=args.model_name,
            tokenizer=tokenizer,
            timeout_seconds=args.timeout_seconds,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        for base_url in base_urls
    }

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[AgentTraceResult] = []
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for index, spec in enumerate(specs):
            base_url = base_urls[index % len(base_urls)]
            futures.append(
                executor.submit(
                    _run_trace,
                    spec=spec,
                    request_index=index,
                    base_url=base_url,
                    client=clients[base_url],
                    max_turns=args.max_turns,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
            )
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if completed % args.log_every == 0 or completed == len(futures):
                print(
                    f"completed={completed}/{len(futures)} "
                    f"ok={sum(row.ok for row in results)} failed={sum(not row.ok for row in results)}",
                    flush=True,
                )
    wall_time_s = time.perf_counter() - start
    results.sort(key=lambda row: row.request_id)
    summary = _summarize(results=results, wall_time_s=wall_time_s, base_urls=base_urls, args=args)
    _write_jsonl(output_dir / "trajectories.jsonl", results)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--base-urls", nargs="+", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--dataset-parquet", default="data/verl_spider/validation.parquet")
    parser.add_argument("--data-dir", default="data/spider")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--gpu-monitor-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.max_turns <= 0:
        parser.error("--max-turns must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
