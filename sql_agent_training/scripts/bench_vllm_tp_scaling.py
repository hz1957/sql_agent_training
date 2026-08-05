#!/usr/bin/env python3
"""Benchmark OpenAI-compatible vLLM servers with fixed SQL-agent prompts."""

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
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_agent_training.agent.model_client import format_hf_prompt  # noqa: E402
from sql_agent_training.agent.trace_format import AgentTurn  # noqa: E402


@dataclass(frozen=True)
class PromptSpec:
    request_id: str
    uid: str
    prompt: str


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    uid: str
    base_url: str
    ok: bool
    latency_s: float
    ttft_s: float | None
    tpot_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_chars: int
    error: str | None = None


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_tokenizer(tokenizer_path: str | None) -> Any | None:
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(_resolve_path(tokenizer_path)), trust_remote_code=True)


def _messages_to_prompt(messages: list[dict[str, Any]], tokenizer: Any | None) -> str:
    turns = [
        AgentTurn(role=str(message.get("role", "user")), content=str(message.get("content", "")))
        for message in messages
    ]
    if tokenizer is not None:
        return format_hf_prompt(tokenizer, turns)
    lines = [f"{turn.role}: {turn.content}" for turn in turns]
    lines.append("assistant:")
    return "\n".join(lines)


def _load_parquet_prompts(
    *,
    dataset_parquet: Path,
    tokenizer: Any | None,
    limit: int,
    seed: int,
    shuffle: bool,
    repetitions: int,
) -> list[PromptSpec]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pyarrow is required to read verl parquet prompts.") from exc

    if not dataset_parquet.exists():
        raise FileNotFoundError(f"Missing dataset parquet: {dataset_parquet}")
    rows = pq.read_table(dataset_parquet, columns=["prompt", "extra_info"]).to_pylist()
    if shuffle:
        random.Random(seed).shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows loaded from {dataset_parquet}")

    prompts: list[PromptSpec] = []
    for repeat_index in range(repetitions):
        for row_index, row in enumerate(rows):
            messages = row.get("prompt")
            if not isinstance(messages, list):
                messages = [{"role": "user", "content": str(messages)}]
            extra_info = row.get("extra_info") or {}
            uid = str(extra_info.get("uid", f"row-{row_index}"))
            prompts.append(
                PromptSpec(
                    request_id=f"{uid}:rep{repeat_index}",
                    uid=uid,
                    prompt=_messages_to_prompt(messages, tokenizer),
                )
            )
    return prompts


def _post_json_streaming(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    api_key: str | None,
) -> tuple[str, dict[str, Any] | None, float | None]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    first_token_s: float | None = None
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.split("data:", maxsplit=1)[1].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices")
            if isinstance(choices, list) and choices:
                piece = str(choices[0].get("text", ""))
                if piece and first_token_s is None:
                    first_token_s = time.perf_counter() - start
                text_parts.append(piece)
    return "".join(text_parts), usage, first_token_s


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    api_key: str | None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _usage_tokens(usage: dict[str, Any] | None, prompt: str, response: str, tokenizer: Any | None) -> tuple[int | None, int | None, int | None]:
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    if usage:
        prompt_tokens = int(usage["prompt_tokens"]) if usage.get("prompt_tokens") is not None else None
        completion_tokens = int(usage["completion_tokens"]) if usage.get("completion_tokens") is not None else None
        total_tokens = int(usage["total_tokens"]) if usage.get("total_tokens") is not None else None
    if tokenizer is not None and (prompt_tokens is None or completion_tokens is None):
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        completion_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def _run_request(
    *,
    spec: PromptSpec,
    base_url: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    timeout_seconds: float,
    api_key: str | None,
    stream: bool,
    tokenizer: Any | None,
) -> RequestResult:
    endpoint = f"{base_url.rstrip('/')}/completions"
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": spec.prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if top_k is not None:
        payload["top_k"] = top_k
    if stream:
        payload["stream"] = True

    start = time.perf_counter()
    try:
        if stream:
            response_text, usage, ttft_s = _post_json_streaming(
                endpoint,
                payload,
                timeout_seconds=timeout_seconds,
                api_key=api_key,
            )
        else:
            response = _post_json(endpoint, payload, timeout_seconds=timeout_seconds, api_key=api_key)
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError(f"missing choices in response: {response!r}")
            response_text = str(choices[0].get("text", ""))
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
            ttft_s = None
        latency_s = time.perf_counter() - start
        prompt_tokens, completion_tokens, total_tokens = _usage_tokens(usage, spec.prompt, response_text, tokenizer)
        tpot_s = None
        if ttft_s is not None and completion_tokens and completion_tokens > 1:
            tpot_s = max(0.0, latency_s - ttft_s) / (completion_tokens - 1)
        return RequestResult(
            request_id=spec.request_id,
            uid=spec.uid,
            base_url=base_url,
            ok=True,
            latency_s=latency_s,
            ttft_s=ttft_s,
            tpot_s=tpot_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_chars=len(response_text),
        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        return RequestResult(
            request_id=spec.request_id,
            uid=spec.uid,
            base_url=base_url,
            ok=False,
            latency_s=time.perf_counter() - start,
            ttft_s=None,
            tpot_s=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            response_chars=0,
            error=str(exc),
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
        reader = csv.reader(handle)
        for row in reader:
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
    results: list[RequestResult],
    wall_time_s: float,
    base_urls: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ok_results = [row for row in results if row.ok]
    latencies = [row.latency_s for row in ok_results]
    ttfts = [row.ttft_s for row in ok_results if row.ttft_s is not None]
    tpots = [row.tpot_s for row in ok_results if row.tpot_s is not None]
    output_tokens = sum(row.completion_tokens or 0 for row in ok_results)
    total_tokens = sum(row.total_tokens or 0 for row in ok_results)
    summary = {
        "case_name": args.case_name,
        "model_name": args.model_name,
        "dataset_parquet": str(_resolve_path(args.dataset_parquet)),
        "base_urls": base_urls,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "repetitions": args.repetitions,
        "stream": args.stream,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "requests_total": len(results),
        "requests_ok": len(ok_results),
        "requests_failed": len(results) - len(ok_results),
        "wall_time_s": wall_time_s,
        "requests_per_s": len(ok_results) / wall_time_s if wall_time_s > 0 else 0.0,
        "output_tokens_per_s": output_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        "total_tokens_per_s": total_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_mean_s": statistics.fmean(latencies) if latencies else None,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "latency_p99_s": _percentile(latencies, 0.99),
        "ttft_mean_s": statistics.fmean(ttfts) if ttfts else None,
        "ttft_p50_s": _percentile(ttfts, 0.50),
        "ttft_p95_s": _percentile(ttfts, 0.95),
        "tpot_mean_s": statistics.fmean(tpots) if tpots else None,
        "tpot_p50_s": _percentile(tpots, 0.50),
        "tpot_p95_s": _percentile(tpots, 0.95),
    }
    summary.update(_gpu_summary(_resolve_path(args.gpu_monitor_csv) if args.gpu_monitor_csv else None))
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[RequestResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = _load_tokenizer(args.tokenizer_path)
    prompts = _load_parquet_prompts(
        dataset_parquet=_resolve_path(args.dataset_parquet),
        tokenizer=tokenizer,
        limit=args.limit,
        seed=args.seed,
        shuffle=args.shuffle,
        repetitions=args.repetitions,
    )
    base_urls = [url.rstrip("/") for url in args.base_urls]
    if not base_urls:
        raise ValueError("--base-urls must not be empty")

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[RequestResult] = []
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for index, prompt in enumerate(prompts):
            futures.append(
                executor.submit(
                    _run_request,
                    spec=prompt,
                    base_url=base_urls[index % len(base_urls)],
                    model_name=args.model_name,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k if args.top_k >= 0 else None,
                    timeout_seconds=args.timeout_seconds,
                    api_key=args.api_key,
                    stream=args.stream,
                    tokenizer=tokenizer,
                )
            )
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed % args.log_every == 0 or completed == len(futures):
                print(
                    f"completed={completed}/{len(futures)} ok={sum(row.ok for row in results)} "
                    f"failed={sum(not row.ok for row in results)}",
                    flush=True,
                )
    wall_time_s = time.perf_counter() - start
    summary = _summarize(results=results, wall_time_s=wall_time_s, base_urls=base_urls, args=args)
    _write_jsonl(output_dir / "requests.jsonl", results)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--base-urls", nargs="+", required=True)
    parser.add_argument("--model-name", default="qwen25-coder-14b-sql")
    parser.add_argument("--dataset-parquet", default="data/verl_spider/validation.parquet")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--stream", action="store_true", help="Measure TTFT/TPOT with streaming completions.")
    parser.add_argument("--gpu-monitor-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
