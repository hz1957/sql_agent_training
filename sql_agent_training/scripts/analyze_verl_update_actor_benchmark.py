#!/usr/bin/env python3
"""Summarize verl timing and memory for update_actor batching experiments."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEP_RE = re.compile(r"\bstep:(\d+)\s+-")
GPU_MONITOR_RE = re.compile(r"\bGPU_MONITOR\s+[^,]+,\s*\d+,\s*(\d+),\s*(\d+),\s*(\d+)")
BENCHMARK_CONFIG_RE = re.compile(r"\b(?:UPDATE_ACTOR|ROLLOUT_CONCURRENCY)_BENCHMARK_CONFIG\s+(.*)")

METRICS = {
    "step": "timing_s/step",
    "rollout": "timing_s/gen",
    "old_log_prob": "timing_s/old_log_prob",
    "reference": "timing_s/ref",
    "update_actor": "timing_s/update_actor",
    "update_weights": "timing_s/update_weights",
    "throughput": "perf/throughput",
    "tokens": "perf/total_num_tokens",
    "prompt_length": "prompt_length/mean",
    "response_length": "response_length/mean",
    "actor_mfu": "perf/mfu/actor",
    "cpu_memory_gb": "actor/perf/cpu_memory_used_gb",
    "gpu_allocated_gb": "actor/perf/max_memory_allocated_gb",
    "gpu_reserved_gb": "actor/perf/max_memory_reserved_gb",
}
METRIC_PATTERNS = {
    name: re.compile(rf"{re.escape(log_key)}:(?:np\.\w+\()?({NUMBER})\)?") for name, log_key in METRICS.items()
}
CONFIG_PREFIXES = (
    "verl TRAIN_BATCH_SIZE=",
    "verl LOG_PROB_USE_DYNAMIC_BSZ=",
    "verl MODEL_USE_REMOVE_PADDING=",
    "verl MAX_PROMPT_LENGTH=",
    "verl ROLLOUT_MAX_NUM_BATCHED_TOKENS=",
)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))]


def _mean(summary: dict[str, Any], metric: str) -> float | None:
    values = [record[metric] for record in summary["measured"] if metric in record]
    return statistics.fmean(values) if values else None


def _metric_values(summary: dict[str, Any], metric: str) -> list[float]:
    return [record[metric] for record in summary["measured"] if metric in record]


def parse_log(path: Path, warmup_steps: int) -> dict[str, Any]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    records: list[dict[str, float]] = []
    config_lines: list[str] = []
    benchmark_config = ""

    for line in text.splitlines():
        if not benchmark_config:
            match = BENCHMARK_CONFIG_RE.search(line)
            if match:
                benchmark_config = match.group(1)
        for prefix in CONFIG_PREFIXES:
            if prefix in line and line.strip() not in config_lines:
                config_lines.append(line.strip())

        step_match = STEP_RE.search(line)
        if step_match is None or "timing_s/step:" not in line:
            continue
        record: dict[str, float] = {"global_step": float(step_match.group(1))}
        for name, pattern in METRIC_PATTERNS.items():
            match = pattern.search(line)
            if match:
                record[name] = float(match.group(1))
        if "update_actor" in record and "tokens" in record and record["tokens"] > 0:
            record["update_actor_s_per_1k_tokens"] = record["update_actor"] / record["tokens"] * 1000.0
        records.append(record)

    records.sort(key=lambda record: record["global_step"])
    measured = records[warmup_steps:]
    if not measured:
        raise ValueError(
            f"{path}: found {len(records)} completed steps; warmup_steps={warmup_steps} leaves no measured steps"
        )

    gpu_samples = [int(match.group(1)) for match in GPU_MONITOR_RE.finditer(text)]
    return {
        "path": path,
        "benchmark_config": benchmark_config,
        "config_lines": config_lines,
        "records": records,
        "measured": measured,
        "gpu_monitor_peak_mb": max(gpu_samples) if gpu_samples else None,
    }


def print_summary(summary: dict[str, Any], warmup_steps: int) -> None:
    print(f"log: {summary['path']}")
    if summary["benchmark_config"]:
        print(f"benchmark_config: {summary['benchmark_config']}")
    for line in summary["config_lines"]:
        print(line)
    print(
        f"completed_steps: {len(summary['records'])}  "
        f"warmup_removed: {warmup_steps}  measured_steps: {len(summary['measured'])}"
    )
    print()
    print(f"{'metric':30} {'mean':>12} {'median':>12} {'P95/peak':>12}")
    print("-" * 70)

    for metric in (
        "step",
        "rollout",
        "old_log_prob",
        "reference",
        "update_actor",
        "update_actor_s_per_1k_tokens",
        "update_weights",
        "throughput",
        "tokens",
        "prompt_length",
        "response_length",
        "actor_mfu",
    ):
        values = _metric_values(summary, metric)
        if values:
            print(
                f"{metric:30} {statistics.fmean(values):12.3f} "
                f"{statistics.median(values):12.3f} {_p95(values):12.3f}"
            )

    for metric in ("cpu_memory_gb", "gpu_allocated_gb", "gpu_reserved_gb"):
        values = _metric_values(summary, metric)
        if values:
            print(f"{metric:30} {'-':>12} {'-':>12} {max(values):12.3f}")
    if summary["gpu_monitor_peak_mb"] is not None:
        print(f"{'gpu_monitor_mb':30} {'-':>12} {'-':>12} {summary['gpu_monitor_peak_mb']:12d}")


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    if len(summaries) < 2:
        return
    baseline = summaries[0]
    print()
    print(f"Comparison relative to first log: {baseline['path'].name}")
    print("-" * 70)
    print(f"{'metric':30} {'baseline':>12} {'candidate':>12} {'delta%':>10}")
    for candidate in summaries[1:]:
        print(f"candidate: {candidate['path'].name}")
        for metric in (
            "step",
            "rollout",
            "update_actor",
            "update_actor_s_per_1k_tokens",
            "update_weights",
            "old_log_prob",
            "reference",
            "throughput",
        ):
            base_value = _mean(baseline, metric)
            candidate_value = _mean(candidate, metric)
            if base_value is None or candidate_value is None or base_value == 0:
                continue
            delta = (candidate_value - base_value) / base_value * 100.0
            print(f"{metric:30} {base_value:12.3f} {candidate_value:12.3f} {delta:10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")

    summaries = [parse_log(path, args.warmup_steps) for path in args.logs]
    for index, summary in enumerate(summaries):
        if index:
            print()
        print_summary(summary, args.warmup_steps)
    print_comparison(summaries)


if __name__ == "__main__":
    main()
