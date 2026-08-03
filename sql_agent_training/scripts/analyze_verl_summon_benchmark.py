#!/usr/bin/env python3
"""Summarize verl's native timing and memory metrics for summon A/B runs."""

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
MODE_RE = re.compile(r"\bSUMMON_BENCHMARK_CONFIG mode=(full|layered)\b")
GPU_MONITOR_RE = re.compile(r"\bGPU_MONITOR\s+[^,]+,\s*\d+,\s*(\d+),\s*(\d+),\s*(\d+)")
LAYERED_RESULT_RE = re.compile(r"\bLAYERED_SUMMON_PATCH_RESULT tensors=(\d+)\b")

METRICS = {
    "step": "timing_s/step",
    "update_weights": "timing_s/update_weights",
    "rollout": "timing_s/gen",
    "update_actor": "timing_s/update_actor",
    "throughput": "perf/throughput",
    "tokens": "perf/total_num_tokens",
    "response_length": "response_length/mean",
    "cpu_memory_gb": "actor/perf/cpu_memory_used_gb",
    "gpu_allocated_gb": "actor/perf/max_memory_allocated_gb",
    "gpu_reserved_gb": "actor/perf/max_memory_reserved_gb",
}
METRIC_PATTERNS = {
    name: re.compile(rf"{re.escape(log_key)}:(?:np\.\w+\()?({NUMBER})\)?") for name, log_key in METRICS.items()
}


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))]


def _infer_mode(path: Path, text: str) -> str:
    match = MODE_RE.search(text)
    if match:
        return match.group(1)
    lowered = path.name.lower()
    if "_layered_" in lowered:
        return "layered"
    if "_full_" in lowered:
        return "full"
    return "unknown"


def parse_log(path: Path, warmup_steps: int) -> dict[str, Any]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    records: list[dict[str, float]] = []
    for line in text.splitlines():
        step_match = STEP_RE.search(line)
        if step_match is None or "timing_s/step:" not in line:
            continue
        record: dict[str, float] = {"global_step": float(step_match.group(1))}
        for name, pattern in METRIC_PATTERNS.items():
            match = pattern.search(line)
            if match:
                record[name] = float(match.group(1))
        records.append(record)

    records.sort(key=lambda record: record["global_step"])
    measured = records[warmup_steps:]
    if not measured:
        raise ValueError(
            f"{path}: found {len(records)} completed steps; " f"warmup_steps={warmup_steps} leaves no measured steps"
        )

    gpu_samples = [int(match.group(1)) for match in GPU_MONITOR_RE.finditer(text)]
    layered_result_counts = [int(match.group(1)) for match in LAYERED_RESULT_RE.finditer(text)]
    return {
        "path": path,
        "mode": _infer_mode(path, text),
        "records": records,
        "measured": measured,
        "gpu_monitor_peak_mb": max(gpu_samples) if gpu_samples else None,
        "fallback_warnings": text.count("layered_summon returned empty"),
        "patch_enabled": "LAYERED_SUMMON_PATCH_ENABLED" in text,
        "layered_result_counts": layered_result_counts,
    }


def _metric_values(summary: dict[str, Any], metric: str) -> list[float]:
    return [record[metric] for record in summary["measured"] if metric in record]


def print_summary(summary: dict[str, Any], warmup_steps: int) -> None:
    print(f"log: {summary['path']}")
    print(
        f"mode: {summary['mode']}  completed_steps: {len(summary['records'])}  "
        f"warmup_removed: {warmup_steps}  measured_steps: {len(summary['measured'])}"
    )
    print(
        f"patch_enabled: {summary['patch_enabled']}  "
        f"layered_empty_fallback_warnings: {summary['fallback_warnings']}"
    )
    if summary["layered_result_counts"]:
        counts = summary["layered_result_counts"]
        print(f"layered_lora_tensors_per_worker: range={min(counts)}-{max(counts)} workers={len(counts)}")
    print()
    print(f"{'metric':20} {'mean':>12} {'median':>12} {'P95/peak':>12}")
    print("-" * 60)
    for metric in ("step", "update_weights", "rollout", "update_actor", "throughput", "tokens", "response_length"):
        values = _metric_values(summary, metric)
        if values:
            print(
                f"{metric:20} {statistics.fmean(values):12.3f} "
                f"{statistics.median(values):12.3f} {_p95(values):12.3f}"
            )
    for metric in ("cpu_memory_gb", "gpu_allocated_gb", "gpu_reserved_gb"):
        values = _metric_values(summary, metric)
        if values:
            print(f"{metric:20} {'-':>12} {'-':>12} {max(values):12.3f}")
    if summary["gpu_monitor_peak_mb"] is not None:
        print(f"{'gpu_monitor_mb':20} {'-':>12} {'-':>12} {summary['gpu_monitor_peak_mb']:12d}")


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    by_mode = {summary["mode"]: summary for summary in summaries}
    if "full" not in by_mode or "layered" not in by_mode:
        return

    full = by_mode["full"]
    layered = by_mode["layered"]
    print()
    print("A/B comparison: layered relative to full")
    print("-" * 60)
    for metric in ("step", "update_weights"):
        full_values = _metric_values(full, metric)
        layered_values = _metric_values(layered, metric)
        if full_values and layered_values:
            full_mean = statistics.fmean(full_values)
            layered_mean = statistics.fmean(layered_values)
            reduction = 100.0 * (full_mean - layered_mean) / full_mean
            print(f"{metric:20} full={full_mean:.3f}s  " f"layered={layered_mean:.3f}s  reduction={reduction:.1f}%")
    full_throughput = _metric_values(full, "throughput")
    layered_throughput = _metric_values(layered, "throughput")
    if full_throughput and layered_throughput:
        full_mean = statistics.fmean(full_throughput)
        layered_mean = statistics.fmean(layered_throughput)
        gain = 100.0 * (layered_mean - full_mean) / full_mean
        print(f"{'throughput':20} full={full_mean:.3f}  " f"layered={layered_mean:.3f}  gain={gain:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=5)
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
