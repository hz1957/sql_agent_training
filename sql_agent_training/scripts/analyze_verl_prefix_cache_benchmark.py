#!/usr/bin/env python3
"""Summarize and compare S3 vLLM prefix-cache benchmark logs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEP_RE = re.compile(r"\bstep:(\d+)\s+-")
MODE_RE = re.compile(r"\bPREFIX_CACHE_BENCHMARK_CONFIG mode=(off|on)\b")
CONFIG_RE = re.compile(r"\bPREFIX_CACHE_BENCHMARK_CONFIG\s+(.+)$", re.MULTILINE)
CONFIG_ITEM_RE = re.compile(r"(\w+)=([^\s]+)")
GPU_MONITOR_RE = re.compile(r"\bGPU_MONITOR\s+[^,]+,\s*\d+,\s*(\d+),\s*(\d+),\s*(\d+)")
VLLM_STATS_RE = re.compile(
    rf"Avg prompt throughput:\s*({NUMBER})\s*tokens/s,\s*"
    rf"Avg generation throughput:\s*({NUMBER})\s*tokens/s,.*?"
    rf"GPU KV cache usage:\s*({NUMBER})%,\s*"
    rf"Prefix cache hit rate:\s*({NUMBER})%"
)
PREFIX_CACHE_PATCH_RE = re.compile(r"\bPREFIX_CACHE_PATCH applied\b.*?\bengine_after=(True|False)\b")
ENGINE_PREFIX_CACHE_RE = re.compile(r"\bInitializing a V1 LLM engine\b.*?\benable_prefix_caching=(True|False)\b")
FINGERPRINT_MARKER = "PREFIX_CACHE_WORKLOAD_FINGERPRINT "

METRICS = {
    "step": "timing_s/step",
    "rollout": "timing_s/gen",
    "update_weights": "timing_s/update_weights",
    "update_actor": "timing_s/update_actor",
    "throughput": "perf/throughput",
    "tokens": "perf/total_num_tokens",
    "prompt_length": "prompt_length/mean",
    "response_length": "response_length/mean",
}
METRIC_PATTERNS = {
    name: re.compile(rf"{re.escape(log_key)}:(?:np\.\w+\()?({NUMBER})\)?") for name, log_key in METRICS.items()
}


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))]


def _mean(summary: dict[str, Any], metric: str) -> float | None:
    values = [record[metric] for record in summary["measured"] if metric in record]
    return statistics.fmean(values) if values else None


def _infer_mode(path: Path, text: str) -> str:
    match = MODE_RE.search(text)
    if match:
        return match.group(1)
    lowered = path.name.lower()
    if "_prefix_cache_on_" in lowered:
        return "on"
    if "_prefix_cache_off_" in lowered:
        return "off"
    return "unknown"


def parse_log(path: Path, warmup_steps: int) -> dict[str, Any]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    config_match = CONFIG_RE.search(text)
    config = dict(CONFIG_ITEM_RE.findall(config_match.group(1))) if config_match else {}

    records: list[dict[str, float]] = []
    vllm_stats: list[dict[str, float]] = []
    fingerprints: list[dict[str, Any]] = []
    prefix_cache_patch_values: list[bool] = []
    engine_prefix_cache_values: list[bool] = []
    malformed_fingerprints = 0
    completed_steps = 0
    for line in text.splitlines():
        if FINGERPRINT_MARKER in line:
            payload = line.split(FINGERPRINT_MARKER, maxsplit=1)[1].strip()
            try:
                fingerprint = json.loads(payload)
            except json.JSONDecodeError:
                malformed_fingerprints += 1
            else:
                if isinstance(fingerprint, dict):
                    fingerprints.append(fingerprint)
                else:
                    malformed_fingerprints += 1
            continue

        patch_match = PREFIX_CACHE_PATCH_RE.search(line)
        if patch_match is not None:
            prefix_cache_patch_values.append(patch_match.group(1) == "True")

        engine_prefix_match = ENGINE_PREFIX_CACHE_RE.search(line)
        if engine_prefix_match is not None:
            engine_prefix_cache_values.append(engine_prefix_match.group(1) == "True")

        step_match = STEP_RE.search(line)
        if step_match is not None and "timing_s/step:" in line:
            record: dict[str, float] = {"global_step": float(step_match.group(1))}
            for name, pattern in METRIC_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    record[name] = float(match.group(1))
            records.append(record)
            completed_steps += 1
            continue

        stats_match = VLLM_STATS_RE.search(line)
        if stats_match is not None and completed_steps >= warmup_steps:
            prompt_tps, generation_tps, kv_usage, cache_hit_rate = map(float, stats_match.groups())
            if prompt_tps > 0 or generation_tps > 0:
                vllm_stats.append(
                    {
                        "prompt_tps": prompt_tps,
                        "generation_tps": generation_tps,
                        "kv_usage_pct": kv_usage,
                        "cache_hit_rate_pct": cache_hit_rate,
                    }
                )

    records.sort(key=lambda record: record["global_step"])
    measured = records[warmup_steps:]
    if not measured:
        raise ValueError(
            f"{path}: found {len(records)} completed steps; " f"warmup_steps={warmup_steps} leaves no measured steps"
        )

    gpu_samples = [int(match.group(1)) for match in GPU_MONITOR_RE.finditer(text)]
    return {
        "path": path,
        "mode": _infer_mode(path, text),
        "config": config,
        "records": records,
        "measured": measured,
        "vllm_stats": vllm_stats,
        "fingerprints": fingerprints,
        "prefix_cache_patch_values": prefix_cache_patch_values,
        "engine_prefix_cache_values": engine_prefix_cache_values,
        "malformed_fingerprints": malformed_fingerprints,
        "gpu_monitor_peak_mb": max(gpu_samples) if gpu_samples else None,
        "exit_success": "EXIT_CODE 0" in text,
    }


def _print_stats_row(name: str, values: list[float]) -> None:
    if values:
        print(f"{name:24} {statistics.fmean(values):12.3f} " f"{statistics.median(values):12.3f} {_p95(values):12.3f}")


def print_summary(summary: dict[str, Any], warmup_steps: int) -> None:
    print(f"log: {summary['path']}")
    print(
        f"mode: {summary['mode']}  exit_success: {summary['exit_success']}  "
        f"completed_steps: {len(summary['records'])}  warmup_removed: {warmup_steps}  "
        f"measured_steps: {len(summary['measured'])}  fingerprints: {len(summary['fingerprints'])}"
    )
    if summary["malformed_fingerprints"]:
        print(f"WARNING: malformed workload fingerprints: {summary['malformed_fingerprints']}")
    print()
    print(f"{'metric':24} {'mean':>12} {'median':>12} {'P95/peak':>12}")
    print("-" * 64)
    for metric in METRICS:
        values = [record[metric] for record in summary["measured"] if metric in record]
        _print_stats_row(metric, values)

    stats = summary["vllm_stats"]
    for metric in ("prompt_tps", "generation_tps", "kv_usage_pct", "cache_hit_rate_pct"):
        _print_stats_row(f"vllm_{metric}", [sample[metric] for sample in stats])
    if not stats:
        print(f"{'vllm_log_samples':24} {'0':>12} {'-':>12} {'-':>12}")
    if summary["gpu_monitor_peak_mb"] is not None:
        print(f"{'gpu_monitor_mb':24} {'-':>12} {'-':>12} {summary['gpu_monitor_peak_mb']:12d}")


def _percent_change(baseline: float, candidate: float) -> float:
    return 100.0 * (candidate - baseline) / baseline


def _compare_configs(off: dict[str, Any], on: dict[str, Any]) -> list[str]:
    ignored = {"mode", "enable_prefix_caching"}
    keys = (set(off["config"]) | set(on["config"])) - ignored
    return [key for key in sorted(keys) if off["config"].get(key) != on["config"].get(key)]


def _fingerprint_counter(summary: dict[str, Any], fields: tuple[str, ...]) -> Counter[tuple[Any, ...]]:
    return Counter(tuple(fingerprint.get(field) for field in fields) for fingerprint in summary["fingerprints"])


def _last_prefix_cache_value(summary: dict[str, Any]) -> bool | None:
    values = summary["prefix_cache_patch_values"] or summary["engine_prefix_cache_values"]
    return values[-1] if values else None


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    by_mode = {summary["mode"]: summary for summary in summaries}
    if "off" not in by_mode or "on" not in by_mode:
        return

    off = by_mode["off"]
    on = by_mode["on"]
    config_differences = _compare_configs(off, on)
    config_present = bool(off["config"]) and bool(on["config"])
    toggle_valid = (
        off["config"].get("enable_prefix_caching") == "False" and on["config"].get("enable_prefix_caching") == "True"
    )
    stable_seeds_valid = (
        off["config"].get("stable_request_seeds") == "True" and on["config"].get("stable_request_seeds") == "True"
    )
    batch_invariant_consistent = off["config"].get("batch_invariant") == on["config"].get("batch_invariant")
    frozen_actor_valid = (
        off["config"].get("frozen_actor") == "True"
        and on["config"].get("frozen_actor") == "True"
        and off["config"].get("actor_lr") == "0"
        and on["config"].get("actor_lr") == "0"
    )
    fingerprint_config_valid = (
        off["config"].get("workload_fingerprint") == "True" and on["config"].get("workload_fingerprint") == "True"
    )
    controlled_config_match = (
        config_present
        and toggle_valid
        and stable_seeds_valid
        and batch_invariant_consistent
        and frozen_actor_valid
        and fingerprint_config_valid
        and not config_differences
    )

    fingerprints_present = bool(off["fingerprints"]) and bool(on["fingerprints"])
    fingerprints_well_formed = off["malformed_fingerprints"] == 0 and on["malformed_fingerprints"] == 0
    prompt_fields = ("root_uid", "requests", "prompt_tokens", "prompt_sha256")
    response_fields = ("root_uid", "requests", "response_tokens", "response_sha256")
    request_fields = (
        "root_uid",
        "requests",
        "prompt_tokens",
        "response_tokens",
        "prompt_sha256",
        "response_sha256",
        "request_sha256",
    )
    exact_prompt_match = fingerprints_present and _fingerprint_counter(off, prompt_fields) == _fingerprint_counter(
        on, prompt_fields
    )
    exact_response_match = fingerprints_present and _fingerprint_counter(off, response_fields) == _fingerprint_counter(
        on, response_fields
    )
    exact_request_workload_match = (
        fingerprints_present
        and fingerprints_well_formed
        and _fingerprint_counter(off, request_fields) == _fingerprint_counter(on, request_fields)
    )
    vllm_stats_present = bool(off["vllm_stats"]) and bool(on["vllm_stats"])
    completed_steps_match = off["exit_success"] and on["exit_success"] and len(off["records"]) == len(on["records"])
    off_engine_prefix_cache = _last_prefix_cache_value(off)
    on_engine_prefix_cache = _last_prefix_cache_value(on)
    engine_prefix_cache_valid = off_engine_prefix_cache is False and on_engine_prefix_cache is True
    experiment_valid = (
        controlled_config_match
        and engine_prefix_cache_valid
        and exact_prompt_match
        and exact_response_match
        and exact_request_workload_match
        and vllm_stats_present
        and completed_steps_match
    )

    print()
    print("A/B comparison: prefix cache ON relative to OFF")
    print("-" * 64)
    print(f"controlled_config_match: {controlled_config_match}")
    print(f"prefix_cache_toggle_valid: {toggle_valid}")
    print(f"stable_request_seeds_valid: {stable_seeds_valid}")
    print(
        "batch_invariant: "
        f"off={off['config'].get('batch_invariant', 'missing')} "
        f"on={on['config'].get('batch_invariant', 'missing')} "
        f"consistent={batch_invariant_consistent}"
    )
    print(f"frozen_actor_valid: {frozen_actor_valid}")
    print(
        "engine_prefix_cache: "
        f"off={off_engine_prefix_cache} on={on_engine_prefix_cache} valid={engine_prefix_cache_valid}"
    )
    print("fingerprint_count: " f"off={len(off['fingerprints'])} on={len(on['fingerprints'])}")
    print(f"exact_prompt_fingerprints_match: {exact_prompt_match}")
    print(f"exact_response_fingerprints_match: {exact_response_match}")
    print(f"exact_request_workload_match: {exact_request_workload_match}")
    print(f"vllm_stats_present: {vllm_stats_present}")
    print(f"completed_steps_match: {completed_steps_match}")
    print(f"experiment_valid: {experiment_valid}")
    if not config_present:
        print("WARNING: benchmark config line is missing from one or both logs.")
    if config_differences:
        print(f"unexpected_config_differences: {', '.join(config_differences)}")

    for metric, unit, lower_is_better in (
        ("rollout", "s", True),
        ("step", "s", True),
        ("throughput", "", False),
    ):
        off_mean = _mean(off, metric)
        on_mean = _mean(on, metric)
        if off_mean is None or on_mean is None or off_mean == 0:
            continue
        change = _percent_change(off_mean, on_mean)
        label = "reduction" if lower_is_better else "gain"
        effect = -change if lower_is_better else change
        metric_name = metric if experiment_valid else f"raw_{metric}"
        print(f"{metric_name:24} off={off_mean:.3f}{unit}  on={on_mean:.3f}{unit}  " f"{label}={effect:.1f}%")

    for metric in ("tokens", "prompt_length", "response_length"):
        off_mean = _mean(off, metric)
        on_mean = _mean(on, metric)
        if off_mean is not None and on_mean is not None and off_mean != 0:
            print(
                f"{metric + '_parity':24} off={off_mean:.3f}  on={on_mean:.3f}  "
                f"difference={_percent_change(off_mean, on_mean):+.1f}%"
            )

    for summary in (off, on):
        hit_rates = [sample["cache_hit_rate_pct"] for sample in summary["vllm_stats"]]
        if hit_rates:
            print(
                f"{'cache_hit_' + summary['mode']:24} "
                f"mean={statistics.fmean(hit_rates):.1f}%  median={statistics.median(hit_rates):.1f}%"
            )

    off_peak = off["gpu_monitor_peak_mb"]
    on_peak = on["gpu_monitor_peak_mb"]
    if off_peak is not None and on_peak is not None:
        print(f"{'gpu_peak_delta':24} off={off_peak}MB  on={on_peak}MB  delta={on_peak - off_peak:+d}MB")

    parity_failures = []
    for metric in ("tokens", "prompt_length", "response_length"):
        off_mean = _mean(off, metric)
        on_mean = _mean(on, metric)
        if off_mean and on_mean and abs(_percent_change(off_mean, on_mean)) > 5.0:
            parity_failures.append(metric)
    print(f"aggregate_workload_parity_pass: {not parity_failures}")
    if parity_failures:
        print(
            "WARNING: workload differs by more than 5% for "
            f"{', '.join(parity_failures)}; do not attribute timing differences to prefix caching."
        )
    if not experiment_valid:
        print(
            "WARNING: exact workload or native vLLM-stat validation failed; "
            "raw timing differences are not a causal prefix-cache result."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")

    summaries = [parse_log(path, args.warmup_steps) for path in args.logs]
    modes = [summary["mode"] for summary in summaries]
    if len(modes) != len(set(modes)):
        parser.error("pass at most one log for each prefix-cache mode")

    for index, summary in enumerate(summaries):
        if index:
            print()
        print_summary(summary, args.warmup_steps)
    print_comparison(summaries)


if __name__ == "__main__":
    main()
