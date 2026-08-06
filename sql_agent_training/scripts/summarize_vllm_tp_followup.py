#!/usr/bin/env python3
"""Aggregate repeated vLLM TP benchmark summaries into mean and sample standard deviation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PROMPT_METRICS = (
    "requests_per_s",
    "output_tokens_per_s",
    "total_tokens_per_s",
    "latency_mean_s",
    "latency_p50_s",
    "latency_p95_s",
    "latency_p99_s",
    "ttft_mean_s",
    "tpot_mean_s",
    "peak_memory_mb_max",
    "peak_memory_mb_sum",
)

AGENT_METRICS = (
    "trajectories_per_s",
    "model_calls_per_s",
    "output_tokens_per_s",
    "total_tokens_per_s",
    "trajectory_latency_mean_s",
    "trajectory_latency_p50_s",
    "trajectory_latency_p95_s",
    "trajectory_latency_p99_s",
    "avg_model_calls",
    "avg_sql_actions",
    "rewrite_rate",
    "execution_accuracy",
    "executable_rate",
    "ran_out_of_turns_rate",
    "parse_error_rate",
    "peak_memory_mb_max",
    "peak_memory_mb_sum",
)


def _read_runs(result_root: Path) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for summary_path in sorted(result_root.glob("*/*/run_*/*/summary.json")):
        relative = summary_path.relative_to(result_root)
        phase = relative.parts[0]
        concurrency_part = relative.parts[1]
        if not concurrency_part.startswith("concurrency_"):
            continue
        concurrency = int(concurrency_part.removeprefix("concurrency_"))
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["summary_path"] = str(summary_path)
        grouped[(phase, concurrency, str(payload["case_name"]))].append(payload)
    return grouped


def _metric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def aggregate(result_root: Path) -> dict[str, Any]:
    grouped = _read_runs(result_root)
    if not grouped:
        raise FileNotFoundError(f"No repeated benchmark summaries found under {result_root}")

    groups: list[dict[str, Any]] = []
    for (phase, concurrency, case_name), runs in sorted(grouped.items()):
        benchmark_kind = str(runs[0].get("benchmark_kind", "prompt"))
        metric_names = AGENT_METRICS if benchmark_kind == "agent_trace" else PROMPT_METRICS
        metrics: dict[str, dict[str, float | int | None]] = {}
        for metric_name in metric_names:
            values = [float(run[metric_name]) for run in runs if run.get(metric_name) is not None]
            if values:
                metrics[metric_name] = _metric_summary(values)
        groups.append(
            {
                "phase": phase,
                "benchmark_kind": benchmark_kind,
                "concurrency": concurrency,
                "case_name": case_name,
                "run_count": len(runs),
                "summary_paths": [str(run["summary_path"]) for run in runs],
                "metrics": metrics,
            }
        )
    return {"result_root": str(result_root), "groups": groups}


def write_outputs(payload: dict[str, Any], *, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "phase",
                "benchmark_kind",
                "concurrency",
                "case_name",
                "run_count",
                "metric",
                "n",
                "mean",
                "sample_std",
                "min",
                "max",
            ),
        )
        writer.writeheader()
        for group in payload["groups"]:
            for metric_name, metric in group["metrics"].items():
                writer.writerow(
                    {
                        "phase": group["phase"],
                        "benchmark_kind": group["benchmark_kind"],
                        "concurrency": group["concurrency"],
                        "case_name": group["case_name"],
                        "run_count": group["run_count"],
                        "metric": metric_name,
                        **metric,
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).resolve()
    payload = aggregate(result_root)
    json_path = Path(args.json_output).resolve() if args.json_output else result_root / "aggregate_summary.json"
    csv_path = Path(args.csv_output).resolve() if args.csv_output else result_root / "aggregate_summary.csv"
    write_outputs(payload, json_path=json_path, csv_path=csv_path)
    print(json.dumps({"groups": len(payload["groups"]), "json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
