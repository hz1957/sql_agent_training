from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_analyzer() -> ModuleType:
    script = Path(__file__).parents[1] / "scripts" / "analyze_verl_prefix_cache_benchmark.py"
    spec = importlib.util.spec_from_file_location("analyze_verl_prefix_cache_benchmark", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_log(path: Path, mode: str, enabled: str, rollout_times: list[float], cache_hit_rate: float) -> None:
    lines = [
        (
            "PREFIX_CACHE_BENCHMARK_CONFIG "
            f"mode={mode} enable_prefix_caching={enabled} disable_log_stats=False "
            "steps=3 warmup=1 data_seed=42 rollout_seed=42 stable_request_seeds=True "
            "batch_invariant=False frozen_actor=True workload_fingerprint=True actor_lr=0 "
            "prefix_cache_patch=True"
        ),
        (
            "PREFIX_CACHE_PATCH applied "
            f"rollout_enable_prefix_caching={enabled} engine_before=True engine_after={enabled}"
        ),
    ]
    for step, rollout_time in enumerate(rollout_times, start=1):
        lines.append(
            "PREFIX_CACHE_WORKLOAD_FINGERPRINT "
            + json.dumps(
                {
                    "root_uid": f"root-{step}",
                    "requests": 2,
                    "prompt_tokens": 8,
                    "response_tokens": 4,
                    "prompt_sha256": f"prompt-{step}",
                    "response_sha256": f"response-{step}",
                    "request_sha256": f"request-{step}",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        lines.append(
            "Engine 000: Avg prompt throughput: 100.0 tokens/s, "
            "Avg generation throughput: 20.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs, "
            f"GPU KV cache usage: 10.0%, Prefix cache hit rate: {cache_hit_rate}%"
        )
        lines.append(
            f"step:{step} - timing_s/step:{rollout_time + 40.0} - timing_s/gen:{rollout_time} "
            "- timing_s/update_weights:7.0 - timing_s/update_actor:30.0 "
            "- perf/throughput:40.0 - perf/total_num_tokens:10000 "
            "- prompt_length/mean:1000 - response_length/mean:10"
        )
    lines.extend(
        [
            "GPU_MONITOR 2026/07/29 10:23:40.602, 0, 59893, 81559, 90",
            "EXIT_CODE 0 Wed Jul 29 10:23:42 EDT 2026",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_parse_log_collects_native_and_vllm_metrics(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    log = tmp_path / "prefix_cache_on.log"
    _write_log(log, "on", "True", [20.0, 18.0, 16.0], 75.0)

    summary = analyzer.parse_log(log, warmup_steps=1)

    assert summary["mode"] == "on"
    assert summary["exit_success"] is True
    assert len(summary["measured"]) == 2
    assert summary["measured"][0]["rollout"] == 18.0
    assert summary["vllm_stats"][0]["cache_hit_rate_pct"] == 75.0
    assert len(summary["fingerprints"]) == 3
    assert summary["gpu_monitor_peak_mb"] == 59893


def test_comparison_checks_controlled_config(tmp_path: Path, capsys: Any) -> None:
    analyzer = _load_analyzer()
    off_log = tmp_path / "prefix_cache_off.log"
    on_log = tmp_path / "prefix_cache_on.log"
    _write_log(off_log, "off", "False", [20.0, 20.0, 20.0], 0.0)
    _write_log(on_log, "on", "True", [20.0, 15.0, 15.0], 75.0)

    analyzer.print_comparison(
        [
            analyzer.parse_log(off_log, warmup_steps=1),
            analyzer.parse_log(on_log, warmup_steps=1),
        ]
    )
    output = capsys.readouterr().out

    assert "controlled_config_match: True" in output
    assert "prefix_cache_toggle_valid: True" in output
    assert "stable_request_seeds_valid: True" in output
    assert "batch_invariant: off=False on=False consistent=True" in output
    assert "engine_prefix_cache: off=False on=True valid=True" in output
    assert "exact_prompt_fingerprints_match: True" in output
    assert "exact_response_fingerprints_match: True" in output
    assert "exact_request_workload_match: True" in output
    assert "vllm_stats_present: True" in output
    assert "experiment_valid: True" in output
    assert "aggregate_workload_parity_pass: True" in output
    assert "rollout                  off=20.000s  on=15.000s  reduction=25.0%" in output


def test_comparison_rejects_different_prompt_fingerprints(tmp_path: Path, capsys: Any) -> None:
    analyzer = _load_analyzer()
    off_log = tmp_path / "prefix_cache_off.log"
    on_log = tmp_path / "prefix_cache_on.log"
    _write_log(off_log, "off", "False", [20.0, 20.0, 20.0], 0.0)
    _write_log(on_log, "on", "True", [20.0, 15.0, 15.0], 75.0)
    text = on_log.read_text(encoding="utf-8").replace('"prompt_sha256":"prompt-2"', '"prompt_sha256":"changed"')
    on_log.write_text(text, encoding="utf-8")

    analyzer.print_comparison(
        [
            analyzer.parse_log(off_log, warmup_steps=1),
            analyzer.parse_log(on_log, warmup_steps=1),
        ]
    )
    output = capsys.readouterr().out

    assert "exact_prompt_fingerprints_match: False" in output
    assert "experiment_valid: False" in output
    assert "raw_rollout" in output


def test_comparison_rejects_engine_prefix_cache_mismatch(tmp_path: Path, capsys: Any) -> None:
    analyzer = _load_analyzer()
    off_log = tmp_path / "prefix_cache_off.log"
    on_log = tmp_path / "prefix_cache_on.log"
    _write_log(off_log, "off", "False", [20.0, 20.0, 20.0], 0.0)
    _write_log(on_log, "on", "True", [20.0, 15.0, 15.0], 75.0)
    text = off_log.read_text(encoding="utf-8").replace("engine_after=False", "engine_after=True")
    off_log.write_text(text, encoding="utf-8")

    analyzer.print_comparison(
        [
            analyzer.parse_log(off_log, warmup_steps=1),
            analyzer.parse_log(on_log, warmup_steps=1),
        ]
    )
    output = capsys.readouterr().out

    assert "engine_prefix_cache: off=True on=True valid=False" in output
    assert "experiment_valid: False" in output
