"""Analyze GRPO rollout JSONL files for reward signal and SQL failure causes."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return rows


def _parent_rollout_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("parent_rollout_id") or row.get("rollout_id") or "")


def _group_id(row: dict[str, Any]) -> str:
    return str(row.get("group_id") or row.get("uid") or "")


def _reward(row: dict[str, Any]) -> float:
    return float(row.get("reward") or 0.0)


def _clip(text: str | None, limit: int) -> str:
    if text is None:
        return ""
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _error_bucket(error: object) -> str:
    if error is None:
        return "none"
    text = str(error).strip()
    lowered = text.lower()
    if not text:
        return "empty"
    patterns = [
        ("no such table", "no_such_table"),
        ("no such column", "no_such_column"),
        ("ambiguous column", "ambiguous_column"),
        ("syntax error", "syntax_error"),
        ("incomplete input", "incomplete_input"),
        ("misuse of aggregate", "aggregate_misuse"),
        ("no such function", "no_such_function"),
        ("near ", "syntax_near"),
        ("only select", "safety_not_select"),
        ("read-only", "safety_not_read_only"),
    ]
    for needle, bucket in patterns:
        if needle in lowered:
            return bucket
    return "other"


def _parent_rollout_rewards(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[_group_id(row)][_parent_rollout_id(row)] = _reward(row)
    return grouped


def _print_header(title: str) -> None:
    print(f"\n## {title}")


def analyze(path: Path, *, max_response_length: int | None, examples: int) -> None:
    rows = _read_rows(path)
    if not rows:
        print("No rows found.")
        return

    parent_rewards_by_group = _parent_rollout_rewards(rows)
    parent_rewards = [reward for rewards in parent_rewards_by_group.values() for reward in rewards.values()]
    zero_variance_groups = 0
    mixed_groups = 0
    group_patterns: Counter[tuple[float, ...]] = Counter()
    for rewards in parent_rewards_by_group.values():
        pattern = tuple(sorted(rewards.values()))
        group_patterns[pattern] += 1
        if len(set(rewards.values())) <= 1:
            zero_variance_groups += 1
        else:
            mixed_groups += 1

    _print_header("Reward Signal")
    print(f"rows/transitions: {len(rows)}")
    print(f"groups/tasks: {len(parent_rewards_by_group)}")
    print(f"parent rollouts: {len(parent_rewards)}")
    print(f"transition reward counts: {dict(Counter(_reward(row) for row in rows))}")
    print(f"parent rollout reward counts: {dict(Counter(parent_rewards))}")
    print(f"zero-variance groups: {zero_variance_groups}/{len(parent_rewards_by_group)}")
    print(f"mixed-signal groups: {mixed_groups}/{len(parent_rewards_by_group)}")
    print("top group reward patterns:")
    for pattern, count in group_patterns.most_common(8):
        print(f"  {pattern}: {count}")

    response_lengths = [int(row.get("response_tokens") or 0) for row in rows]
    capped = sum(1 for length in response_lengths if max_response_length is not None and length >= max_response_length)
    _print_header("Length And Turns")
    print(f"avg response tokens: {sum(response_lengths) / len(response_lengths):.2f}")
    print(f"max response tokens: {max(response_lengths)}")
    if max_response_length is not None:
        print(f"responses at/above cap {max_response_length}: {capped}/{len(rows)}")
    turn_counts = Counter(int((row.get("metadata") or {}).get("turn_index", 0)) for row in rows)
    print(f"turn index counts: {dict(sorted(turn_counts.items()))}")
    print(f"ran_out_of_turns rows: {sum(bool((row.get('metadata') or {}).get('ran_out_of_turns')) for row in rows)}/{len(rows)}")

    _print_header("SQL Execution")
    tool_ok_counts = Counter(bool((row.get("metadata") or {}).get("tool_ok")) for row in rows)
    error_buckets = Counter(_error_bucket((row.get("metadata") or {}).get("tool_error")) for row in rows)
    raw_errors = Counter(str((row.get("metadata") or {}).get("tool_error")) for row in rows if (row.get("metadata") or {}).get("tool_error"))
    print(f"tool_ok counts: {dict(tool_ok_counts)}")
    print(f"tool error buckets: {dict(error_buckets)}")
    print("top raw tool errors:")
    for error, count in raw_errors.most_common(12):
        print(f"  [{count}] {_clip(error, 180)}")

    failing_rows = [
        row
        for row in rows
        if not bool((row.get("metadata") or {}).get("tool_ok")) or _reward(row) == 0.0
    ]
    _print_header("Failure Examples")
    for index, row in enumerate(failing_rows[:examples], start=1):
        metadata = row.get("metadata") or {}
        print(f"\n### Example {index}")
        print(f"uid: {row.get('uid')} rollout: {_parent_rollout_id(row)} turn: {metadata.get('turn_index')}")
        print(f"reward: {row.get('reward')} tool_ok: {metadata.get('tool_ok')} error: {_clip(str(metadata.get('tool_error')), 240)}")
        print(f"final_sql: {_clip(str(metadata.get('final_sql')), 300)}")
        print(f"prompt: {_clip(row.get('prompt'), 700)}")
        print(f"response: {_clip(row.get('response'), 700)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts_jsonl", type=Path)
    parser.add_argument("--max-response-length", type=int, default=None)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    analyze(args.rollouts_jsonl, max_response_length=args.max_response_length, examples=args.examples)


if __name__ == "__main__":
    main()
