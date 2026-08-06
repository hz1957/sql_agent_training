"""Compare paired SQL-agent evaluation outputs and prepare an error-analysis review queue."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


Row = dict[str, Any]


def read_jsonl(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
    return rows


def index_rows(name: str, rows: list[Row]) -> dict[str, Row]:
    indexed: dict[str, Row] = {}
    for row in rows:
        uid = str(row.get("uid") or "")
        if not uid:
            raise ValueError(f"{name} contains a row without uid")
        if uid in indexed:
            raise ValueError(f"{name} contains duplicate uid: {uid}")
        indexed[uid] = row
    return indexed


def is_correct(row: Row) -> bool:
    return float(row.get("reward") or 0.0) > 0.0


def normalize_sql(sql: object) -> str:
    value = re.sub(r"\s+", " ", str(sql or "")).strip().rstrip(";")
    parts = re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")", value)
    return "".join(part if part.startswith(("'", '"')) else part.lower() for part in parts)


def sql_actions(row: Row) -> list[str]:
    actions: list[str] = []
    for turn in row.get("turns") or []:
        metadata = turn.get("metadata") or {}
        if turn.get("role") != "assistant":
            continue
        if metadata.get("agent_step") in {"write_query", "rewrite_query"}:
            content = str(turn.get("content") or "").strip()
            if content:
                actions.append(content)
    return actions


def checker_outputs(row: Row) -> list[str]:
    outputs: list[str] = []
    for turn in row.get("turns") or []:
        metadata = turn.get("metadata") or {}
        if turn.get("role") == "assistant" and metadata.get("agent_step") == "check_query":
            outputs.append(str(turn.get("content") or "").strip())
    return outputs


def execution_outputs(row: Row) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for turn in row.get("turns") or []:
        metadata = turn.get("metadata") or {}
        if turn.get("role") == "tool" and metadata.get("agent_step") == "execute_query":
            outputs.append(
                {
                    "ok": bool(metadata.get("ok")),
                    "sql": str(metadata.get("sql") or ""),
                    "error": metadata.get("error"),
                    "result": str(turn.get("content") or ""),
                }
            )
    return outputs


def extract_schema(row: Row) -> str:
    for turn in row.get("turns") or []:
        if turn.get("role") != "user":
            continue
        content = str(turn.get("content") or "")
        match = re.search(r"## Schema\n(.*?)(?:\n\nSQL:|\n\n##)", content, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def sql_features(sql: object) -> dict[str, Any]:
    normalized = normalize_sql(sql)
    alias_free = re.sub(r"\bt\d+\.", "", normalized)
    table_names = sorted(
        match.group(1).strip('`"[]')
        for match in re.finditer(r"\b(?:from|join)\s+([`\"\[]?[a-z_][\w$]*[`\"\]]?)", normalized)
    )
    on_clauses = [
        re.sub(r"\s+", " ", match.group(1)).strip()
        for match in re.finditer(
            r"\bon\s+(.+?)(?=\b(?:join|where|group\s+by|having|order\s+by|limit|union|intersect|except)\b|$)",
            alias_free,
        )
    ]
    where_match = re.search(
        r"\bwhere\s+(.+?)(?=\b(?:group\s+by|having|order\s+by|limit|union|intersect|except)\b|$)",
        alias_free,
    )
    projection_match = re.search(r"\bselect\s+(.*?)\s+\bfrom\b", alias_free)
    return {
        "tables": table_names,
        "join_count": len(re.findall(r"\bjoin\b", normalized)),
        "on_clauses": on_clauses,
        "where": where_match.group(1).strip() if where_match else "",
        "aggregates": sorted(re.findall(r"\b(?:count|sum|avg|min|max)\s*\(", normalized)),
        "group_by": bool(re.search(r"\bgroup\s+by\b", normalized)),
        "having": bool(re.search(r"\bhaving\b", normalized)),
        "select_count": len(re.findall(r"\bselect\b", normalized)),
        "set_ops": sorted(re.findall(r"\b(?:union|intersect|except)\b", normalized)),
        "distinct": bool(re.search(r"\bselect\s+distinct\b", normalized)),
        "order_by": bool(re.search(r"\border\s+by\b", normalized)),
        "limit": bool(re.search(r"\blimit\b", normalized)),
        "projection": projection_match.group(1).strip() if projection_match else "",
    }


def semantic_candidates(row: Row) -> list[str]:
    if is_correct(row):
        return []
    predicted = row.get("final_sql")
    gold = row.get("gold_sql")
    if not predicted or not bool(row.get("executable")):
        return ["Syntax/protocol"]
    if normalize_sql(predicted) == normalize_sql(gold):
        return ["Benchmark/evaluator ambiguity"]

    pred_features = sql_features(predicted)
    gold_features = sql_features(gold)
    labels: list[str] = []
    if pred_features["tables"] != gold_features["tables"]:
        labels.append("Schema linking")
    if (
        pred_features["join_count"] != gold_features["join_count"]
        or pred_features["on_clauses"] != gold_features["on_clauses"]
    ):
        labels.append("Join path/key")
    if pred_features["where"] != gold_features["where"]:
        labels.append("Predicate/value grounding")
    if any(
        pred_features[key] != gold_features[key]
        for key in ("aggregates", "group_by", "having")
    ):
        labels.append("Aggregation/GROUP BY/HAVING")
    if any(pred_features[key] != gold_features[key] for key in ("select_count", "set_ops")):
        labels.append("Nested query/set operation")
    if any(pred_features[key] != gold_features[key] for key in ("distinct", "order_by", "limit")):
        labels.append("DISTINCT/ORDER BY/LIMIT")
    if pred_features["projection"] != gold_features["projection"]:
        labels.append("Projection/schema linking")
    return labels or ["Other semantic mismatch"]


def behavior_labels(row: Row) -> list[str]:
    if is_correct(row):
        return []
    actions = sql_actions(row)
    normalized_actions = [normalize_sql(sql) for sql in actions]
    checkers = checker_outputs(row)
    labels: list[str] = []
    if bool(row.get("executable")):
        labels.append("Executable but semantically wrong")
    else:
        labels.append("Execution or protocol failure")
    if bool(row.get("ran_out_of_turns")):
        labels.append("Exhausted interaction budget")
    if len(set(normalized_actions)) < len(normalized_actions):
        labels.append("Repeated the same SQL")
    if len(set(normalized_actions)) > 1 and len(actions) > 1:
        labels.append("Unsuccessful repair")
    if checkers and "THE QUERY IS CORRECT." in checkers[-1]:
        labels.append("Checker accepted an incorrect final SQL")
    if bool(row.get("ran_out_of_turns")) and checkers and "THE QUERY IS INCORRECT." in checkers[-1]:
        labels.append("Checker rejected through final turn")
    return labels


def exact_mcnemar(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(baseline_only, candidate_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_summary(baseline_name: str, candidate_name: str, rows: dict[str, dict[str, Row]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    uids: dict[str, list[str]] = {
        "both_correct": [],
        "baseline_only": [],
        "candidate_only": [],
        "both_wrong": [],
    }
    for uid in rows[baseline_name]:
        baseline_correct = is_correct(rows[baseline_name][uid])
        candidate_correct = is_correct(rows[candidate_name][uid])
        if baseline_correct and candidate_correct:
            group = "both_correct"
        elif baseline_correct:
            group = "baseline_only"
        elif candidate_correct:
            group = "candidate_only"
        else:
            group = "both_wrong"
        counts[group] += 1
        uids[group].append(uid)

    baseline_only = counts["baseline_only"]
    candidate_only = counts["candidate_only"]
    total = len(rows[baseline_name])
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "total": total,
        "both_correct": counts["both_correct"],
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "both_wrong": counts["both_wrong"],
        "accuracy_difference_points": 100.0 * (candidate_only - baseline_only) / total,
        "discordant_pairs": baseline_only + candidate_only,
        "matched_odds_ratio": candidate_only / baseline_only if baseline_only else None,
        "exact_mcnemar_two_sided_p": exact_mcnemar(baseline_only, candidate_only),
        "uids": uids,
    }


def model_summary(rows: dict[str, Row]) -> dict[str, Any]:
    values = list(rows.values())
    failures = [row for row in values if not is_correct(row)]
    semantic_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    for row in failures:
        semantic_counts.update(semantic_candidates(row))
        behavior_counts.update(behavior_labels(row))
    return {
        "total": len(values),
        "correct": sum(is_correct(row) for row in values),
        "accuracy": sum(is_correct(row) for row in values) / len(values),
        "failures": len(failures),
        "executable_failures": sum(bool(row.get("executable")) for row in failures),
        "non_executable_failures": sum(not bool(row.get("executable")) for row in failures),
        "semantic_candidate_counts": dict(semantic_counts.most_common()),
        "behavior_counts": dict(behavior_counts.most_common()),
    }


def validate_alignment(rows: dict[str, dict[str, Row]]) -> list[str]:
    names = list(rows)
    reference_name = names[0]
    reference = rows[reference_name]
    reference_uids = set(reference)
    for name in names[1:]:
        if set(rows[name]) != reference_uids:
            missing = sorted(reference_uids - set(rows[name]))
            extra = sorted(set(rows[name]) - reference_uids)
            raise ValueError(f"UID mismatch for {name}: missing={missing[:5]} extra={extra[:5]}")
        for uid, reference_row in reference.items():
            row = rows[name][uid]
            question_mismatch = row.get("question") != reference_row.get("question")
            gold_mismatch = row.get("gold_sql") != reference_row.get("gold_sql")
            if question_mismatch or gold_mismatch:
                raise ValueError(f"Question or gold SQL mismatch for {name}, uid={uid}")
    return list(reference)


def write_review_queue(output_path: Path, uid_order: list[str], rows: dict[str, dict[str, Row]]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for uid in uid_order:
            if all(is_correct(model_rows[uid]) for model_rows in rows.values()):
                continue
            reference = rows["SFT"][uid]
            handle.write(f"## {uid}\n\n")
            handle.write(f"Question: {reference.get('question')}\n\n")
            handle.write(f"Schema:\n```text\n{extract_schema(reference)}\n```\n\n")
            handle.write(f"Gold SQL:\n```sql\n{reference.get('gold_sql')}\n```\n\n")
            for name in ("SFT", "S1", "S3"):
                row = rows[name][uid]
                handle.write(
                    f"### {name}: {'correct' if is_correct(row) else 'wrong'}; "
                    f"executable={bool(row.get('executable'))}; turns={row.get('assistant_turns')}; "
                    f"ran_out={bool(row.get('ran_out_of_turns'))}\n\n"
                )
                handle.write(f"Final SQL:\n```sql\n{row.get('final_sql') or '(empty)'}\n```\n\n")
                handle.write(f"Semantic candidates: {', '.join(semantic_candidates(row)) or 'none'}\n\n")
                handle.write(f"Behavior labels: {', '.join(behavior_labels(row)) or 'none'}\n\n")
                actions = sql_actions(row)
                checkers = checker_outputs(row)
                executions = execution_outputs(row)
                for index, action in enumerate(actions):
                    handle.write(f"- action {index + 1}: `{action}`\n")
                    if index < len(executions):
                        handle.write(f"  execution: `{executions[index]['result']}`\n")
                    if index < len(checkers):
                        handle.write(f"  checker: `{checkers[index]}`\n")
                handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--s3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = {
        "SFT": index_rows("SFT", read_jsonl(args.sft)),
        "S1": index_rows("S1", read_jsonl(args.s1)),
        "S3": index_rows("S3", read_jsonl(args.s3)),
    }
    uid_order = validate_alignment(rows)
    summary = {
        "sample_size": len(uid_order),
        "models": {name: model_summary(model_rows) for name, model_rows in rows.items()},
        "pairs": {
            "SFT_vs_S3": paired_summary("SFT", "S3", rows),
            "S1_vs_S3": paired_summary("S1", "S3", rows),
        },
        "limitations": [
            "SQL semantic labels are structural review candidates and can be multi-label; "
            "representative cases require manual review.",
            "Final eval traces do not contain all S3 tree candidates, so branch diversity and "
            "pruned-correct-branch claims are unobservable.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_review_queue(args.output_dir / "review_queue.md", uid_order, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
