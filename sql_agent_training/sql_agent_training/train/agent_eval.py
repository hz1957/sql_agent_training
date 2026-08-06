"""Evaluate SQL-agent checkpoints on Spider splits."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from sql_agent_training.agent.model_client import HuggingFaceModelClient, ModelClient, OpenAIChatModelClient
from sql_agent_training.agent.sql_agent_loop import SqlAgentInput, SqlAgentLoop
from sql_agent_training.agent.trace_format import AgentTrajectory
from sql_agent_training.agent.tree_sql_agent_loop import TreeSqlAgentEvalLoop
from sql_agent_training.data.schema import build_schema_prompt, load_tables_json
from sql_agent_training.data.spider_dataset import SpiderExample, expected_sqlite_path, load_spider_file
from sql_agent_training.train.eval_sampling import select_eval_examples


@dataclass(frozen=True)
class AgentEvalResult:
    """One evaluated SQL-agent trajectory."""

    uid: str
    db_id: str
    question: str
    gold_sql: str
    final_sql: str | None
    final_sql_source: str
    reward: float
    executable: bool
    assistant_turns: int
    num_execute_calls: int
    num_parse_errors: int
    ran_out_of_turns: bool
    no_parseable_sql: bool
    turns: list[dict[str, Any]]
    metadata: dict[str, Any]


def _trajectory_to_result(example: SpiderExample, trajectory: AgentTrajectory) -> AgentEvalResult:
    metadata = trajectory.metadata
    assistant_turns = sum(
        1 for turn in trajectory.turns if turn.role == "assistant" and turn.metadata.get("trainable") is not False
    )

    def clean_turn(turn: Any) -> dict[str, Any]:
        row = asdict(turn)
        for key in ("prompt_ids", "response_ids", "prompt_text", "response_text"):
            row["metadata"].pop(key, None)
        return row

    return AgentEvalResult(
        uid=example.uid,
        db_id=example.db_id,
        question=example.question,
        gold_sql=example.gold_sql,
        final_sql=trajectory.final_sql,
        final_sql_source=trajectory.final_sql_source,
        reward=float(trajectory.reward or 0.0),
        executable=trajectory.final_sql is not None,
        assistant_turns=assistant_turns,
        num_execute_calls=int(metadata.get("num_execute_calls", 0)),
        num_parse_errors=int(metadata.get("num_parse_errors", 0)),
        ran_out_of_turns=bool(metadata.get("ran_out_of_turns", False)),
        no_parseable_sql=bool(metadata.get("no_parseable_sql", False)),
        turns=[clean_turn(turn) for turn in trajectory.turns],
        metadata=dict(metadata),
    )


def evaluate_agent(
    examples: list[SpiderExample],
    tables_index: dict[str, dict[str, Any]],
    data_dir: str | Path,
    *,
    model_client: ModelClient | None = None,
    checker_model_client: ModelClient | None = None,
    dry_run_gold: bool = False,
    max_turns: int = 2,
    max_tokens: int = 256,
    temperature: float = 0.0,
    checker_temperature: float | None = None,
    inference_mode: str = "chain",
    concurrency: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
    tree_branch_n: int = 4,
    tree_beam_size: int = 2,
    tree_beam_tau: float = 1.0,
    tree_beam_epsilon_random: float = 0.0,
    tree_seed: int = 0,
) -> list[AgentEvalResult]:
    """Run the SQL agent over examples and collect per-example results."""

    if model_client is None and not dry_run_gold:
        raise ValueError("model_client is required unless dry_run_gold is enabled")
    if inference_mode not in {"chain", "tree"}:
        raise ValueError(f"inference_mode must be 'chain' or 'tree', got {inference_mode!r}")
    if concurrency <= 0:
        raise ValueError("concurrency must be greater than zero")

    def evaluate_one(index: int, example: SpiderExample) -> AgentEvalResult:
        sample = SqlAgentInput(
            uid=example.uid,
            rollout_id=f"{example.uid}:eval{index}",
            question=example.question,
            db_id=example.db_id,
            schema_prompt=build_schema_prompt(example.db_id, tables_index),
            gold_sql=example.gold_sql,
        )
        sqlite_path = expected_sqlite_path(data_dir, example.db_id)
        if dry_run_gold:
            chain_loop = SqlAgentLoop(max_turns=max_turns)
            trajectory = chain_loop.run_with_responses(sample, [example.gold_sql], sqlite_path)
        else:
            assert model_client is not None
            if inference_mode == "tree":
                tree_loop = TreeSqlAgentEvalLoop(
                    max_turns=max_turns,
                    branch_n=tree_branch_n,
                    beam_size=tree_beam_size,
                    beam_tau=tree_beam_tau,
                    beam_epsilon_random=tree_beam_epsilon_random,
                    seed=tree_seed,
                )
                trajectory = tree_loop.run(
                    sample,
                    model_client,
                    sqlite_path,
                    checker_model_client=checker_model_client,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    checker_temperature=checker_temperature,
                )
            else:
                chain_loop = SqlAgentLoop(max_turns=max_turns)
                trajectory = chain_loop.run(
                    sample,
                    model_client,
                    sqlite_path,
                    checker_model_client=checker_model_client,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    checker_temperature=checker_temperature,
                )
        return _trajectory_to_result(example, trajectory)

    total = len(examples)
    if concurrency == 1 or total <= 1:
        rows: list[AgentEvalResult] = []
        for index, example in enumerate(examples):
            rows.append(evaluate_one(index, example))
            if progress_callback is not None:
                progress_callback(index + 1, total)
        return rows

    ordered_rows: list[AgentEvalResult | None] = [None] * total
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(evaluate_one, index, example): index for index, example in enumerate(examples)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            ordered_rows[index] = future.result()
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)

    if any(row is None for row in ordered_rows):
        raise RuntimeError("Agent evaluation completed without a result for every example")
    return [row for row in ordered_rows if row is not None]


def summarize_agent_eval(rows: list[AgentEvalResult]) -> dict[str, float | int]:
    """Aggregate agent evaluation rows into headline and diagnostic metrics."""

    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "execution_accuracy": 0.0,
            "executable_rate": 0.0,
            "avg_turns": 0.0,
            "avg_execute_calls": 0.0,
            "parse_error_rate": 0.0,
            "ran_out_of_turns_rate": 0.0,
            "no_parseable_sql_rate": 0.0,
            "rewrite_rate": 0.0,
            "rewrite_success_rate": 0.0,
        }

    rewrite_rows = [row for row in rows if row.assistant_turns > 1]
    metrics: dict[str, float | int] = {
        "total": total,
        "execution_accuracy": sum(row.reward for row in rows) / total,
        "executable_rate": sum(1 for row in rows if row.executable) / total,
        "avg_turns": sum(row.assistant_turns for row in rows) / total,
        "avg_execute_calls": sum(row.num_execute_calls for row in rows) / total,
        "parse_error_rate": sum(1 for row in rows if row.num_parse_errors > 0) / total,
        "ran_out_of_turns_rate": sum(1 for row in rows if row.ran_out_of_turns) / total,
        "no_parseable_sql_rate": sum(1 for row in rows if row.no_parseable_sql) / total,
        "rewrite_rate": len(rewrite_rows) / total,
        "rewrite_success_rate": (
            sum(1 for row in rewrite_rows if row.reward == 1.0) / len(rewrite_rows) if rewrite_rows else 0.0
        ),
    }
    tree_rows = [row for row in rows if row.metadata.get("inference_mode") == "tree"]
    if tree_rows:
        metrics.update(
            {
                "avg_tree_nodes": sum(float(row.metadata.get("tree_nodes", 0)) for row in tree_rows)
                / len(tree_rows),
                "avg_tree_terminal_candidates": sum(
                    float(row.metadata.get("tree_terminal_candidates", 0)) for row in tree_rows
                )
                / len(tree_rows),
                "tree_checker_approved_final_rate": sum(
                    1
                    for row in tree_rows
                    if row.metadata.get("tree_final_node_id")
                    and row.metadata.get("tree_final_node_id") is not None
                    and row.final_sql_source == "tree_checker_approved"
                )
                / len(tree_rows),
                "tree_executable_fallback_final_rate": sum(
                    1
                    for row in tree_rows
                    if row.metadata.get("tree_final_node_id")
                    and row.metadata.get("tree_final_node_id") is not None
                    and row.final_sql_source == "tree_executable_fallback"
                )
                / len(tree_rows),
            }
        )
    return metrics


def write_eval_outputs(
    rows: list[AgentEvalResult],
    metrics: dict[str, float | int],
    *,
    predictions_jsonl: str | Path,
    metrics_json: str | Path,
) -> None:
    """Write per-example predictions and aggregate metrics."""

    predictions_path = Path(predictions_jsonl)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    metrics_path = Path(metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def _split_file(config: dict[str, Any], split: str) -> str:
    data = config["data"]
    if split == "validation":
        return str(data.get("validation_file", "dev.json"))
    return str(data.get("train_file", "train_spider.json"))


def _eval_sample_size(config: dict[str, Any], cli_value: int | None) -> int | None:
    if cli_value is not None:
        return cli_value
    value = config.get("eval", {}).get("sample_size")
    return int(value) if value is not None else None


def _eval_sample_seed(config: dict[str, Any], cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    return int(config.get("eval", {}).get("sample_seed", 0))


def _rollout_str(config: dict[str, Any], key: str, default: str) -> str:
    rollout_config = config.get("rollout", {})
    return str(rollout_config.get(key, default))


def _rollout_int(config: dict[str, Any], key: str, default: int) -> int:
    rollout_config = config.get("rollout", {})
    return int(rollout_config.get(key, default))


def _rollout_float(config: dict[str, Any], key: str, default: float) -> float:
    rollout_config = config.get("rollout", {})
    return float(rollout_config.get(key, default))


def _has_tokenizer_files(path: str | Path) -> bool:
    root = Path(path)
    return any((root / name).exists() for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"))


def _load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Invalid .env entry at {env_path}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _load_model_client(
    config: dict[str, Any],
    checkpoint: str | None,
    tokenizer_path: str | None,
    *,
    backend: str | None = None,
    api_url: str | None = None,
    model_name: str | None = None,
    api_key_env: str | None = None,
    request_timeout_seconds: float | None = None,
) -> ModelClient:
    model_config = config.get("model", {})
    rollout_config = config.get("rollout", {})
    resolved_backend = str(backend or model_config.get("backend", "hf")).strip().lower()

    if resolved_backend in {"openai_chat", "sglang"}:
        api_url_env = str(model_config.get("api_url_env", "LLM_API_URL_AGENT"))
        model_name_env = str(model_config.get("model_name_env", "LLM_MODEL_NAME"))
        resolved_api_key_env = str(api_key_env or model_config.get("api_key_env", "LLM_API_KEY_AGENT"))
        resolved_api_url = str(api_url or model_config.get("base_url") or os.environ.get(api_url_env, ""))
        resolved_model_name = str(model_name or model_config.get("model_name") or os.environ.get(model_name_env, ""))
        resolved_api_key = os.environ.get(resolved_api_key_env)
        if not resolved_api_url:
            raise ValueError(f"Remote chat backend requires model.base_url or ${api_url_env}")
        if not resolved_model_name:
            raise ValueError(f"Remote chat backend requires model.model_name or ${model_name_env}")
        if not resolved_api_key:
            raise ValueError(f"Remote chat backend requires ${resolved_api_key_env}")
        timeout_seconds = (
            request_timeout_seconds
            if request_timeout_seconds is not None
            else float(model_config.get("request_timeout_seconds", 300.0))
        )
        return OpenAIChatModelClient(
            base_url=resolved_api_url,
            model_name=resolved_model_name,
            api_key=resolved_api_key,
            timeout_seconds=timeout_seconds,
            max_new_tokens=int(rollout_config.get("max_response_length", 256)),
            temperature=float(rollout_config.get("temperature", 0.0)),
            top_p=float(rollout_config["top_p"]) if rollout_config.get("top_p") is not None else None,
            max_retries=int(model_config.get("request_retries", 2)),
            retry_backoff_seconds=float(model_config.get("retry_backoff_seconds", 1.0)),
        )

    if resolved_backend != "hf":
        raise ValueError(f"Unsupported model backend: {resolved_backend!r}")

    tokenizer_config = config.get("tokenizer", {})
    training_config = config.get("training", {})
    model_path = checkpoint or str(model_config["path"])
    resolved_tokenizer = tokenizer_path or str(
        model_config.get("tokenizer_path") or tokenizer_config.get("path") or ""
    )
    if not resolved_tokenizer:
        resolved_tokenizer = model_path if _has_tokenizer_files(model_path) else str(model_config["path"])

    return HuggingFaceModelClient(
        model_path,
        tokenizer_name_or_path=resolved_tokenizer,
        base_model_name_or_path=str(model_config["path"]) if model_config.get("path") else None,
        device=str(model_config.get("device", training_config.get("device", "auto"))),
        torch_dtype=str(model_config["torch_dtype"]) if model_config.get("torch_dtype") else None,
        max_new_tokens=int(rollout_config.get("max_response_length", 256)),
        temperature=float(rollout_config.get("temperature", 0.0)),
        top_p=float(rollout_config["top_p"]) if rollout_config.get("top_p") is not None else None,
        top_k=int(rollout_config["top_k"]) if rollout_config.get("top_k") is not None else None,
    )


def _config_with_model_section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Return a config copy that uses a model override section."""

    model_override = config.get(section_name) or {}
    if not isinstance(model_override, dict):
        raise ValueError(f"{section_name} must be a mapping when provided")
    merged = dict(config)
    merged["model"] = {**config.get("model", {}), **model_override}
    rollout_override = model_override.get("rollout")
    if isinstance(rollout_override, dict):
        merged["rollout"] = {**config.get("rollout", {}), **rollout_override}
    return merged


def _checker_requested(config: dict[str, Any], args: argparse.Namespace) -> bool:
    checker_config = config.get("checker_model")
    return bool(
        checker_config
        or args.checker_backend
        or args.checker_api_url
        or args.checker_model_name
        or args.checker_api_key_env
        or args.checker_request_timeout_seconds is not None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SQL-agent model on a Spider split.")
    parser.add_argument("--config", default="configs/agent_eval.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint to evaluate. Defaults to config model.path.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path. Defaults to checkpoint tokenizer or model.path.",
    )
    parser.add_argument("--backend", choices=["hf", "openai_chat", "sglang"], default=None)
    parser.add_argument("--env-file", default=None, help="Environment file. Defaults to the project .env file.")
    parser.add_argument("--api-url", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--model-name", default=None, help="Remote model name.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable containing the API key.")
    parser.add_argument("--request-timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--checker-backend",
        choices=["hf", "openai_chat", "sglang"],
        default=None,
        help="Optional separate checker backend. If set, SQL generation still uses --backend/config model.",
    )
    parser.add_argument("--checker-api-url", default=None, help="OpenAI-compatible checker API base URL.")
    parser.add_argument("--checker-model-name", default=None, help="Remote checker model name.")
    parser.add_argument("--checker-api-key-env", default=None, help="Environment variable containing checker API key.")
    parser.add_argument("--checker-request-timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--checker-temperature",
        type=float,
        default=None,
        help="Sampling temperature for the separate checker. Defaults to 0.0 when a checker model is configured.",
    )
    parser.add_argument("--concurrency", type=int, default=None, help="Concurrent complete agent trajectories.")
    parser.add_argument("--log-every", type=int, default=None, help="Report progress after this many examples.")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None, help="Randomly sample this many examples for eval.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed used with --sample-size/config eval.sample_size.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--predictions-jsonl", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--dry-run-gold", action="store_true", help="Evaluate by emitting gold SQL for plumbing tests.")
    parser.add_argument("--inference-mode", choices=["chain", "tree"], default=None)
    parser.add_argument("--tree-branch-n", type=int, default=None)
    parser.add_argument("--tree-beam-size", type=int, default=None)
    parser.add_argument("--tree-beam-tau", type=float, default=None)
    parser.add_argument("--tree-beam-epsilon-random", type=float, default=None)
    parser.add_argument("--tree-seed", type=int, default=None)
    args = parser.parse_args()

    default_env_file = Path(__file__).resolve().parents[2] / ".env"
    env_file = Path(args.env_file) if args.env_file else default_env_file
    if args.env_file and not env_file.exists():
        parser.error(f"--env-file does not exist: {env_file}")
    _load_env_file(env_file)

    config = _load_config(args.config)
    data_dir = Path(config["data"]["data_dir"])
    examples = load_spider_file(data_dir / _split_file(config, args.split))
    examples = select_eval_examples(
        examples,
        limit=args.limit,
        sample_size=_eval_sample_size(config, args.sample_size),
        sample_seed=_eval_sample_seed(config, args.sample_seed),
    )
    tables_index = load_tables_json(data_dir / "tables.json")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.checkpoint) / "eval" if args.checkpoint else Path("artifacts/eval/agent")
    )
    predictions_jsonl = Path(args.predictions_jsonl or output_dir / "eval_predictions.jsonl")
    metrics_json = Path(args.metrics_json or output_dir / "eval_metrics.json")

    model_backend = str(args.backend or config.get("model", {}).get("backend", "hf")).strip().lower()
    eval_config = config.get("eval", {})
    concurrency = args.concurrency if args.concurrency is not None else int(eval_config.get("concurrency", 1))
    log_every = args.log_every if args.log_every is not None else int(eval_config.get("log_every", 20))
    if concurrency <= 0:
        parser.error("--concurrency must be greater than zero")
    if log_every <= 0:
        parser.error("--log-every must be greater than zero")
    if model_backend == "hf" and concurrency != 1 and not args.dry_run_gold:
        parser.error("The HF backend supports only --concurrency 1.")

    model_client = (
        None
        if args.dry_run_gold
        else _load_model_client(
            config,
            args.checkpoint,
            args.tokenizer,
            backend=model_backend,
            api_url=args.api_url,
            model_name=args.model_name,
            api_key_env=args.api_key_env,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    )
    checker_model_client = None
    if not args.dry_run_gold and _checker_requested(config, args):
        checker_config = config.get("checker_model") or {}
        if not isinstance(checker_config, dict):
            parser.error("config checker_model must be a mapping when provided")
        checker_backend = str(args.checker_backend or checker_config.get("backend", "openai_chat")).strip().lower()
        checker_model_client = _load_model_client(
            _config_with_model_section(config, "checker_model"),
            checkpoint=None,
            tokenizer_path=None,
            backend=checker_backend,
            api_url=args.checker_api_url,
            model_name=args.checker_model_name,
            api_key_env=args.checker_api_key_env,
            request_timeout_seconds=args.checker_request_timeout_seconds,
        )
    rollout_config = config.get("rollout", {})
    inference_mode = args.inference_mode or _rollout_str(config, "inference_mode", "chain")

    def report_progress(completed: int, total: int) -> None:
        if completed % log_every == 0 or completed == total:
            print(f"completed={completed}/{total}", file=sys.stderr, flush=True)

    rows = evaluate_agent(
        examples,
        tables_index,
        data_dir,
        model_client=model_client,
        checker_model_client=checker_model_client,
        dry_run_gold=args.dry_run_gold,
        max_turns=int(rollout_config.get("max_turns", 2)),
        max_tokens=int(rollout_config.get("max_response_length", 256)),
        temperature=float(rollout_config.get("temperature", 0.0)),
        checker_temperature=args.checker_temperature,
        inference_mode=inference_mode,
        concurrency=concurrency,
        progress_callback=report_progress,
        tree_branch_n=(
            args.tree_branch_n if args.tree_branch_n is not None else _rollout_int(config, "tree_branch_n", 4)
        ),
        tree_beam_size=(
            args.tree_beam_size if args.tree_beam_size is not None else _rollout_int(config, "tree_beam_size", 2)
        ),
        tree_beam_tau=(
            args.tree_beam_tau if args.tree_beam_tau is not None else _rollout_float(config, "tree_beam_tau", 1.0)
        ),
        tree_beam_epsilon_random=(
            args.tree_beam_epsilon_random
            if args.tree_beam_epsilon_random is not None
            else _rollout_float(config, "tree_beam_epsilon_random", 0.0)
        ),
        tree_seed=args.tree_seed if args.tree_seed is not None else _rollout_int(config, "tree_seed", 0),
    )
    metrics = summarize_agent_eval(rows)
    write_eval_outputs(rows, metrics, predictions_jsonl=predictions_jsonl, metrics_json=metrics_json)
    print(
        json.dumps(
            {
                "predictions": str(predictions_jsonl),
                "metrics_json": str(metrics_json),
                **metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
