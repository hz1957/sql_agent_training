"""Evaluate generated SQL predictions with Spider execution reward."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sql_agent_training.data.spider_dataset import expected_sqlite_path
from sql_agent_training.reward.spider_reward import spider_execution_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JSONL predictions against Spider gold SQL.")
    parser.add_argument("--predictions", required=True, help="JSONL with db_id, prediction, and gold_sql fields.")
    parser.add_argument("--data-dir", default="data/spider")
    args = parser.parse_args()

    total = 0
    correct = 0.0
    with Path(args.predictions).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            db_path = expected_sqlite_path(args.data_dir, row["db_id"])
            correct += spider_execution_reward(row["prediction"], row["gold_sql"], db_path)
            total += 1
    accuracy = correct / total if total else 0.0
    print(json.dumps({"total": total, "execution_accuracy": accuracy}, indent=2))


if __name__ == "__main__":
    main()
