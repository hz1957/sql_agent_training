"""Download a Hugging Face model snapshot into the local data directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a model snapshot for local smoke tests.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--output-dir", default="data/models/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--cache-dir", default="data/.hf_cache")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum concurrent Hugging Face download workers. Keep this low on memory-constrained servers.",
    )
    args = parser.parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=output_dir,
        cache_dir=Path(args.cache_dir),
        max_workers=args.max_workers,
    )
    print(path)


if __name__ == "__main__":
    main()
