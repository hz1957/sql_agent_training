"""Download a Hugging Face model snapshot into the local data directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a model snapshot for local smoke tests.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--output-dir", default="data/models/SmolLM2-135M-Instruct")
    parser.add_argument("--cache-dir", default="data/.hf_cache")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=output_dir,
        cache_dir=Path(args.cache_dir),
    )
    print(path)


if __name__ == "__main__":
    main()
