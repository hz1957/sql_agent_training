"""Download Spider database assets that are not included in xlangai/spider."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def _copy_path(source: Path, target: Path, *, overwrite: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing expected source asset: {source}")
    if target.exists() and not overwrite:
        print(f"skip existing: {target}")
        return
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f"copied: {source} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official Spider tables.json and SQLite databases.")
    parser.add_argument("--data-dir", default="data/spider")
    parser.add_argument("--repo-id", default="dreamerdeo/multispider")
    parser.add_argument("--repo-subdir", default="dataset/spider")
    parser.add_argument("--cache-dir", default="data/.hf_cache")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=[f"{args.repo_subdir}/tables.json", f"{args.repo_subdir}/database/**"],
            cache_dir=cache_dir,
        )
    )
    source_root = snapshot_path / args.repo_subdir

    _copy_path(source_root / "tables.json", data_dir / "tables.json", overwrite=args.overwrite)
    _copy_path(source_root / "database", data_dir / "database", overwrite=args.overwrite)


if __name__ == "__main__":
    main()
