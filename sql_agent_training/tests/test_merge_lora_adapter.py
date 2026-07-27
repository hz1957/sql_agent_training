from pathlib import Path

import pytest

from scripts.merge_lora_adapter import _prepare_output_dir, _same_existing_path


def test_same_existing_path_detects_same_directory(tmp_path: Path) -> None:
    assert _same_existing_path(tmp_path, tmp_path)


def test_prepare_output_dir_rejects_non_empty_without_overwrite(tmp_path: Path) -> None:
    (tmp_path / "sentinel.txt").write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _prepare_output_dir(tmp_path, overwrite=False)


def test_prepare_output_dir_allows_empty_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "merged"

    _prepare_output_dir(output_dir, overwrite=False)

    assert output_dir.is_dir()
