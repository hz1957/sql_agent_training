"""Merge a PEFT LoRA adapter into a base causal LM.

This is primarily used before verl GRPO. verl/FSDP may initialize the actor on
meta tensors, and loading an existing LoRA adapter in that path can emit
non-meta-to-meta no-op warnings. Merging the SFT LoRA into a normal HF model
directory avoids that adapter-loading path; GRPO can then train a fresh LoRA on
top of the merged SFT model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Base HF model path or id.")
    parser.add_argument("--adapter", required=True, help="PEFT LoRA adapter checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Directory for the merged HF model.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Passed to transformers.from_pretrained. Use 'cpu' to merge on CPU.",
    )
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _same_existing_path(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and left.resolve() == right.resolve()
    except OSError:
        return False


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} already exists and is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if _same_existing_path(output_dir, Path(args.base_model)) or _same_existing_path(output_dir, Path(args.adapter)):
        raise ValueError("--output-dir must be different from --base-model and --adapter.")
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict[str, Any] = {
        "torch_dtype": _torch_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map:
        load_kwargs["device_map"] = args.device_map

    print(f"merge_lora base_model={args.base_model}")
    print(f"merge_lora adapter={args.adapter}")
    print(f"merge_lora output_dir={output_dir}")
    print(f"merge_lora dtype={args.dtype} device_map={args.device_map}")

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    peft_model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(output_dir)

    merge_info = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_shard_size": args.max_shard_size,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "merged SFT starting policy for verl GRPO fresh LoRA training",
    }
    (output_dir / "merge_info.json").write_text(json.dumps(merge_info, indent=2) + "\n", encoding="utf-8")

    print("merge_lora done")


if __name__ == "__main__":
    main()
