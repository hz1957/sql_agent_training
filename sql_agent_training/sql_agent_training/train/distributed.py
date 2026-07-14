"""Small helpers for single-node distributed training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DistributedContext:
    """Runtime distributed metadata derived from torchrun environment variables."""

    rank: int
    local_rank: int
    world_size: int
    device: str

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _requested_cuda_device(requested_device: str, local_rank: int) -> str:
    if requested_device == "auto" or requested_device == "cuda":
        return f"cuda:{local_rank}"
    if requested_device.startswith("cuda:"):
        return f"cuda:{local_rank}"
    return requested_device


def init_distributed(requested_device: str = "auto") -> DistributedContext:
    """Initialize torch.distributed when launched by torchrun.

    Single-process runs are left untouched and keep the existing device behavior.
    """

    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover - torch is required for training
        raise RuntimeError("Install torch to use distributed training.") from exc

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if requested_device == "cpu" or not torch.cuda.is_available():
            device = "cpu"
            backend = "gloo"
        else:
            device = _requested_cuda_device(requested_device, local_rank)
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size, device=device)

    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    return DistributedContext(rank=0, local_rank=0, world_size=1, device=device)


def barrier(context: DistributedContext) -> None:
    """Synchronize ranks when distributed is active."""

    if not context.is_distributed:
        return
    import torch.distributed as dist

    dist.barrier()


def broadcast_object(value: Any, context: DistributedContext, *, source_rank: int = 0) -> Any:
    """Broadcast a Python object from source rank."""

    if not context.is_distributed:
        return value
    import torch.distributed as dist

    objects = [value if context.rank == source_rank else None]
    dist.broadcast_object_list(objects, src=source_rank)
    return objects[0]


def all_ranks_true(value: bool, context: DistributedContext) -> bool:
    """Return True only when every rank reports True."""

    if not context.is_distributed:
        return value
    import torch
    import torch.distributed as dist

    device = context.device if context.device.startswith("cuda") else "cpu"
    tensor = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def rank_suffix_path(path: str | Path, context: DistributedContext) -> Path:
    """Keep rank 0 on the canonical path and suffix auxiliary rank outputs."""

    output_path = Path(path)
    if not context.is_distributed or context.rank == 0:
        return output_path
    return output_path.with_name(f"{output_path.stem}_rank{context.rank:03d}{output_path.suffix}")


def unwrap_distributed_model(model: Any) -> Any:
    """Return the underlying module when a model is wrapped in DDP."""

    return getattr(model, "module", model)
