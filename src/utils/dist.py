from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int


def init_distributed(backend: str = "nccl") -> DistInfo:
    """Initialize torch.distributed if launched with torchrun/SLURM."""
    if dist.is_available() and not dist.is_initialized():
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if world_size > 1:
            dist.init_process_group(backend=backend, init_method="env://")
            torch.cuda.set_device(local_rank)
            return DistInfo(True, rank, world_size, local_rank)

    return DistInfo(False, 0, 1, 0)


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_gather_object(obj):
    """Gather a python object from all ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(out, obj)
    return out


def broadcast_tensor(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from src to all ranks (no-op if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(t, src)
    return t
