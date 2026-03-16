from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import torch
import torch.distributed as dist

from device_utils import resolve_device


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None = None

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _resolve_process_device(requested: str | None, local_rank: int) -> torch.device:
    if requested is None:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device(f"xpu:{local_rank}")
        return resolve_device(None)

    device = torch.device(requested)
    if device.type in {"cuda", "xpu"}:
        if device.index is None:
            return torch.device(f"{device.type}:{local_rank}")
        if device.index != local_rank:
            raise ValueError(
                f"Requested device {requested!r} conflicts with LOCAL_RANK={local_rank}. "
                "Use '--device cuda' (or '--device xpu') with torchrun, or omit --device."
            )
    return resolve_device(str(device))


def _set_process_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "xpu":
        torch.xpu.set_device(device)


def _choose_backend(device: torch.device) -> str:
    if device.type == "cuda" and sys.platform != "win32":
        return "nccl"
    return "gloo"


def init_distributed(requested_device: str | None = None) -> DistributedContext:
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    if world_size <= 1:
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=resolve_device(requested_device),
            backend=None,
        )

    device = _resolve_process_device(requested_device, local_rank)
    _set_process_device(device)

    backend = _choose_backend(device)
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
    )


def destroy_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier()


def reduce_scalar(value: float, context: DistributedContext, average: bool = True) -> float:
    if not context.enabled:
        return float(value)

    tensor = torch.tensor(float(value), device=context.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= context.world_size
    return float(tensor.item())


def reduce_metrics(metrics: dict[str, float], context: DistributedContext, average: bool = True) -> dict[str, float]:
    if not context.enabled or not metrics:
        return {key: float(value) for key, value in metrics.items()}

    keys = sorted(metrics.keys())
    tensor = torch.tensor([float(metrics[key]) for key in keys], device=context.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= context.world_size
    return {key: float(value) for key, value in zip(keys, tensor.tolist())}
