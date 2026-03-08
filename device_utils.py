"""Shared device helpers for CPU, CUDA, and Intel XPU execution."""

from __future__ import annotations

import time

import torch

ACCELERATOR_DEVICE_TYPES = ("xpu", "cuda")
AUTO_DEVICE_PREFERENCE = ("xpu", "cuda", "cpu")


def _as_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def is_device_available(device_type: str) -> bool:
    """Return whether the requested device type is usable in this runtime."""
    if device_type == "cpu":
        return True
    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    return False


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve an explicit or automatic device choice.

    Auto mode prefers XPU first so Intel GPU systems do not silently fall back
    to CPU when CUDA is unavailable.
    """
    if requested is not None:
        device = torch.device(requested)
        if not is_device_available(device.type):
            raise ValueError(f"Requested device '{requested}' is not available")
        return device

    for device_type in AUTO_DEVICE_PREFERENCE:
        if is_device_available(device_type):
            return torch.device(device_type)

    return torch.device("cpu")


def is_accelerator_device(device: torch.device | str) -> bool:
    return _as_device(device).type in ACCELERATOR_DEVICE_TYPES


def is_amp_enabled(device: torch.device | str, enabled: bool) -> bool:
    return enabled and is_accelerator_device(device)


def get_autocast_device_type(device: torch.device | str) -> str:
    return _as_device(device).type


def get_device_name(device: torch.device | str) -> str:
    device = _as_device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "xpu":
        return torch.xpu.get_device_name(device)
    return "CPU"


def get_peak_memory_allocated_gb(device: torch.device | str) -> float | None:
    device = _as_device(device)
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9
    if device.type == "xpu":
        return torch.xpu.max_memory_allocated(device) / 1e9
    return None


def synchronize_device(device: torch.device | str) -> None:
    device = _as_device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def empty_device_cache(device: torch.device | str) -> None:
    device = _as_device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()


def time_device_call(fn, device: torch.device | str) -> float:
    """Run a callable and return elapsed time in milliseconds."""
    synchronize_device(device)
    start = time.perf_counter()
    fn()
    synchronize_device(device)
    return (time.perf_counter() - start) * 1000.0
