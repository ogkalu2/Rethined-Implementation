from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from device_utils import get_device_name


class NullSummaryWriter:
    """Fallback writer used when TensorBoard is unavailable."""

    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _is_tensorboard_compatible() -> bool:
    """Check TensorBoard's protobuf compatibility before importing it."""
    try:
        from google.protobuf.message_factory import MessageFactory
    except Exception:
        return False
    return hasattr(MessageFactory, "GetPrototype")


def create_summary_writer(log_dir: Path):
    if not _is_tensorboard_compatible():
        print(
            "TensorBoard logging disabled: installed protobuf is incompatible "
            "with torch.utils.tensorboard. Pin protobuf<6 to re-enable it."
        )
        return NullSummaryWriter()

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(
            "TensorBoard logging disabled because SummaryWriter could not be "
            f"imported: {exc.__class__.__name__}: {exc}"
        )
        return NullSummaryWriter()

    return SummaryWriter(log_dir)


def build_model_config(cfg):
    coarse_cfg = cfg["model"]["coarse_model"]
    inpainter_params = {
        key: value
        for key, value in cfg["model"]["inpainter"].items()
        if key != "copy_mode"
    }
    return {
        "coarse_model": {
            "class": coarse_cfg.get("class", "CoarseModel"),
            "parameters": {key: value for key, value in coarse_cfg.items() if key != "class"},
        },
        "inpainter": {
            "inpainter_class": "PatchInpainting",
            "params": inpainter_params,
        },
    }


def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def steps_from_epochs(epoch_count: float, steps_per_epoch: float) -> int:
    return max(1, int(math.ceil(max(float(epoch_count), 0.0) * max(steps_per_epoch, 1e-8))))


def epochs_from_steps(step_count: int, steps_per_epoch: float) -> float:
    return float(step_count) / max(steps_per_epoch, 1e-8)


def next_epoch_interval_target(current_step: int, steps_per_epoch: float, interval_epochs: float) -> float:
    current_epoch = epochs_from_steps(current_step, steps_per_epoch)
    completed_intervals = math.floor(current_epoch / interval_epochs + 1e-9)
    return (completed_intervals + 1) * interval_epochs


def composite_with_known(pred, target, mask):
    return pred * mask + target * (1 - mask)


def gaussian_prefilter_downsample(image: torch.Tensor, model_image_size: int, blur_layer=None):
    if image.shape[-2:] == (model_image_size, model_image_size):
        return image
    if blur_layer is not None:
        image = blur_layer(image)
    return F.interpolate(image, size=(model_image_size, model_image_size), mode="bicubic", align_corners=False)


def prepare_multiscale_batch(batch, device, model_image_size: int, blur_layer=None):
    image_hr = batch["image"].to(device, non_blocking=True)
    mask_hr = batch["mask"].to(device, non_blocking=True)
    masked_image_hr = image_hr * (1 - mask_hr)

    if image_hr.shape[-2:] == (model_image_size, model_image_size):
        image = image_hr
        refine_target = image_hr
        mask = mask_hr
        masked_image = masked_image_hr
    else:
        refine_target = F.interpolate(
            image_hr,
            size=(model_image_size, model_image_size),
            mode="bicubic",
            align_corners=False,
        )
        image = gaussian_prefilter_downsample(image_hr, model_image_size, blur_layer=blur_layer)
        mask = F.interpolate(mask_hr, size=(model_image_size, model_image_size), mode="nearest")
        mask = (mask > 0.5).to(image.dtype)
        masked_image = image * (1 - mask)

    return {
        "image": image,
        "refine_target": refine_target,
        "mask": mask,
        "masked_image": masked_image,
        "image_hr": image_hr,
        "mask_hr": mask_hr,
        "masked_image_hr": masked_image_hr,
        "has_hr_target": image_hr.shape[-2:] != image.shape[-2:],
    }


def print_device_banner(device):
    print(f"Device: {device}")
    if device.type in {"cuda", "xpu"}:
        print(f"Accelerator: {get_device_name(device)}")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metric_to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def masked_l1(pred, target, mask):
    denom = (mask.sum(dim=(1, 2, 3)) * pred.shape[1]).clamp_min(1e-8)
    return (torch.abs(pred - target) * mask).sum(dim=(1, 2, 3)) / denom


def mean_metric(values):
    return float(sum(values) / len(values)) if values else None


def is_better_metric(candidate, best, mode):
    if candidate is None:
        return False
    if best is None:
        return True
    if mode == "min":
        return candidate < best
    if mode == "max":
        return candidate > best
    raise ValueError(f"Unsupported best metric mode: {mode}")
