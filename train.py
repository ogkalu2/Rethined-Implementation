"""RETHINED training entry point.

Usage:
    # Normal training
    python train.py --config configs/train_celeba_256.yaml

    # Overfit test (8 images, 2000 steps)
    python train.py --config configs/train_celeba_256.yaml --overfit 8 --steps 2000

    # Resume from checkpoint
    python train.py --config configs/train_celeba_256.yaml --resume checkpoints/step_10000.pth
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from device_utils import (
    get_autocast_device_type,
    get_device_name,
    get_peak_memory_allocated_gb,
    is_amp_enabled,
    resolve_device,
)
from model import AttentionUpscaling, InpaintingModel
from losses import InpaintingLoss
from data.dataset import get_dataloader

REFINEMENT_TIE_EPS = 1e-6


def build_model_config(cfg):
    """Build the model config dict from YAML config."""
    coarse_cfg = cfg["model"]["coarse_model"]
    return {
        "coarse_model": {
            "class": coarse_cfg.get("class", "CoarseModel"),
            "parameters": {k: v for k, v in coarse_cfg.items() if k != "class"},
        },
        "generator": {
            "generator_class": "PatchInpainting",
            "params": {k: v for k, v in cfg["model"]["generator"].items()},
        },
    }


def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    """Warmup + cosine decay learning rate schedule."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_refinement_scale(step, start_step, ramp_steps):
    """Coarse-only warmup, then linear ramp-in for refinement branch."""
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min((step - start_step) / ramp_steps, 1.0)


def save_checkpoint(model, optimizer, scaler, step, loss_dict, cfg, path):
    """Save training checkpoint."""
    try:
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss_dict": loss_dict,
            "config": cfg,
        }, path)
        return True
    except Exception as exc:
        print(f"\nCheckpoint save failed at {path}: {exc}")
        return False


def load_model_checkpoint(model, state_dict):
    """Load model state while allowing newly added HR rescoring modules."""
    model_state = model.state_dict()
    filtered_state = {}
    dropped_shape_mismatches = []
    allowed_prefixes = ("generator.hr_",)
    allowed_missing_substrings = (
        ".depthwise_scale.",
        ".depthwise_scale_bn.",
        ".depthwise_identity_bn.",
    )
    for key, value in state_dict.items():
        if key not in model_state:
            filtered_state[key] = value
            continue
        if model_state[key].shape != value.shape:
            if key.startswith(allowed_prefixes):
                dropped_shape_mismatches.append(key)
                continue
            raise RuntimeError(
                "Checkpoint/model mismatch: "
                f"shape mismatch for {key}: checkpoint {tuple(value.shape)} vs model {tuple(model_state[key].shape)}"
            )
        filtered_state[key] = value

    incompatible = model.load_state_dict(filtered_state, strict=False)
    missing = [
        k for k in incompatible.missing_keys
        if not k.startswith(allowed_prefixes) and not any(token in k for token in allowed_missing_substrings)
    ]
    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith(allowed_prefixes)]
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing keys: {missing}")
        if unexpected:
            problems.append(f"unexpected keys: {unexpected}")
        raise RuntimeError("Checkpoint/model mismatch: " + "; ".join(problems))
    if incompatible.missing_keys or dropped_shape_mismatches:
        print("Initialized new HR/rep-branch parameters from scratch.")


def composite_with_known(pred, target, mask):
    """Preserve known pixels exactly and only use predictions inside the hole."""
    return pred * mask + target * (1 - mask)


def gaussian_prefilter_downsample(image: torch.Tensor, model_image_size: int, blur_layer=None):
    """Low-pass filter before subsampling to match the paper's Section 3.4 assumption."""
    if image.shape[-2:] == (model_image_size, model_image_size):
        return image

    if blur_layer is not None:
        image = blur_layer(image)

    return F.interpolate(image, size=(model_image_size, model_image_size), mode="bicubic", align_corners=False)


def prepare_multiscale_batch(batch, device, model_image_size: int, blur_layer=None):
    """Prepare fixed-resolution model inputs from a possibly larger HR crop."""
    image_hr = batch["image"].to(device, non_blocking=True)
    mask_hr = batch["mask"].to(device, non_blocking=True)
    masked_image_hr = image_hr * (1 - mask_hr)

    if image_hr.shape[-2:] == (model_image_size, model_image_size):
        image = image_hr
        mask = mask_hr
        masked_image = masked_image_hr
    else:
        image = gaussian_prefilter_downsample(image_hr, model_image_size, blur_layer=blur_layer)
        mask = F.interpolate(mask_hr, size=(model_image_size, model_image_size), mode="nearest")
        mask = (mask > 0.5).to(image.dtype)
        masked_image = image * (1 - mask)

    return {
        "image": image,
        "mask": mask,
        "masked_image": masked_image,
        "image_hr": image_hr,
        "mask_hr": mask_hr,
        "masked_image_hr": masked_image_hr,
        "has_hr_supervision": image_hr.shape[-2:] != image.shape[-2:],
    }


def print_device_banner(device: torch.device):
    """Print the active training device and accelerator name when present."""
    print(f"Device: {device}")
    if device.type in {"cuda", "xpu"}:
        print(f"Accelerator: {get_device_name(device)}")


def clear_device_cache(device: torch.device) -> None:
    """Release cached accelerator memory between large eval phases."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()


def validate_joint_hr_pipeline(cfg, model):
    """Validate the config needed to train the LR+HR pipeline jointly."""
    joint_hr_pipeline = cfg["training"].get("joint_hr_pipeline", False)
    model_image_size = model.generator.image_size
    data_image_size = cfg["data"]["image_size"]

    if joint_hr_pipeline:
        if data_image_size <= model_image_size:
            raise ValueError(
                "training.joint_hr_pipeline=true requires data.image_size > "
                f"model.generator.image_size ({data_image_size} <= {model_image_size})"
            )
        if data_image_size % model_image_size != 0:
            raise ValueError(
                "training.joint_hr_pipeline requires data.image_size to be an integer "
                f"multiple of model.generator.image_size ({data_image_size} vs {model_image_size})"
            )
        if model.generator.nheads != 1:
            raise ValueError(
                "training.joint_hr_pipeline currently requires nheads=1 because "
                "AttentionUpscaling expects a single learned attention map."
            )

    return joint_hr_pipeline, model_image_size


def compute_hr_refined(attn_upscaler, batch_views, refined_lr, attn_map):
    """Run the attention upscaler and preserve known HR pixels exactly."""
    refined_lr = refined_lr.clamp(0, 1)
    hr_refined_raw = attn_upscaler(
        batch_views["masked_image_hr"],
        refined_lr,
        attn_map,
        mask_hr=batch_views["mask_hr"],
    ).clamp(0, 1)
    hr_refined = composite_with_known(hr_refined_raw, batch_views["image_hr"], batch_views["mask_hr"])
    hr_base_raw = F.interpolate(
        refined_lr,
        size=batch_views["image_hr"].shape[-2:],
        mode="bicubic",
        align_corners=False,
    ).clamp(0, 1)
    hr_base = composite_with_known(hr_base_raw, batch_views["image_hr"], batch_views["mask_hr"])
    return hr_refined_raw, hr_refined, hr_base_raw, hr_base


def set_parameter_trainability(model, freeze_coarse: bool = False):
    """Configure trainability for the paper path."""
    for param in model.parameters():
        param.requires_grad = True

    if freeze_coarse:
        for param in model.coarse_model.parameters():
            param.requires_grad = False
        model.coarse_model.eval()


def should_freeze_coarse(step, initial_freeze_coarse, coarse_freeze_step=None):
    """Return whether the coarse branch should be frozen at this step."""
    if initial_freeze_coarse:
        return True
    if coarse_freeze_step is None:
        return False
    return step >= coarse_freeze_step


def masked_l1_per_sample(pred, target, mask):
    """Per-sample masked L1 aligned with the reported gain metrics."""
    denom = (mask.sum(dim=(1, 2, 3)) * pred.shape[1]).clamp_min(1e-8)
    return (torch.abs(pred - target) * mask).sum(dim=(1, 2, 3)) / denom


def resize_for_aux_loss(pred, target, mask, image_size):
    """Optionally downsample tensors for auxiliary HR losses to save memory."""
    if image_size is None:
        return pred, target, mask
    target_h, target_w = target.shape[-2:]
    if target_h <= image_size and target_w <= image_size:
        return pred, target, mask

    size = (image_size, image_size)
    pred_small = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
    target_small = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
    mask_small = F.interpolate(mask, size=size, mode="nearest")
    mask_small = (mask_small > 0.5).to(pred_small.dtype)
    return pred_small, target_small, mask_small


def extract_mask_focus_crops(pred, target, mask, crop_size, pad_frac=0.25):
    """Crop around each hole with some context, then resize for focused perceptual loss."""
    if crop_size is None:
        return pred, target

    crops_pred = []
    crops_target = []
    batch_size, _, height, width = pred.shape
    mask_bool = mask[:, 0] > 0.5
    for batch_idx in range(batch_size):
        nonzero = torch.nonzero(mask_bool[batch_idx], as_tuple=False)
        if nonzero.numel() == 0:
            y0, x0, y1, x1 = 0, 0, height, width
        else:
            y0 = int(nonzero[:, 0].min().item())
            y1 = int(nonzero[:, 0].max().item()) + 1
            x0 = int(nonzero[:, 1].min().item())
            x1 = int(nonzero[:, 1].max().item()) + 1
            span = max(y1 - y0, x1 - x0)
            pad = max(4, int(round(span * pad_frac)))
            y0 = max(0, y0 - pad)
            y1 = min(height, y1 + pad)
            x0 = max(0, x0 - pad)
            x1 = min(width, x1 + pad)

        pred_crop = pred[batch_idx:batch_idx + 1, :, y0:y1, x0:x1]
        target_crop = target[batch_idx:batch_idx + 1, :, y0:y1, x0:x1]
        if pred_crop.shape[-2:] != (crop_size, crop_size):
            pred_crop = F.interpolate(pred_crop, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
            target_crop = F.interpolate(target_crop, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        crops_pred.append(pred_crop)
        crops_target.append(target_crop)
    return torch.cat(crops_pred, dim=0), torch.cat(crops_target, dim=0)


def compute_train_loss(
    criterion,
    coarse_raw,
    refined_raw,
    target,
    mask,
    refinement_loss_scale=1.0,
    coarse_composite=None,
    refined_composite=None,
    hr_refined_raw=None,
    hr_target=None,
    hr_mask=None,
    hr_base_raw=None,
    hr_base_composite=None,
    hr_refined_composite=None,
):
    """Compute the paper-path training loss.

    L1 must run on the raw network outputs; otherwise the valid-region term is
    always zero after compositing known pixels back from the target.
    """
    l1_coarse = criterion.masked_l1(coarse_raw, target, mask)
    l1_refined = criterion.masked_l1(refined_raw, target, mask)

    zero = refined_raw.new_zeros(())
    perceptual = zero
    style = zero
    hr_l1_refined = zero
    hr_perceptual = zero
    hr_style = zero
    hr_delta = zero
    gain_ratio = zero
    hr_gain_ratio = zero
    refined_for_perceptual = (
        refined_composite if refined_composite is not None else refined_raw.clamp(0, 1)
    ).clamp(0, 1)

    if criterion.perceptual_weight > 0:
        perceptual_pred, perceptual_target = extract_mask_focus_crops(
            refined_for_perceptual,
            target,
            mask,
            getattr(criterion, "perceptual_crop_size", None),
            getattr(criterion, "perceptual_crop_pad_frac", 0.25),
        )
        perceptual = criterion.perceptual_loss(perceptual_pred, perceptual_target)
    if criterion.style_weight > 0:
        style_pred, style_target = extract_mask_focus_crops(
            refined_for_perceptual,
            target,
            mask,
            getattr(criterion, "perceptual_crop_size", None),
            getattr(criterion, "perceptual_crop_pad_frac", 0.25),
        )
        style = criterion.style_loss(style_pred, style_target)

    if (
        criterion.gain_ratio_weight > 0
        and coarse_composite is not None
        and refined_composite is not None
    ):
        coarse_hole = masked_l1_per_sample(coarse_composite, target, mask).detach()
        refined_hole = masked_l1_per_sample(refined_composite, target, mask)
        gain_ratio = (refined_hole / coarse_hole.clamp_min(1e-4)).mean()

    if (
        hr_refined_raw is not None
        and hr_target is not None
        and hr_mask is not None
    ):
        hr_for_perceptual = (
            hr_refined_composite if hr_refined_composite is not None else hr_refined_raw
        ).clamp(0, 1)
        hr_aux_image_size = getattr(criterion, "hr_aux_image_size", None)
        hr_perceptual_crop_size = getattr(criterion, "hr_perceptual_crop_size", None)
        if hr_perceptual_crop_size is None:
            hr_perceptual_crop_size = hr_aux_image_size
        if criterion.hr_refined_weight > 0:
            # HR inference restores known pixels after attention transfer, so the
            # HR L1 term should supervise that same composited prediction.
            hr_l1_refined = criterion.masked_l1(hr_for_perceptual, hr_target, hr_mask)
        if criterion.hr_delta_weight > 0 and hr_base_raw is not None:
            hr_base_raw_detached = hr_base_raw.detach()
            hr_delta_pred = hr_refined_raw - hr_base_raw_detached
            hr_delta_target = hr_target - hr_base_raw_detached
            hr_delta_pred, hr_delta_target, hr_delta_mask = resize_for_aux_loss(
                hr_delta_pred,
                hr_delta_target,
                hr_mask,
                hr_aux_image_size,
            )
            hr_delta = criterion.masked_l1(
                hr_delta_pred,
                hr_delta_target,
                hr_delta_mask,
            )
        if criterion.hr_perceptual_weight > 0:
            hr_perceptual_pred, hr_perceptual_target = extract_mask_focus_crops(
                hr_for_perceptual,
                hr_target,
                hr_mask,
                hr_perceptual_crop_size,
                getattr(criterion, "perceptual_crop_pad_frac", 0.25),
            )
            hr_perceptual = criterion.perceptual_loss(hr_perceptual_pred, hr_perceptual_target)
        if criterion.hr_style_weight > 0:
            hr_style_pred, hr_style_target = extract_mask_focus_crops(
                hr_for_perceptual,
                hr_target,
                hr_mask,
                hr_perceptual_crop_size,
                getattr(criterion, "perceptual_crop_pad_frac", 0.25),
            )
            hr_style = criterion.style_loss(hr_style_pred, hr_style_target)
        if criterion.hr_gain_ratio_weight > 0 and hr_base_composite is not None:
            hr_base_hole = masked_l1_per_sample(hr_base_composite, hr_target, hr_mask).detach()
            hr_refined_hole = masked_l1_per_sample(hr_for_perceptual, hr_target, hr_mask)
            hr_gain_ratio = (hr_refined_hole / hr_base_hole.clamp_min(1e-4)).mean()

    coarse_term = criterion.coarse_weight * l1_coarse
    refinement_term = (
        criterion.refined_weight * l1_refined
        + criterion.perceptual_weight * perceptual
        + criterion.style_weight * style
        + criterion.hr_refined_weight * hr_l1_refined
        + criterion.hr_delta_weight * hr_delta
        + criterion.hr_perceptual_weight * hr_perceptual
        + criterion.hr_style_weight * hr_style
        + criterion.gain_ratio_weight * gain_ratio
        + criterion.hr_gain_ratio_weight * hr_gain_ratio
    )
    total = coarse_term + refinement_loss_scale * refinement_term
    total_normalized = coarse_term + refinement_term

    loss_dict = {
        "l1_coarse": l1_coarse.item(),
        "l1_refined": l1_refined.item(),
        "perceptual": perceptual.item(),
        "style": style.item(),
        "hr_l1_refined": hr_l1_refined.item(),
        "hr_delta": hr_delta.item(),
        "hr_perceptual": hr_perceptual.item(),
        "hr_style": hr_style.item(),
        "gain_ratio": gain_ratio.item(),
        "hr_gain_ratio": hr_gain_ratio.item(),
        "coarse_term": coarse_term.item(),
        "refinement_term": refinement_term.item(),
        "total": total.item(),
        "total_normalized": total_normalized.item(),
    }
    return total, loss_dict


def write_status(
    log_dir,
    step,
    total_steps,
    loss_dict,
    running_loss,
    running_normalized_loss,
    lr,
    start_time,
    extra=None,
):
    """Write training status to JSON file for remote monitoring."""
    elapsed = time.time() - start_time
    steps_done = max(step, 1)
    it_per_sec = steps_done / elapsed if elapsed > 0 else 0
    eta_sec = (total_steps - step) / it_per_sec if it_per_sec > 0 else 0

    status = {
        "step": step,
        "total_steps": total_steps,
        "progress_pct": round(step / total_steps * 100, 2),
        "running_loss": round(running_loss, 4),
        "running_normalized_loss": round(running_normalized_loss, 4),
        "loss": {k: round(v, 4) for k, v in loss_dict.items()},
        "lr": lr,
        "it_per_sec": round(it_per_sec, 2),
        "elapsed_sec": round(elapsed),
        "elapsed_human": f"{elapsed/3600:.1f}h",
        "eta_sec": round(eta_sec),
        "eta_human": f"{eta_sec/3600:.1f}h",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        status.update(extra)

    status_path = Path(log_dir) / "training_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)


def append_validation_history(log_dir, step, health):
    """Append validation metrics to a JSONL history for later inspection."""
    history_path = Path(log_dir) / "validation_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"step": step, **health}
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def add_distribution_metrics(result, values, prefix):
    """Add quartile-style summary metrics for a list of scalar values."""
    if not values:
        result[f"{prefix}_p25"] = None
        result[f"{prefix}_p50"] = None
        result[f"{prefix}_p75"] = None
        result[f"{prefix}_min"] = None
        result[f"{prefix}_max"] = None
        return

    tensor = torch.tensor(values, dtype=torch.float32)
    result[f"{prefix}_p25"] = float(torch.quantile(tensor, 0.25).item())
    result[f"{prefix}_p50"] = float(torch.quantile(tensor, 0.50).item())
    result[f"{prefix}_p75"] = float(torch.quantile(tensor, 0.75).item())
    result[f"{prefix}_min"] = float(tensor.min().item())
    result[f"{prefix}_max"] = float(tensor.max().item())


def save_vis(writer, batch, coarse, refined, step, log_dir=None):
    """Save visualization to TensorBoard and as PNG file."""
    n = min(4, batch["image"].shape[0])
    gt = batch["image"][:n].cpu()
    masked = batch["masked_image"][:n].cpu()
    mask_vis = batch["mask"][:n].repeat(1, 3, 1, 1).cpu()
    coarse_vis = coarse[:n].detach().clamp(0, 1).cpu()
    refined_vis = refined[:n].detach().clamp(0, 1).cpu()

    grid = make_grid(
        torch.cat([masked, mask_vis, coarse_vis, refined_vis, gt], dim=0),
        nrow=n, normalize=False, padding=2,
    )
    writer.add_image("samples/masked|mask|coarse|refined|gt", grid, step)

    # Save as PNG file for visual inspection
    if log_dir is not None:
        vis_dir = Path(log_dir) / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_image(grid, vis_dir / f"step_{step:06d}.png")


@torch.no_grad()
def evaluate_refinement_health(
    model,
    dataloader,
    device,
    use_amp,
    max_batches=8,
    freeze_coarse=False,
    model_image_size=None,
    joint_hr_pipeline=False,
    refinement_eval_scale=None,
):
    """Quick validation focused on whether refinement helps or hurts."""
    model.eval()
    amp_device_type = get_autocast_device_type(device)
    amp_enabled = is_amp_enabled(device, use_amp)
    if model_image_size is None:
        model_image_size = model.generator.image_size
    attn_upscaler = AttentionUpscaling(model.generator) if joint_hr_pipeline else None
    original_refinement_scale = model.generator.refinement_runtime_scale
    if refinement_eval_scale is not None:
        model.generator.set_refinement_runtime_scale(refinement_eval_scale)

    stats = {
        "masked_l1_coarse": [],
        "masked_l1_refined": [],
        "valid_l1_coarse": [],
        "valid_l1_refined": [],
        "raw_valid_l1_coarse": [],
        "raw_valid_l1_refined": [],
        "refined_better_flags": [],
        "refined_tie_flags": [],
        "refined_worse_flags": [],
        "masked_delta_mean": [],
        "hr_masked_l1_base": [],
        "hr_masked_l1_refined": [],
        "hr_refined_better_flags": [],
        "hr_refined_tie_flags": [],
        "hr_refined_worse_flags": [],
    }
    gain_abs_values = []
    gain_pct_values = []
    hr_gain_abs_values = []
    hr_gain_pct_values = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
            break

        batch_views = prepare_multiscale_batch(
            batch,
            device,
            model_image_size,
            blur_layer=model.generator.final_gaussian_blur,
        )
        image = batch_views["image"]
        mask = batch_views["mask"]
        masked_image = batch_views["masked_image"]

        with torch.amp.autocast(amp_device_type, enabled=amp_enabled):
            refined_raw, attn_map, coarse_raw = model(masked_image, mask)

        refined_raw = refined_raw.clamp(0, 1)
        coarse_raw = coarse_raw.clamp(0, 1)
        refined = composite_with_known(refined_raw, image, mask)
        coarse = composite_with_known(coarse_raw, image, mask)
        valid_mask = 1 - mask

        hole_denom = (mask.sum(dim=(1, 2, 3)) * image.shape[1]).clamp_min(1e-8)
        valid_denom = (valid_mask.sum(dim=(1, 2, 3)) * image.shape[1]).clamp_min(1e-8)

        coarse_hole = (torch.abs(coarse - image) * mask).sum(dim=(1, 2, 3)) / hole_denom
        refined_hole = (torch.abs(refined - image) * mask).sum(dim=(1, 2, 3)) / hole_denom
        coarse_valid = (torch.abs(coarse_raw - image) * valid_mask).sum(dim=(1, 2, 3)) / valid_denom
        refined_valid = (torch.abs(refined_raw - image) * valid_mask).sum(dim=(1, 2, 3)) / valid_denom
        raw_coarse_valid = coarse_valid
        raw_refined_valid = refined_valid
        masked_delta = (torch.abs(refined - coarse) * mask).sum(dim=(1, 2, 3)) / hole_denom
        gain_abs = coarse_hole - refined_hole
        gain_pct = gain_abs / coarse_hole.clamp_min(1e-8) * 100.0
        better_flags = (gain_abs > REFINEMENT_TIE_EPS).float()
        tie_flags = (torch.abs(gain_abs) <= REFINEMENT_TIE_EPS).float()
        worse_flags = (gain_abs < -REFINEMENT_TIE_EPS).float()

        stats["masked_l1_coarse"].extend(coarse_hole.detach().cpu().tolist())
        stats["masked_l1_refined"].extend(refined_hole.detach().cpu().tolist())
        stats["valid_l1_coarse"].extend(coarse_valid.detach().cpu().tolist())
        stats["valid_l1_refined"].extend(refined_valid.detach().cpu().tolist())
        stats["raw_valid_l1_coarse"].extend(raw_coarse_valid.detach().cpu().tolist())
        stats["raw_valid_l1_refined"].extend(raw_refined_valid.detach().cpu().tolist())
        stats["masked_delta_mean"].extend(masked_delta.detach().cpu().tolist())
        stats["refined_better_flags"].extend(better_flags.detach().cpu().tolist())
        stats["refined_tie_flags"].extend(tie_flags.detach().cpu().tolist())
        stats["refined_worse_flags"].extend(worse_flags.detach().cpu().tolist())
        gain_abs_values.extend(gain_abs.detach().cpu().tolist())
        gain_pct_values.extend(gain_pct.detach().cpu().tolist())

        if joint_hr_pipeline and batch_views["has_hr_supervision"]:
            with torch.amp.autocast(amp_device_type, enabled=amp_enabled):
                _, hr_refined, _, hr_base = compute_hr_refined(
                    attn_upscaler,
                    batch_views,
                    refined,
                    attn_map,
                )

            image_hr = batch_views["image_hr"]
            mask_hr = batch_views["mask_hr"]
            hole_denom_hr = (mask_hr.sum(dim=(1, 2, 3)) * image_hr.shape[1]).clamp_min(1e-8)
            hr_base_hole = (torch.abs(hr_base - image_hr) * mask_hr).sum(dim=(1, 2, 3)) / hole_denom_hr
            hr_refined_hole = (torch.abs(hr_refined - image_hr) * mask_hr).sum(dim=(1, 2, 3)) / hole_denom_hr
            hr_gain_abs = hr_base_hole - hr_refined_hole
            hr_gain_pct = hr_gain_abs / hr_base_hole.clamp_min(1e-8) * 100.0
            hr_better_flags = (hr_gain_abs > REFINEMENT_TIE_EPS).float()
            hr_tie_flags = (torch.abs(hr_gain_abs) <= REFINEMENT_TIE_EPS).float()
            hr_worse_flags = (hr_gain_abs < -REFINEMENT_TIE_EPS).float()

            stats["hr_masked_l1_base"].extend(hr_base_hole.detach().cpu().tolist())
            stats["hr_masked_l1_refined"].extend(hr_refined_hole.detach().cpu().tolist())
            stats["hr_refined_better_flags"].extend(hr_better_flags.detach().cpu().tolist())
            stats["hr_refined_tie_flags"].extend(hr_tie_flags.detach().cpu().tolist())
            stats["hr_refined_worse_flags"].extend(hr_worse_flags.detach().cpu().tolist())
            hr_gain_abs_values.extend(hr_gain_abs.detach().cpu().tolist())
            hr_gain_pct_values.extend(hr_gain_pct.detach().cpu().tolist())

    result = {}
    for key, values in stats.items():
        result[key] = float(sum(values) / len(values)) if values else None

    coarse_masked = result["masked_l1_coarse"]
    refined_masked = result["masked_l1_refined"]
    if coarse_masked is not None and refined_masked is not None and coarse_masked > 0:
        result["refinement_gain_abs"] = float(coarse_masked - refined_masked)
        result["refinement_gain_pct"] = float((coarse_masked - refined_masked) / coarse_masked * 100.0)
    else:
        result["refinement_gain_abs"] = None
        result["refinement_gain_pct"] = None

    better_rate = result["refined_better_flags"]
    result["refined_better_rate"] = better_rate
    result["refined_tie_rate"] = result["refined_tie_flags"]
    result["refined_worse_rate"] = result["refined_worse_flags"]
    add_distribution_metrics(result, gain_abs_values, "gain_abs")
    add_distribution_metrics(result, gain_pct_values, "gain_pct")

    hr_base_masked = result["hr_masked_l1_base"]
    hr_refined_masked = result["hr_masked_l1_refined"]
    if hr_base_masked is not None and hr_refined_masked is not None and hr_base_masked > 0:
        result["hr_refinement_gain_abs"] = float(hr_base_masked - hr_refined_masked)
        result["hr_refinement_gain_pct"] = float((hr_base_masked - hr_refined_masked) / hr_base_masked * 100.0)
    else:
        result["hr_refinement_gain_abs"] = None
        result["hr_refinement_gain_pct"] = None
    result["hr_refined_better_rate"] = result["hr_refined_better_flags"]
    result["hr_refined_tie_rate"] = result["hr_refined_tie_flags"]
    result["hr_refined_worse_rate"] = result["hr_refined_worse_flags"]
    add_distribution_metrics(result, hr_gain_abs_values, "hr_gain_abs")
    add_distribution_metrics(result, hr_gain_pct_values, "hr_gain_pct")

    warnings = []
    if result["refinement_gain_pct"] is not None and result["refinement_gain_pct"] < -2.0:
        warnings.append("refinement_worse_than_coarse")
    if better_rate is not None and better_rate < 0.45:
        warnings.append("refinement_rarely_beats_coarse")
    if result["hr_refinement_gain_pct"] is not None and result["hr_refinement_gain_pct"] < -2.0:
        warnings.append("hr_refinement_worse_than_bilinear_base")
    if (
        result["raw_valid_l1_coarse"] is not None and
        result["raw_valid_l1_refined"] is not None and
        result["raw_valid_l1_refined"] > result["raw_valid_l1_coarse"] * 1.1
    ):
        warnings.append("raw_refinement_hurts_valid_region")
    result["warnings"] = warnings
    result["runtime_refinement_scale"] = float(model.generator.refinement_runtime_scale)

    if refinement_eval_scale is not None:
        model.generator.set_refinement_runtime_scale(original_refinement_scale)
    model.train()
    set_parameter_trainability(model, freeze_coarse=freeze_coarse)
    return result


def run_eval_only(cfg, args):
    """Run validation only on a checkpoint without training."""
    if not args.resume:
        raise ValueError("--eval-only requires --resume CHECKPOINT")

    device = resolve_device(args.device)
    print_device_banner(device)

    torch.manual_seed(cfg["training"]["seed"])

    model_config = build_model_config(cfg)
    model = InpaintingModel(model_config).to(device)
    freeze_coarse = cfg["training"].get("freeze_coarse", False)
    joint_hr_pipeline, model_image_size = validate_joint_hr_pipeline(cfg, model)
    set_parameter_trainability(model, freeze_coarse=freeze_coarse)

    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    load_model_checkpoint(model, ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.resume}")

    val_dir = cfg["data"].get("val_dir")
    manifest_path = cfg["data"].get("manifest_path")
    eval_loader = get_dataloader(
        root_dir=cfg["data"]["root_dir"],
        image_size=cfg["data"]["image_size"],
        split="val",
        batch_size=cfg["data"].get("eval_batch_size", cfg["data"]["batch_size"]),
        num_workers=max(1, min(2, cfg["data"]["num_workers"])),
        persistent_workers=cfg["data"].get("persistent_workers"),
        prefetch_factor=cfg["data"].get("prefetch_factor"),
        max_images=None,
        mask_min_coverage=cfg["data"]["mask_min_coverage"],
        mask_max_coverage=cfg["data"]["mask_max_coverage"],
        val_dir=val_dir,
        manifest_path=manifest_path,
        deterministic=args.force_random_masks,
        fixed_mask_seed=cfg["training"]["seed"],
        force_random_masks=args.force_random_masks,
    )
    print(f"Validation images: {len(eval_loader.dataset)}")

    if args.eval_batches is not None:
        max_batches = args.eval_batches if args.eval_batches > 0 else None
    else:
        max_batches = cfg.get("logging", {}).get("eval_batches", 8)
        if max_batches is not None and max_batches <= 0:
            max_batches = None
    eval_refinement_scale = cfg.get("logging", {}).get("eval_refinement_scale", 1.0)

    health = evaluate_refinement_health(
        model,
        eval_loader,
        device,
        cfg["training"]["mixed_precision"],
        max_batches=max_batches,
        freeze_coarse=freeze_coarse,
        model_image_size=model_image_size,
        joint_hr_pipeline=joint_hr_pipeline,
        refinement_eval_scale=eval_refinement_scale,
    )
    clear_device_cache(device)

    print("\nFull validation results:")
    print(json.dumps(health, indent=2))

    log_dir = Path(cfg["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_path = log_dir / "eval_only_full_val.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2)
    print(f"Saved eval summary: {eval_path}")


def log_validation(
    writer,
    log_dir,
    step,
    total_steps,
    loss_dict,
    running_loss,
    running_normalized_loss,
    lr,
    start_time,
    health,
):
    """Persist validation metrics and print a compact summary."""
    writer.add_scalar("val/masked_l1_coarse", health["masked_l1_coarse"], step)
    writer.add_scalar("val/masked_l1_refined", health["masked_l1_refined"], step)
    writer.add_scalar("val/refinement_gain_pct", health["refinement_gain_pct"], step)
    writer.add_scalar("val/refined_better_rate", health["refined_better_rate"], step)
    if health.get("gain_pct_p50") is not None:
        writer.add_scalar("val/gain_pct_p50", health["gain_pct_p50"], step)
    if health.get("hr_masked_l1_base") is not None:
        writer.add_scalar("val/hr_masked_l1_base", health["hr_masked_l1_base"], step)
    if health.get("hr_masked_l1_refined") is not None:
        writer.add_scalar("val/hr_masked_l1_refined", health["hr_masked_l1_refined"], step)
    if health.get("hr_refinement_gain_pct") is not None:
        writer.add_scalar("val/hr_refinement_gain_pct", health["hr_refinement_gain_pct"], step)
    if health.get("hr_refined_better_rate") is not None:
        writer.add_scalar("val/hr_refined_better_rate", health["hr_refined_better_rate"], step)
    if health.get("hr_gain_pct_p50") is not None:
        writer.add_scalar("val/hr_gain_pct_p50", health["hr_gain_pct_p50"], step)
    write_status(
        log_dir,
        step,
        total_steps,
        loss_dict,
        running_loss,
        running_normalized_loss,
        lr,
        start_time,
        extra={"validation": health},
    )
    append_validation_history(log_dir, step, health)

    hr_suffix = ""
    if health.get("hr_refinement_gain_pct") is not None:
        hr_suffix = (
            f", HR {health['hr_masked_l1_base']:.4f}->{health['hr_masked_l1_refined']:.4f}"
            f" (gain {health['hr_refinement_gain_pct']:.2f}%, better {health['hr_refined_better_rate']:.2f}"
        )
        if health.get("hr_gain_pct_p50") is not None:
            hr_suffix += f", p50 {health['hr_gain_pct_p50']:.2f}%"
        hr_suffix += ")"
    valid_suffix = ""
    if "raw_refinement_hurts_valid_region" in health["warnings"]:
        valid_suffix = f", raw_valid_refined={health['raw_valid_l1_refined']:.4f}"
    warning_suffix = f" warnings={','.join(health['warnings'])}" if health["warnings"] else ""
    print(
        f"\nValidation step {step}: "
        f"LR {health['masked_l1_coarse']:.4f}->{health['masked_l1_refined']:.4f} "
        f"(gain {health['refinement_gain_pct']:.2f}%, better {health['refined_better_rate']:.2f}, "
        f"p50 {health['gain_pct_p50']:.2f}%){hr_suffix}{valid_suffix}{warning_suffix}"
    )


def train(cfg, args):
    device = resolve_device(args.device)
    print_device_banner(device)

    # Seed
    torch.manual_seed(cfg["training"]["seed"])

    # Model
    model_config = build_model_config(cfg)
    model = InpaintingModel(model_config).to(device)
    freeze_coarse = cfg["training"].get("freeze_coarse", False)
    joint_hr_pipeline, model_image_size = validate_joint_hr_pipeline(cfg, model)
    attn_upscaler = AttentionUpscaling(model.generator) if joint_hr_pipeline else None
    set_parameter_trainability(model, freeze_coarse=freeze_coarse)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    if joint_hr_pipeline:
        print(f"Joint HR pipeline: {model_image_size}px -> {cfg['data']['image_size']}px")
    if freeze_coarse:
        frozen_params = sum(p.numel() for p in model.coarse_model.parameters())
        print(f"Frozen coarse parameters: {frozen_params:,}")

    # Loss
    criterion = InpaintingLoss(**cfg["loss"]).to(device)

    # Optimizer
    coarse_params = []
    refinement_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("coarse_model."):
            coarse_params.append(param)
        else:
            refinement_params.append(param)

    refinement_lr_scale = cfg["training"].get("refinement_lr_scale", 0.1)
    optimizer_groups = []
    if coarse_params:
        optimizer_groups.append({"params": coarse_params, "lr": cfg["training"]["lr"], "name": "coarse"})
    if refinement_params:
        optimizer_groups.append({
            "params": refinement_params,
            "lr": cfg["training"]["lr"] * refinement_lr_scale,
            "name": "refinement",
        })
    trainable_params = coarse_params + refinement_params
    optimizer = torch.optim.Adam(optimizer_groups)
    use_amp = is_amp_enabled(device, cfg["training"]["mixed_precision"])
    if cfg["training"]["mixed_precision"] and not use_amp:
        print("Mixed precision requested, but no supported accelerator is active; disabling AMP.")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # Data
    max_images = args.overfit if args.overfit else None
    deterministic_overfit = bool(args.overfit)
    val_dir = cfg["data"].get("val_dir")
    manifest_path = cfg["data"].get("manifest_path")
    train_loader = get_dataloader(
        root_dir=cfg["data"]["root_dir"],
        image_size=cfg["data"]["image_size"],
        split="train",
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        persistent_workers=cfg["data"].get("persistent_workers"),
        prefetch_factor=cfg["data"].get("prefetch_factor"),
        max_images=max_images,
        mask_min_coverage=cfg["data"]["mask_min_coverage"],
        mask_max_coverage=cfg["data"]["mask_max_coverage"],
        val_dir=val_dir,
        manifest_path=manifest_path,
        deterministic=deterministic_overfit,
        fixed_mask_seed=cfg["training"]["seed"],
        force_random_masks=args.force_random_masks,
        shuffle_override=(False if deterministic_overfit else None),
    )
    print(f"Training images: {len(train_loader.dataset)}")
    if args.force_random_masks:
        print(
            "Mask mode: generated freeform masks (manifest masks ignored). "
            f"Coverage target: {cfg['data']['mask_min_coverage']:.2f}-{cfg['data']['mask_max_coverage']:.2f}"
        )

    eval_loader = None
    eval_batches = cfg["logging"].get("eval_batches", 8)
    if log_cfg := cfg.get("logging"):
        pass
    if cfg["logging"].get("eval_interval", 0):
        eval_split = "train" if deterministic_overfit else "val"
        eval_loader = get_dataloader(
            root_dir=cfg["data"]["root_dir"],
            image_size=cfg["data"]["image_size"],
            split=eval_split,
            batch_size=cfg["data"].get("eval_batch_size", cfg["data"]["batch_size"]),
            num_workers=max(1, min(2, cfg["data"]["num_workers"])),
            persistent_workers=cfg["data"].get("persistent_workers"),
            prefetch_factor=cfg["data"].get("prefetch_factor"),
            max_images=max_images,
            mask_min_coverage=cfg["data"]["mask_min_coverage"],
            mask_max_coverage=cfg["data"]["mask_max_coverage"],
            val_dir=val_dir,
            manifest_path=manifest_path,
            deterministic=(deterministic_overfit or args.force_random_masks),
            fixed_mask_seed=cfg["training"]["seed"],
            force_random_masks=args.force_random_masks,
            shuffle_override=False,
        )
        print(f"Validation images: {len(eval_loader.dataset)}")
        if deterministic_overfit:
            print("Overfit eval uses the same deterministic training subset and masks.")

    # Training params
    total_steps = args.steps if args.steps else cfg["training"]["total_steps"]
    grad_accum = 1 if args.overfit else cfg["training"]["grad_accum_steps"]
    max_lr = cfg["training"]["lr"]
    min_lr = cfg["training"]["min_lr"]
    warmup_steps = cfg["training"]["warmup_steps"]
    grad_clip = cfg["training"]["grad_clip"]
    refinement_start_step = cfg["training"].get("refinement_start_step", 0)
    refinement_warmup_steps = cfg["training"].get("refinement_warmup_steps", 0)
    coarse_freeze_step = cfg["training"].get("coarse_freeze_step")
    auto_freeze_coarse = cfg["training"].get("auto_freeze_coarse", joint_hr_pipeline)
    if coarse_freeze_step is None and auto_freeze_coarse:
        coarse_freeze_step = refinement_start_step + max(refinement_warmup_steps, 1)
    # Logging
    log_cfg = cfg["logging"]
    eval_interval = log_cfg.get("eval_interval", 0)
    eval_refinement_scale = log_cfg.get("eval_refinement_scale", 1.0)
    if args.overfit and eval_interval:
        eval_interval = min(eval_interval, max(200, total_steps // 10))
    log_dir = Path(log_cfg["log_dir"])
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir / "tb")
    if eval_interval:
        print(f"Validation interval: every {eval_interval} steps")

    # Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        load_model_checkpoint(model, ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    set_parameter_trainability(model, freeze_coarse=freeze_coarse)
    coarse_frozen_now = freeze_coarse
    data_iter = iter(train_loader)
    optimizer.zero_grad()

    pbar = tqdm(range(start_step, total_steps), desc="Training", dynamic_ncols=True)
    running_loss = 0.0
    running_normalized_loss = 0.0
    train_start_time = time.time()
    loss_dict = {
        "l1_coarse": 0.0,
        "l1_refined": 0.0,
        "perceptual": 0.0,
        "style": 0.0,
        "hr_l1_refined": 0.0,
        "hr_delta": 0.0,
        "hr_perceptual": 0.0,
        "hr_style": 0.0,
        "gain_ratio": 0.0,
        "hr_gain_ratio": 0.0,
        "coarse_term": 0.0,
        "refinement_term": 0.0,
        "total": 0.0,
        "total_normalized": 0.0,
    }
    coarse = None
    refined = None
    batch_views = None

    for step_idx in pbar:
        step = step_idx + 1
        lr = get_lr(step, warmup_steps, total_steps, max_lr, min_lr)
        refinement_scale = get_refinement_scale(step, refinement_start_step, refinement_warmup_steps)
        model.generator.set_refinement_runtime_scale(refinement_scale)
        target_coarse_frozen = should_freeze_coarse(step, freeze_coarse, coarse_freeze_step)
        if target_coarse_frozen != coarse_frozen_now:
            set_parameter_trainability(model, freeze_coarse=target_coarse_frozen)
            coarse_frozen_now = target_coarse_frozen
        refinement_lr = lr * refinement_lr_scale if refinement_scale > 0 else 0.0
        for pg in optimizer.param_groups:
            if pg.get("name") == "refinement":
                pg["lr"] = refinement_lr
            else:
                pg["lr"] = 0.0 if coarse_frozen_now else lr

        loss_sums = {k: 0.0 for k in loss_dict}
        step_has_nonfinite = False

        for _ in range(grad_accum):
            # Get batch (cycle through dataset)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch_views = prepare_multiscale_batch(
                batch,
                device,
                model_image_size,
                blur_layer=model.generator.final_gaussian_blur,
            )
            image = batch_views["image"]
            mask = batch_views["mask"]
            masked_image = batch_views["masked_image"]

            # Forward
            with torch.amp.autocast(device.type, enabled=use_amp):
                refined_raw, attn, coarse_raw = model(masked_image, mask)
                coarse = composite_with_known(coarse_raw.clamp(0, 1), image, mask)
                refined = composite_with_known(refined_raw.clamp(0, 1), image, mask)
                hr_refined_raw = None
                hr_refined = None
                hr_base_raw = None
                hr_base = None
                if joint_hr_pipeline and batch_views["has_hr_supervision"]:
                    hr_refined_raw, hr_refined, hr_base_raw, hr_base = compute_hr_refined(
                        attn_upscaler,
                        batch_views,
                        refined,
                        attn,
                    )
                micro_loss, micro_loss_dict = compute_train_loss(
                    criterion,
                    coarse_raw,
                    refined_raw,
                    image,
                    mask,
                    refinement_loss_scale=refinement_scale,
                    coarse_composite=coarse,
                    refined_composite=refined,
                    hr_refined_raw=hr_refined_raw,
                    hr_target=batch_views["image_hr"] if hr_refined_raw is not None else None,
                    hr_mask=batch_views["mask_hr"] if hr_refined_raw is not None else None,
                    hr_base_raw=hr_base_raw,
                    hr_base_composite=hr_base,
                    hr_refined_composite=hr_refined,
                )
                if not torch.isfinite(micro_loss):
                    step_has_nonfinite = True
                    break
                loss = micro_loss / grad_accum

            if step_has_nonfinite:
                break

            # Backward
            scaler.scale(loss).backward()
            for key, value in micro_loss_dict.items():
                loss_sums[key] += value

        if step_has_nonfinite:
            optimizer.zero_grad(set_to_none=True)
            print(f"\nSkipping non-finite step {step}")
            continue

        loss_dict = {k: v / grad_accum for k, v in loss_sums.items()}

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss = 0.9 * running_loss + 0.1 * loss_dict["total"]
        running_normalized_loss = 0.9 * running_normalized_loss + 0.1 * loss_dict["total_normalized"]
        pbar.set_postfix(
            nl=f"{running_normalized_loss:.3f}",
            loss=f"{running_loss:.3f}",
        )

        # Logging
        if step == 1 or step % log_cfg["log_interval"] == 0:
            writer.add_scalar("loss/total", loss_dict["total"], step)
            writer.add_scalar("loss/l1_coarse", loss_dict["l1_coarse"], step)
            writer.add_scalar("loss/l1_refined", loss_dict["l1_refined"], step)
            writer.add_scalar("loss/perceptual", loss_dict["perceptual"], step)
            writer.add_scalar("loss/style", loss_dict["style"], step)
            writer.add_scalar("loss/hr_l1_refined", loss_dict["hr_l1_refined"], step)
            writer.add_scalar("loss/hr_delta", loss_dict["hr_delta"], step)
            writer.add_scalar("loss/hr_perceptual", loss_dict["hr_perceptual"], step)
            writer.add_scalar("loss/hr_style", loss_dict["hr_style"], step)
            writer.add_scalar("loss/gain_ratio", loss_dict["gain_ratio"], step)
            writer.add_scalar("loss/hr_gain_ratio", loss_dict["hr_gain_ratio"], step)
            writer.add_scalar("loss/coarse_term", loss_dict["coarse_term"], step)
            writer.add_scalar("loss/refinement_term", loss_dict["refinement_term"], step)
            writer.add_scalar("loss/total_normalized", loss_dict["total_normalized"], step)
            writer.add_scalar("loss/running_total", running_loss, step)
            writer.add_scalar("loss/running_total_normalized", running_normalized_loss, step)
            writer.add_scalar("lr", lr, step)
            writer.add_scalar("refinement_scale", refinement_scale, step)
            writer.add_scalar("refinement_lr", refinement_lr, step)
            writer.add_scalar("coarse_frozen", 1.0 if coarse_frozen_now else 0.0, step)

            peak_memory_gb = get_peak_memory_allocated_gb(device)
            if peak_memory_gb is not None:
                writer.add_scalar("accelerator_mem_gb", peak_memory_gb, step)
                if device.type == "cuda":
                    writer.add_scalar("vram_gb", peak_memory_gb, step)
                elif device.type == "xpu":
                    writer.add_scalar("xpu_mem_gb", peak_memory_gb, step)

            # Write status file for remote monitoring
            write_status(
                log_cfg["log_dir"],
                step,
                total_steps,
                loss_dict,
                running_loss,
                running_normalized_loss,
                lr,
                train_start_time,
            )
            print(
                f"\nTrain step {step}: "
                f"loss={running_loss:.4f}, nl={running_normalized_loss:.4f}, "
                f"l1r={loss_dict['l1_refined']:.4f}, scale={refinement_scale:.3f}"
            )

        if eval_loader is not None and step % eval_interval == 0:
            clear_device_cache(device)
            health = evaluate_refinement_health(
                model,
                eval_loader,
                device,
                use_amp,
                max_batches=eval_batches,
                freeze_coarse=freeze_coarse,
                model_image_size=model_image_size,
                joint_hr_pipeline=joint_hr_pipeline,
                refinement_eval_scale=eval_refinement_scale,
            )
            log_validation(
                writer,
                log_cfg["log_dir"],
                step,
                total_steps,
                loss_dict,
                running_loss,
                running_normalized_loss,
                lr,
                train_start_time,
                health,
            )
            clear_device_cache(device)

        # Visualization
        if step % log_cfg["vis_interval"] == 0:
            save_vis(writer, batch_views, coarse, refined, step, log_dir=log_cfg["log_dir"])

        # Checkpoint at specific steps
        checkpoint_steps = log_cfg.get("checkpoint_steps", None)
        save_checkpoints = log_cfg.get("save_checkpoints", True)
        should_save = False
        if save_checkpoints and checkpoint_steps and step in checkpoint_steps:
            should_save = True
        elif save_checkpoints and not checkpoint_steps and step % log_cfg["save_interval"] == 0:
            should_save = True

        if should_save:
            ckpt_path = ckpt_dir / f"step_{step}.pth"
            if save_checkpoint(model, optimizer, scaler, step, loss_dict, cfg, ckpt_path):
                print(f"\nSaved checkpoint: {ckpt_path}")
                write_status(
                    log_cfg["log_dir"],
                    step,
                    total_steps,
                    loss_dict,
                    running_loss,
                    running_normalized_loss,
                    lr,
                    train_start_time,
                    extra={"event": "checkpoint", "checkpoint_path": str(ckpt_path)},
                )

            # Clean old checkpoints (keep all if using checkpoint_steps)
            if not checkpoint_steps:
                keep = log_cfg.get("keep_last_checkpoints", 3)
                ckpts = sorted(ckpt_dir.glob("step_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
                for old in ckpts[:-keep]:
                    old.unlink()

    # Final checkpoint
    final_path = ckpt_dir / f"step_{total_steps}.pth"
    save_final_checkpoint = log_cfg.get("save_final_checkpoint", True)
    if save_final_checkpoint:
        save_checkpoint(model, optimizer, scaler, total_steps, loss_dict, cfg, final_path)
    final_lr = get_lr(total_steps, warmup_steps, total_steps, max_lr, min_lr)
    if eval_loader is not None:
        clear_device_cache(device)
        final_health = evaluate_refinement_health(
            model,
            eval_loader,
            device,
            use_amp,
            max_batches=eval_batches,
            freeze_coarse=freeze_coarse,
            model_image_size=model_image_size,
            joint_hr_pipeline=joint_hr_pipeline,
            refinement_eval_scale=eval_refinement_scale,
        )
        log_validation(
            writer,
            log_cfg["log_dir"],
            total_steps,
            total_steps,
            loss_dict,
            running_loss,
            running_normalized_loss,
            final_lr,
            train_start_time,
            final_health,
        )
        clear_device_cache(device)
    if save_final_checkpoint:
        print(f"\nTraining complete. Final checkpoint: {final_path}")
    else:
        print("\nTraining complete. Final checkpoint saving disabled for this run.")
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="RETHINED Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. xpu, cuda, cpu, or xpu:0")
    parser.add_argument("--overfit", type=int, default=None, help="Overfit on N images (sanity check)")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--eval-only", action="store_true", help="Run validation only on a checkpoint")
    parser.add_argument("--eval-batches", type=int, default=None, help="Override eval batches; use 0 for full validation set")
    parser.add_argument(
        "--force-random-masks",
        action="store_true",
        help="Ignore manifest masks and generate new freeform masks from the configured coverage range.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.eval_only:
        run_eval_only(cfg, args)
    else:
        train(cfg, args)


if __name__ == "__main__":
    main()
