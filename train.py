from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from data.dataset import get_dataloader
from device_utils import (
    empty_device_cache,
    get_autocast_device_type,
    get_device_name,
    get_peak_memory_allocated_gb,
    is_amp_enabled,
)
from distributed_utils import barrier, destroy_distributed, init_distributed, reduce_metrics, reduce_scalar, unwrap_model
from upscale import AttentionUpscaling
from losses import InpaintingLoss
from model import InpaintingModel


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
        k: v
        for k, v in cfg["model"]["inpainter"].items()
        if k != "copy_mode"
    }
    return {
        "coarse_model": {
            "class": coarse_cfg.get("class", "CoarseModel"),
            "parameters": {k: v for k, v in coarse_cfg.items() if k != "class"},
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


def save_checkpoint(
    model,
    optimizer_g,
    scaler,
    step,
    metrics,
    cfg,
    path,
):
    raw_model = unwrap_model(model)
    torch.save(
        {
            "step": step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def load_model_checkpoint(model, state_dict):
    raw_model = unwrap_model(model)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    missing_keys, unexpected_keys = raw_model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible with the current model. "
            f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
        )


def load_eval_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError("Expected a training checkpoint containing 'model_state_dict'.")
    load_model_checkpoint(model, ckpt["model_state_dict"])
    return int(ckpt.get("step", 0))


def load_training_checkpoint(
    path,
    model,
    optimizer_g,
    scaler,
    device,
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    required_keys = (
        "step",
        "model_state_dict",
        "optimizer_g_state_dict",
        "scaler_state_dict",
    )
    missing_keys = [key for key in required_keys if key not in ckpt]
    if missing_keys:
        raise KeyError(f"Expected a full training checkpoint, missing keys: {', '.join(missing_keys)}")
    load_model_checkpoint(model, ckpt["model_state_dict"])
    optimizer_g.load_state_dict(ckpt["optimizer_g_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    metrics = ckpt.get("metrics", {}) or {}
    return {
        "step": int(ckpt["step"]),
        "best_metric_name": metrics.get("best_metric_name"),
        "best_metric_mode": metrics.get("best_metric_mode"),
        "best_metric_value": metrics.get("best_metric_value"),
        "best_metric_step": metrics.get("best_metric_step"),
    }


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


def build_checkpoint_metrics(train_metrics, best_metric_name, best_metric_mode, best_metric_value, best_metric_step):
    checkpoint_metrics = dict(train_metrics)
    checkpoint_metrics["best_metric_name"] = best_metric_name
    checkpoint_metrics["best_metric_mode"] = best_metric_mode
    checkpoint_metrics["best_metric_value"] = best_metric_value
    checkpoint_metrics["best_metric_step"] = best_metric_step
    if best_metric_name and best_metric_value is not None:
        checkpoint_metrics[f"val/{best_metric_name}"] = best_metric_value
    return checkpoint_metrics


def format_train_metric_snapshot(metrics, include_retrieval=True, retrieval_margin_label="3pct"):
    summary = (
        f"train i={metrics['inpainter_total']:.4f}, "
        f"l1={metrics['refined_l1']:.4f}, "
        f"perc={metrics['perceptual']:.4f}"
    )
    if "refined_query_patch_l1" in metrics:
        summary += f", qp={metrics['refined_query_patch_l1']:.4f}"
    if include_retrieval and "retrieval_recall1_exact" in metrics:
        summary += f", r1_exact={metrics['retrieval_recall1_exact']:.3f}"
    if include_retrieval and "retrieval_coherence_loss" in metrics:
        summary += f", coh={metrics['retrieval_coherence_loss']:.3f}"
    if include_retrieval and "retrieval_top1_margin_loss" in metrics:
        summary += f", mrg={metrics['retrieval_top1_margin_loss']:.3f}"
    if include_retrieval and "retrieval_recall1" in metrics:
        summary += f", r1_{retrieval_margin_label}={metrics['retrieval_recall1']:.3f}"
    if include_retrieval and "retrieval_recall8" in metrics:
        summary += f", r8_{retrieval_margin_label}={metrics['retrieval_recall8']:.3f}"
    if include_retrieval and "retrieval_recall32" in metrics:
        summary += f", r32_{retrieval_margin_label}={metrics['retrieval_recall32']:.3f}"
    if "transport_selection_recall1_exact" in metrics:
        summary += f", tr1_exact={metrics['transport_selection_recall1_exact']:.3f}"
    if "transport_selection_recall1" in metrics:
        summary += f", tr1_{retrieval_margin_label}={metrics['transport_selection_recall1']:.3f}"
    if "transport_selection_recall8" in metrics:
        summary += f", tr8_{retrieval_margin_label}={metrics['transport_selection_recall8']:.3f}"
    if "transport_selection_recall32" in metrics:
        summary += f", tr32_{retrieval_margin_label}={metrics['transport_selection_recall32']:.3f}"
    if "transport_patch" in metrics:
        summary += f", tp={metrics['transport_patch']:.4f}"
    if "transport_validity" in metrics:
        summary += f", tval={metrics['transport_validity']:.4f}"
    if "transport_valid_ratio" in metrics:
        summary += f", tvr={metrics['transport_valid_ratio']:.3f}"
    if "transport_fallback_ratio" in metrics:
        summary += f", tfr={metrics['transport_fallback_ratio']:.3f}"
    if "transport_offset_smoothness" in metrics:
        summary += f", tsm={metrics['transport_offset_smoothness']:.4f}"
    if "transport_offset_curvature" in metrics:
        summary += f", tcr={metrics['transport_offset_curvature']:.4f}"
    return summary


def format_val_selection_snapshot(metrics, retrieval_margin_label="3pct"):
    parts = []
    if "retrieval_recall1_exact" in metrics:
        parts.append(f"r1_exact={metrics['retrieval_recall1_exact']:.3f}")
    if "retrieval_recall1" in metrics:
        parts.append(f"r1_{retrieval_margin_label}={metrics['retrieval_recall1']:.3f}")
    if "retrieval_recall8" in metrics:
        parts.append(f"r8_{retrieval_margin_label}={metrics['retrieval_recall8']:.3f}")
    if "retrieval_recall32" in metrics:
        parts.append(f"r32_{retrieval_margin_label}={metrics['retrieval_recall32']:.3f}")
    if "transport_selection_recall1_exact" in metrics:
        parts.append(f"tr1_exact={metrics['transport_selection_recall1_exact']:.3f}")
    if "transport_selection_recall1" in metrics:
        parts.append(f"tr1_{retrieval_margin_label}={metrics['transport_selection_recall1']:.3f}")
    if "transport_selection_recall8" in metrics:
        parts.append(f"tr8_{retrieval_margin_label}={metrics['transport_selection_recall8']:.3f}")
    if "transport_selection_recall32" in metrics:
        parts.append(f"tr32_{retrieval_margin_label}={metrics['transport_selection_recall32']:.3f}")
    return ", ".join(parts)


def save_vis(writer, batch, coarse, refined, step, log_dir=None):
    n = min(4, batch["image"].shape[0])
    gt = batch["refine_target"][:n].cpu()
    masked = batch["masked_image"][:n].cpu()
    mask_vis = batch["mask"][:n].repeat(1, 3, 1, 1).cpu()
    coarse_vis = coarse[:n].detach().clamp(0, 1).cpu()
    refined_vis = refined[:n].detach().clamp(0, 1).cpu()
    grid = make_grid(torch.cat([masked, mask_vis, coarse_vis, refined_vis, gt], dim=0), nrow=n, padding=2)
    writer.add_image("samples/masked|mask|coarse|refined|gt", grid, step)
    if log_dir is not None:
        vis_dir = Path(log_dir) / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_image(grid, vis_dir / f"step_{step:06d}.png")


@torch.no_grad()
def render_visualization_batch(model, batch_views, device, use_amp):
    was_training = model.training
    amp_device_type = get_autocast_device_type(device)
    amp_enabled = is_amp_enabled(device, use_amp)
    try:
        model.eval()
        with torch.amp.autocast(amp_device_type, enabled=amp_enabled):
            refined_raw, _, coarse_raw = model(
                batch_views["masked_image"],
                batch_views["mask"],
                value_image=batch_views["refine_target"],
                return_aux=False,
            )
        refined = refined_raw.clamp(0, 1).detach()
        coarse = composite_with_known(
            coarse_raw.clamp(0, 1),
            batch_views["refine_target"],
            batch_views["mask"],
        ).detach()
        return coarse, refined
    finally:
        model.train(was_training)


@torch.no_grad()
def validate_model(model, dataloader, device, use_amp, model_image_size, dist_ctx, criterion=None, max_batches=8):
    model.eval()
    raw_model = unwrap_model(model)
    attn_upscaler = AttentionUpscaling(raw_model.inpainter)
    amp_device_type = get_autocast_device_type(device)
    amp_enabled = is_amp_enabled(device, use_amp)

    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
            break

        batch_views = prepare_multiscale_batch(
            batch,
            device,
            model_image_size,
            blur_layer=raw_model.inpainter.final_gaussian_blur,
        )
        mask = batch_views["mask"]
        masked_image = batch_views["masked_image"]
        refine_target = batch_views["refine_target"]

        with torch.amp.autocast(amp_device_type, enabled=amp_enabled):
            if criterion is not None:
                refined_lr, attn_map, coarse_raw, attention_aux = model(
                    masked_image,
                    mask,
                    value_image=refine_target,
                    return_aux=True,
                )
            else:
                refined_lr, attn_map, coarse_raw = model(
                    masked_image,
                    mask,
                    value_image=refine_target,
                )
                attention_aux = None

        refined_lr = refined_lr.clamp(0, 1)
        coarse_lr = composite_with_known(coarse_raw.clamp(0, 1), refine_target, mask)
        lr_coarse_err = masked_l1(coarse_lr, refine_target, mask)
        lr_refined_err = masked_l1(refined_lr, refine_target, mask)
        lr_gain = lr_coarse_err - lr_refined_err
        metric_sums["masked_l1_lr_coarse"] += float(lr_coarse_err.sum().item())
        metric_counts["masked_l1_lr_coarse"] += int(lr_coarse_err.numel())
        metric_sums["masked_l1_lr_refined"] += float(lr_refined_err.sum().item())
        metric_counts["masked_l1_lr_refined"] += int(lr_refined_err.numel())
        metric_sums["lr_gain_abs"] += float(lr_gain.sum().item())
        metric_counts["lr_gain_abs"] += int(lr_gain.numel())

        if batch_views["has_hr_target"]:
            target = batch_views["image_hr"]
            eval_mask = batch_views["mask_hr"]
            coarse_eval = composite_with_known(
                F.interpolate(coarse_lr, size=target.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1),
                target,
                eval_mask,
            )
            with torch.amp.autocast(amp_device_type, enabled=amp_enabled):
                refined_hr = attn_upscaler(
                    batch_views["masked_image_hr"],
                    refined_lr,
                    attn_map,
                    mask_hr=batch_views["mask_hr"],
                ).clamp(0, 1)
            refined_eval = composite_with_known(refined_hr, target, eval_mask)
            hr_coarse_err = masked_l1(coarse_eval, target, eval_mask)
            hr_refined_err = masked_l1(refined_eval, target, eval_mask)
            hr_gain = hr_coarse_err - hr_refined_err
        else:
            hr_coarse_err = lr_coarse_err
            hr_refined_err = lr_refined_err
            hr_gain = lr_gain

        metric_sums["masked_l1_hr_coarse_baseline"] += float(hr_coarse_err.sum().item())
        metric_counts["masked_l1_hr_coarse_baseline"] += int(hr_coarse_err.numel())
        metric_sums["masked_l1_hr_refined"] += float(hr_refined_err.sum().item())
        metric_counts["masked_l1_hr_refined"] += int(hr_refined_err.numel())
        metric_sums["hr_gain_abs"] += float(hr_gain.sum().item())
        metric_counts["hr_gain_abs"] += int(hr_gain.numel())
        if criterion is not None:
            retrieval_metrics = criterion.attention_supervision_metrics(refine_target, attention_aux)
            for key, value in retrieval_metrics.items():
                metric_sums[key] += float(value)
                metric_counts[key] += 1
            transport_selection_metrics = criterion.transport_selection_metrics(refine_target, attention_aux)
            for key, value in transport_selection_metrics.items():
                metric_sums[key] += float(value)
                metric_counts[key] += 1

    model.train()
    if dist_ctx.enabled:
        reduced_payload = {}
        for key in sorted(metric_sums):
            reduced_payload[f"{key}/sum"] = metric_sums[key]
            reduced_payload[f"{key}/count"] = float(metric_counts[key])
        reduced_payload = reduce_metrics(reduced_payload, dist_ctx, average=False)
        metric_sums = {
            key[:-4]: value
            for key, value in reduced_payload.items()
            if key.endswith("/sum")
        }
        metric_counts = {
            key[:-6]: int(round(value))
            for key, value in reduced_payload.items()
            if key.endswith("/count")
        }

    def mean_from(name):
        count = metric_counts.get(name, 0)
        if count <= 0:
            return None
        return metric_sums[name] / count

    lr_coarse_mean = mean_from("masked_l1_lr_coarse")
    lr_refined_mean = mean_from("masked_l1_lr_refined")
    lr_gain_mean = mean_from("lr_gain_abs")
    hr_coarse_mean = mean_from("masked_l1_hr_coarse_baseline")
    hr_refined_mean = mean_from("masked_l1_hr_refined")
    hr_gain_mean = mean_from("hr_gain_abs")
    result = {
        "masked_l1_lr_coarse": lr_coarse_mean,
        "masked_l1_lr_refined": lr_refined_mean,
        "lr_gain_abs": lr_gain_mean,
        "lr_gain_pct": (
            100.0 * (lr_gain_mean / max(lr_coarse_mean, 1e-8))
            if lr_coarse_mean is not None and lr_gain_mean is not None
            else None
        ),
        "masked_l1_hr_coarse_baseline": hr_coarse_mean,
        "masked_l1_hr_refined": hr_refined_mean,
        "hr_gain_abs": hr_gain_mean,
        "hr_gain_pct": (
            100.0 * (hr_gain_mean / max(hr_coarse_mean, 1e-8))
            if hr_coarse_mean is not None and hr_gain_mean is not None
            else None
        ),
    }
    for key in (
        "retrieval_recall1_exact",
        "retrieval_recall1",
        "retrieval_recall8",
        "retrieval_recall32",
        "transport_selection_recall1_exact",
        "transport_selection_recall1",
        "transport_selection_recall8",
        "transport_selection_recall32",
    ):
        value = mean_from(key)
        if value is not None:
            result[key] = value
    return result


def write_status(log_dir, step, total_steps, metrics, lr):
    status = {
        "step": step,
        "total_steps": total_steps,
        "progress_pct": round(100.0 * step / max(total_steps, 1), 2),
        "lr": lr,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    status_path = Path(log_dir) / "training_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)


def write_validation_history(log_dir, step, metrics):
    history_path = Path(log_dir) / "validation_history.json"
    entry = {
        "step": step,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    history = []
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as handle:
            history = json.load(handle)
        if not isinstance(history, list):
            raise ValueError(f"Expected validation history list in {history_path}")

    replaced = False
    for idx, existing in enumerate(history):
        if isinstance(existing, dict) and existing.get("step") == step:
            history[idx] = entry
            replaced = True
            break
    if not replaced:
        history.append(entry)

    history.sort(key=lambda item: item.get("step", 0))
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def build_train_loader(cfg, args, dist_ctx):
    max_images = args.overfit if args.overfit else None
    num_workers = 0 if args.overfit else cfg["data"]["num_workers"]
    return get_dataloader(
        root_dir=cfg["data"]["root_dir"],
        image_size=cfg["data"]["image_size"],
        split="train",
        batch_size=cfg["data"]["batch_size"],
        num_workers=num_workers,
        persistent_workers=cfg["data"].get("persistent_workers"),
        prefetch_factor=cfg["data"].get("prefetch_factor"),
        max_images=max_images,
        mask_min_coverage=cfg["data"]["mask_min_coverage"],
        mask_max_coverage=cfg["data"]["mask_max_coverage"],
        val_dir=cfg["data"].get("val_dir"),
        manifest_path=cfg["data"].get("manifest_path"),
        deterministic=bool(args.overfit),
        fixed_mask_seed=cfg["training"]["seed"],
        force_random_masks=(cfg["data"].get("force_random_masks_train", False) or args.force_random_masks),
        mask_generator_kwargs=cfg["data"].get("mask_generator"),
        shuffle_override=(False if args.overfit else None),
        distributed=dist_ctx.enabled,
        rank=dist_ctx.rank,
        world_size=dist_ctx.world_size,
        sampler_seed=cfg["training"]["seed"],
    )


def build_eval_loader(cfg, args, dist_ctx):
    eval_interval = cfg.get("logging", {}).get("eval_interval", 0)
    if eval_interval <= 0 and not args.eval_only:
        return None
    max_images = args.overfit if args.overfit else None
    split = "train" if args.overfit else "val"
    num_workers = 0 if args.overfit else max(1, min(2, cfg["data"]["num_workers"]))
    return get_dataloader(
        root_dir=cfg["data"]["root_dir"],
        image_size=cfg["data"]["image_size"],
        split=split,
        batch_size=cfg["data"].get("eval_batch_size", cfg["data"]["batch_size"]),
        num_workers=num_workers,
        persistent_workers=cfg["data"].get("persistent_workers"),
        prefetch_factor=cfg["data"].get("prefetch_factor"),
        max_images=max_images,
        mask_min_coverage=cfg["data"]["mask_min_coverage"],
        mask_max_coverage=cfg["data"]["mask_max_coverage"],
        val_dir=cfg["data"].get("val_dir"),
        manifest_path=cfg["data"].get("manifest_path"),
        deterministic=bool(args.overfit or cfg["data"].get("force_random_masks_eval", False) or args.force_random_masks),
        fixed_mask_seed=cfg["training"]["seed"],
        force_random_masks=(cfg["data"].get("force_random_masks_eval", False) or args.force_random_masks),
        mask_generator_kwargs=cfg["data"].get("mask_generator"),
        shuffle_override=False,
        distributed=dist_ctx.enabled,
        rank=dist_ctx.rank,
        world_size=dist_ctx.world_size,
        sampler_seed=cfg["training"]["seed"],
    )


def run_eval_only(cfg, args, dist_ctx):
    device = dist_ctx.device
    if dist_ctx.is_main_process:
        print_device_banner(device)
    seed_everything(cfg["training"]["seed"])

    model = InpaintingModel(build_model_config(cfg)).to(device)
    if not args.resume:
        raise ValueError("--eval-only requires --resume CHECKPOINT")
    checkpoint_step = load_eval_checkpoint(args.resume, model, device)
    eval_loader = build_eval_loader(cfg, args, dist_ctx)
    if eval_loader is None:
        raise ValueError("No evaluation loader configured.")

    eval_sampler = getattr(eval_loader, "sampler", None)
    if isinstance(eval_sampler, DistributedSampler):
        eval_sampler.set_epoch(0)

    loss_cfg = dict(cfg["loss"])
    criterion = InpaintingLoss(**loss_cfg).to(device)
    health = validate_model(
        model,
        eval_loader,
        device,
        cfg["training"]["mixed_precision"],
        unwrap_model(model).inpainter.image_size,
        dist_ctx,
        criterion=criterion,
        max_batches=(args.eval_batches if args.eval_batches is not None else cfg.get("logging", {}).get("eval_batches", 8)),
    )
    if dist_ctx.is_main_process:
        print(json.dumps(health, indent=2))

        log_dir = Path(cfg["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "eval_only_full_val.json", "w", encoding="utf-8") as handle:
            json.dump(health, handle, indent=2)
    barrier(dist_ctx)


def train(cfg, args, dist_ctx):
    device = dist_ctx.device
    if dist_ctx.is_main_process:
        print_device_banner(device)
        if dist_ctx.enabled:
            print(
                f"Distributed training enabled: world_size={dist_ctx.world_size}, "
                f"backend={dist_ctx.backend}"
            )
    seed_everything(cfg["training"]["seed"])

    raw_model = InpaintingModel(build_model_config(cfg)).to(device)
    loss_cfg = dict(cfg["loss"])
    criterion = InpaintingLoss(**loss_cfg).to(device)

    scorer_param_names = {
        "query_descriptor_head", "key_descriptor_head",
        "matching_descriptor_head", "shared_query_key_descriptor_head",
        "query_context_encoder", "key_context_encoder",
        "query_context_descriptor_head", "query_context_scale",
        "key_coarse_rgb_scale", "key_feature_scale",
        "pre_attention_norm", "multihead_attention",
    }
    scorer_params = []
    base_params = []
    for name, param in raw_model.named_parameters():
        parts = name.split(".")
        # Check if any part of the parameter path matches a scorer module
        if any(part in scorer_param_names for part in parts):
            scorer_params.append(param)
        else:
            base_params.append(param)

    scorer_lr = cfg["training"].get("scorer_lr", cfg["training"]["lr"])
    scorer_min_lr = cfg["training"].get("scorer_min_lr", cfg["training"]["min_lr"])
    optimizer_g = torch.optim.Adam(
        [
            {"params": base_params, "lr": cfg["training"]["lr"]},
            {"params": scorer_params, "lr": scorer_lr, "_group_name": "scorer"},
        ],
        betas=tuple(cfg["training"].get("betas", [0.9, 0.999])),
    )

    use_amp = is_amp_enabled(device, cfg["training"]["mixed_precision"])
    if cfg["training"]["mixed_precision"] and not use_amp:
        if dist_ctx.is_main_process:
            print("Mixed precision requested, but no supported accelerator is active; disabling AMP.")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_step = 0
    best_metric_name = cfg["logging"].get("save_best_metric", "masked_l1_hr_refined")
    best_metric_mode = cfg["logging"].get("save_best_mode", "min")
    save_best_checkpoint = cfg["logging"].get("save_best_checkpoint", True)
    best_metric_value = None
    best_metric_step = None
    if args.resume:
        resume_state = load_training_checkpoint(
            args.resume,
            raw_model,
            optimizer_g,
            scaler,
            device,
        )
        start_step = resume_state["step"]
        if resume_state["best_metric_name"] == best_metric_name and resume_state["best_metric_mode"] == best_metric_mode:
            best_metric_value = resume_state["best_metric_value"]
            best_metric_step = resume_state["best_metric_step"]

    ddp_find_unused = cfg["training"].get("ddp_find_unused_parameters", False)
    if dist_ctx.enabled:
        ddp_kwargs = {"find_unused_parameters": ddp_find_unused}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device.index]
            ddp_kwargs["output_device"] = device.index
        model = DDP(raw_model, **ddp_kwargs)
    else:
        model = raw_model

    train_loader = build_train_loader(cfg, args, dist_ctx)
    eval_loader = build_eval_loader(cfg, args, dist_ctx)
    train_sampler = train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None
    eval_sampler = eval_loader.sampler if (eval_loader is not None and isinstance(eval_loader.sampler, DistributedSampler)) else None
    if train_sampler is not None:
        train_sampler.set_epoch(0)
    if eval_sampler is not None:
        eval_sampler.set_epoch(0)
    if dist_ctx.is_main_process:
        print(f"Training images: {len(train_loader.dataset)}")
        if eval_loader is not None:
            print(f"Validation images: {len(eval_loader.dataset)}")

    log_cfg = cfg["logging"]
    retrieval_margin_label = f"{int(round(100 * cfg['loss'].get('retrieval_target_margin_pct', 0.03)))}pct"
    total_steps = args.steps if args.steps else cfg["training"]["total_steps"]
    grad_accum = 1 if args.overfit else cfg["training"]["grad_accum_steps"]
    max_lr = cfg["training"]["lr"]
    min_lr = cfg["training"]["min_lr"]
    warmup_steps = cfg["training"]["warmup_steps"]
    grad_clip_g = cfg["training"].get("grad_clip", 1.0)
    model_image_size = raw_model.inpainter.image_size

    if dist_ctx.is_main_process:
        print(f"Inpainter parameters: {sum(p.numel() for p in raw_model.parameters()):,}")
        print(f"  Base parameters: {sum(p.numel() for p in base_params):,}")
        print(f"  Scorer parameters: {sum(p.numel() for p in scorer_params):,} (LR: {scorer_lr})")
        print(f"Per-rank effective batch size: {cfg['data']['batch_size'] * grad_accum}")
        print(f"Global effective batch size: {cfg['data']['batch_size'] * grad_accum * dist_ctx.world_size}")

    log_dir = Path(log_cfg["log_dir"])
    ckpt_dir = log_dir / "models"
    if dist_ctx.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = create_summary_writer(log_dir / "tb") if dist_ctx.is_main_process else NullSummaryWriter()
    if args.resume and dist_ctx.is_main_process:
        print(f"Resumed from step {start_step}")

    amp_device_type = get_autocast_device_type(device)
    train_epoch = 0
    data_iter = iter(train_loader)
    running_g = 0.0
    last_batch_views = None
    metrics = {}

    progress_bar = tqdm(
        range(start_step, total_steps),
        desc="Training",
        dynamic_ncols=True,
        disable=not dist_ctx.is_main_process,
    )
    for step_idx in progress_bar:
        step = step_idx + 1
        criterion.set_training_step(step)
        lr_g = get_lr(step, warmup_steps, total_steps, max_lr, min_lr)
        lr_scorer = get_lr(step, warmup_steps, total_steps, scorer_lr, scorer_min_lr)
        for pg in optimizer_g.param_groups:
            if pg.get("_group_name") == "scorer":
                pg["lr"] = lr_scorer
            else:
                pg["lr"] = lr_g

        optimizer_g.zero_grad(set_to_none=True)
        metric_sums = defaultdict(float)
        step_has_nonfinite = False

        for accum_idx in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                train_epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(train_epoch)
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch_views = prepare_multiscale_batch(
                batch,
                device,
                model_image_size,
                blur_layer=raw_model.inpainter.final_gaussian_blur,
            )
            image = batch_views["image"]
            mask = batch_views["mask"]
            masked_image = batch_views["masked_image"]
            refine_target = batch_views["refine_target"]

            sync_context = model.no_sync() if dist_ctx.enabled and accum_idx < (grad_accum - 1) else nullcontext()
            with sync_context:
                with torch.amp.autocast(amp_device_type, enabled=use_amp):
                    refined_raw, attn_map, coarse_raw, attention_aux = model(
                        masked_image,
                        mask,
                        value_image=refine_target,
                        return_aux=True,
                    )
                    g_loss, g_metrics = criterion.inpainter_loss(
                        coarse_raw,
                        refined_raw,
                        image,
                        refine_target,
                        mask,
                        attention_aux=attention_aux,
                    )

                has_nonfinite = not torch.isfinite(g_loss)
                if dist_ctx.enabled:
                    has_nonfinite = bool(reduce_scalar(float(has_nonfinite), dist_ctx, average=False))
                if has_nonfinite:
                    step_has_nonfinite = True
                    break

                scaler.scale(g_loss / grad_accum).backward()

            if step_has_nonfinite:
                step_has_nonfinite = True
                break

            attn_metrics = raw_model.inpainter.summarize_attention(
                attn_map.detach(),
                raw_model.inpainter.flatten_query_mask(mask).detach(),
            )

            for key, value in g_metrics.items():
                metric_sums[key] += metric_to_float(value)
            for key, value in attn_metrics.items():
                metric_sums[key] += metric_to_float(value)

            last_batch_views = batch_views

        if step_has_nonfinite:
            optimizer_g.zero_grad(set_to_none=True)
            if dist_ctx.is_main_process:
                progress_bar.write(f"Skipping non-finite step {step}")
            continue

        scaler.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip_g)
        scaler.step(optimizer_g)
        scaler.update()
        optimizer_g.zero_grad(set_to_none=True)

        metrics = {key: value / grad_accum for key, value in metric_sums.items()}
        metrics = reduce_metrics(metrics, dist_ctx, average=True)
        running_g = 0.9 * running_g + 0.1 * metrics["inpainter_total"]

        if dist_ctx.is_main_process and (step == 1 or step % log_cfg["log_interval"] == 0):
            writer.add_scalar("loss/coarse_l2", metrics["coarse_l2"], step)
            writer.add_scalar("loss/refined_l1", metrics["refined_l1"], step)
            if "refined_query_patch_l1" in metrics:
                writer.add_scalar("loss/refined_query_patch_l1", metrics["refined_query_patch_l1"], step)
            if "retrieval_loss" in metrics:
                writer.add_scalar("loss/retrieval", metrics["retrieval_loss"], step)
            if "retrieval_coherence_loss" in metrics:
                writer.add_scalar("loss/retrieval_coherence", metrics["retrieval_coherence_loss"], step)
            if "retrieval_top1_margin_loss" in metrics:
                writer.add_scalar("loss/retrieval_top1_margin", metrics["retrieval_top1_margin_loss"], step)
            if "transport_patch" in metrics:
                writer.add_scalar("loss/transport_patch", metrics["transport_patch"], step)
            if "transport_validity" in metrics:
                writer.add_scalar("loss/transport_validity", metrics["transport_validity"], step)
            if "transport_valid_ratio" in metrics:
                writer.add_scalar("transport/valid_ratio", metrics["transport_valid_ratio"], step)
            if "transport_fallback_ratio" in metrics:
                writer.add_scalar("transport/fallback_ratio", metrics["transport_fallback_ratio"], step)
            if "transport_offset_smoothness" in metrics:
                writer.add_scalar("loss/transport_offset_smoothness", metrics["transport_offset_smoothness"], step)
            if "transport_offset_curvature" in metrics:
                writer.add_scalar("transport/offset_curvature", metrics["transport_offset_curvature"], step)
            writer.add_scalar("loss/perceptual", metrics["perceptual"], step)
            writer.add_scalar("loss/inpainter_total", metrics["inpainter_total"], step)
            writer.add_scalar("loss/running_inpainter", running_g, step)
            writer.add_scalar("attention/top1", metrics["attention_top1"], step)
            writer.add_scalar("attention/top4", metrics["attention_top4"], step)
            writer.add_scalar("attention/entropy", metrics["attention_entropy"], step)
            writer.add_scalar("attention/masked_ratio", metrics["attention_masked_ratio"], step)
            if "retrieval_recall1_exact" in metrics:
                writer.add_scalar("retrieval/recall1_exact", metrics["retrieval_recall1_exact"], step)
            if "retrieval_recall1" in metrics:
                writer.add_scalar(f"retrieval/recall1_{retrieval_margin_label}", metrics["retrieval_recall1"], step)
            if "retrieval_recall8" in metrics:
                writer.add_scalar(f"retrieval/recall8_{retrieval_margin_label}", metrics["retrieval_recall8"], step)
            if "retrieval_recall32" in metrics:
                writer.add_scalar(f"retrieval/recall32_{retrieval_margin_label}", metrics["retrieval_recall32"], step)
            if "transport_selection_recall1_exact" in metrics:
                writer.add_scalar("transport_selection/recall1_exact", metrics["transport_selection_recall1_exact"], step)
            if "transport_selection_recall1" in metrics:
                writer.add_scalar(
                    f"transport_selection/recall1_{retrieval_margin_label}",
                    metrics["transport_selection_recall1"],
                    step,
                )
            if "transport_selection_recall8" in metrics:
                writer.add_scalar(
                    f"transport_selection/recall8_{retrieval_margin_label}",
                    metrics["transport_selection_recall8"],
                    step,
                )
            if "transport_selection_recall32" in metrics:
                writer.add_scalar(
                    f"transport_selection/recall32_{retrieval_margin_label}",
                    metrics["transport_selection_recall32"],
                    step,
                )
            if "weight/retrieval_loss" in metrics:
                writer.add_scalar("loss_weight/retrieval", metrics["weight/retrieval_loss"], step)
            if "weight/perceptual" in metrics:
                writer.add_scalar("loss_weight/perceptual", metrics["weight/perceptual"], step)
            writer.add_scalar("lr/inpainter", lr_g, step)
            peak_memory_gb = get_peak_memory_allocated_gb(device)
            if peak_memory_gb is not None:
                writer.add_scalar("accelerator_mem_gb", peak_memory_gb, step)
            write_status(log_cfg["log_dir"], step, total_steps, metrics, lr_g)
            if log_cfg.get("print_train_metrics", False):
                postfix = {
                    "i": f"{metrics['inpainter_total']:.4f}",
                    "l1": f"{metrics['refined_l1']:.4f}",
                    "perc": f"{metrics['perceptual']:.4f}",
                }
                if "refined_query_patch_l1" in metrics:
                    postfix["qp"] = f"{metrics['refined_query_patch_l1']:.4f}"
                if "retrieval_recall1_exact" in metrics:
                    postfix["r1_exact"] = f"{metrics['retrieval_recall1_exact']:.3f}"
                if "retrieval_recall1" in metrics:
                    postfix[f"r1_{retrieval_margin_label}"] = f"{metrics['retrieval_recall1']:.3f}"
                if "retrieval_recall8" in metrics:
                    postfix[f"r8_{retrieval_margin_label}"] = f"{metrics['retrieval_recall8']:.3f}"
                if "retrieval_recall32" in metrics:
                    postfix[f"r32_{retrieval_margin_label}"] = f"{metrics['retrieval_recall32']:.3f}"
                if "transport_selection_recall1_exact" in metrics:
                    postfix["tr1_exact"] = f"{metrics['transport_selection_recall1_exact']:.3f}"
                if "transport_selection_recall1" in metrics:
                    postfix[f"tr1_{retrieval_margin_label}"] = f"{metrics['transport_selection_recall1']:.3f}"
                if "transport_selection_recall8" in metrics:
                    postfix[f"tr8_{retrieval_margin_label}"] = f"{metrics['transport_selection_recall8']:.3f}"
                if "transport_selection_recall32" in metrics:
                    postfix[f"tr32_{retrieval_margin_label}"] = f"{metrics['transport_selection_recall32']:.3f}"
                progress_bar.set_postfix(postfix, refresh=False)

        eval_interval = log_cfg.get("eval_interval", 0)
        if eval_loader is not None and eval_interval and step % eval_interval == 0:
            empty_device_cache(device)
            if eval_sampler is not None:
                eval_sampler.set_epoch(step)
            val_metrics = validate_model(
                model,
                eval_loader,
                device,
                cfg["training"]["mixed_precision"],
                model_image_size,
                dist_ctx,
                criterion=criterion,
                max_batches=log_cfg.get("eval_batches", 8),
            )
            if dist_ctx.is_main_process:
                writer.add_scalar("val/lr_masked_l1_coarse", val_metrics["masked_l1_lr_coarse"], step)
                writer.add_scalar("val/lr_masked_l1_refined", val_metrics["masked_l1_lr_refined"], step)
                if val_metrics["lr_gain_pct"] is not None:
                    writer.add_scalar("val/lr_gain_pct", val_metrics["lr_gain_pct"], step)
                writer.add_scalar("val/hr_masked_l1_coarse_baseline", val_metrics["masked_l1_hr_coarse_baseline"], step)
                writer.add_scalar("val/hr_masked_l1_refined", val_metrics["masked_l1_hr_refined"], step)
                if val_metrics["hr_gain_pct"] is not None:
                    writer.add_scalar("val/hr_gain_pct", val_metrics["hr_gain_pct"], step)
                if "retrieval_recall1_exact" in val_metrics:
                    writer.add_scalar("val/retrieval_recall1_exact", val_metrics["retrieval_recall1_exact"], step)
                if "retrieval_recall1" in val_metrics:
                    writer.add_scalar(f"val/retrieval_recall1_{retrieval_margin_label}", val_metrics["retrieval_recall1"], step)
                if "retrieval_recall8" in val_metrics:
                    writer.add_scalar(f"val/retrieval_recall8_{retrieval_margin_label}", val_metrics["retrieval_recall8"], step)
                if "retrieval_recall32" in val_metrics:
                    writer.add_scalar(f"val/retrieval_recall32_{retrieval_margin_label}", val_metrics["retrieval_recall32"], step)
                if "transport_selection_recall1_exact" in val_metrics:
                    writer.add_scalar(
                        "val/transport_selection/recall1_exact",
                        val_metrics["transport_selection_recall1_exact"],
                        step,
                    )
                if "transport_selection_recall1" in val_metrics:
                    writer.add_scalar(
                        f"val/transport_selection/recall1_{retrieval_margin_label}",
                        val_metrics["transport_selection_recall1"],
                        step,
                    )
                if "transport_selection_recall8" in val_metrics:
                    writer.add_scalar(
                        f"val/transport_selection/recall8_{retrieval_margin_label}",
                        val_metrics["transport_selection_recall8"],
                        step,
                    )
                if "transport_selection_recall32" in val_metrics:
                    writer.add_scalar(
                        f"val/transport_selection/recall32_{retrieval_margin_label}",
                        val_metrics["transport_selection_recall32"],
                        step,
                    )
                write_validation_history(log_cfg["log_dir"], step, val_metrics)
                retrieval_snapshot = format_val_selection_snapshot(
                    val_metrics,
                    retrieval_margin_label=retrieval_margin_label,
                )
                retrieval_suffix = f" | val {retrieval_snapshot}" if retrieval_snapshot else ""
                progress_bar.write(
                    f"Validation step {step}: "
                    f"LR {val_metrics['masked_l1_lr_coarse']:.4f} -> {val_metrics['masked_l1_lr_refined']:.4f} "
                    f"(gain {val_metrics['lr_gain_pct']:.2f}%) | "
                    f"HR {val_metrics['masked_l1_hr_coarse_baseline']:.4f} -> {val_metrics['masked_l1_hr_refined']:.4f} "
                    f"(gain {val_metrics['hr_gain_pct']:.2f}%){retrieval_suffix}\n"
                    f"  {format_train_metric_snapshot(metrics, retrieval_margin_label=retrieval_margin_label)}"
                )
                current_best_metric = val_metrics.get(best_metric_name)
                if save_best_checkpoint and is_better_metric(current_best_metric, best_metric_value, best_metric_mode):
                    best_metric_value = float(current_best_metric)
                    best_metric_step = step
                    best_path = ckpt_dir / "best.pth"
                    best_metrics = build_checkpoint_metrics(
                        metrics,
                        best_metric_name,
                        best_metric_mode,
                        best_metric_value,
                        best_metric_step,
                    )
                    save_checkpoint(
                        model,
                        optimizer_g,
                        scaler,
                        step,
                        best_metrics,
                        cfg,
                        best_path,
                    )
                    progress_bar.write(
                        f"Saved best checkpoint: {best_path} "
                        f"({best_metric_name}={best_metric_value:.6f} at step {best_metric_step})"
                    )
            empty_device_cache(device)
            barrier(dist_ctx)

        if dist_ctx.is_main_process and step % log_cfg["vis_interval"] == 0 and last_batch_views is not None:
            vis_coarse, vis_refined = render_visualization_batch(
                model,
                last_batch_views,
                device,
                cfg["training"]["mixed_precision"],
            )
            save_vis(writer, last_batch_views, vis_coarse, vis_refined, step, log_dir=log_cfg["log_dir"])

        checkpoint_steps = log_cfg.get("checkpoint_steps")
        should_save = False
        if checkpoint_steps:
            should_save = step in checkpoint_steps
        elif log_cfg.get("save_checkpoints", True):
            should_save = step % log_cfg["save_interval"] == 0

        if dist_ctx.is_main_process and should_save:
            ckpt_path = ckpt_dir / f"step_{step}.pth"
            checkpoint_metrics = build_checkpoint_metrics(
                metrics,
                best_metric_name,
                best_metric_mode,
                best_metric_value,
                best_metric_step,
            )
            save_checkpoint(
                model,
                optimizer_g,
                scaler,
                step,
                checkpoint_metrics,
                cfg,
                ckpt_path,
            )
            progress_bar.write(f"Saved checkpoint: {ckpt_path}")

    final_path = ckpt_dir / f"step_{total_steps}.pth"
    if dist_ctx.is_main_process and log_cfg.get("save_final_checkpoint", True):
        final_metrics = build_checkpoint_metrics(
            metrics,
            best_metric_name,
            best_metric_mode,
            best_metric_value,
            best_metric_step,
        )
        save_checkpoint(
            model,
            optimizer_g,
            scaler,
            total_steps,
            final_metrics,
            cfg,
            final_path,
        )
        progress_bar.write(f"Training complete. Final checkpoint: {final_path}")
    elif dist_ctx.is_main_process:
        progress_bar.write("Training complete. Final checkpoint saving disabled.")
    progress_bar.close()
    writer.close()
    barrier(dist_ctx)


def main():
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda, xpu, cpu")
    parser.add_argument("--overfit", type=int, default=None, help="Overfit on N images for a quick sanity check")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--eval-only", action="store_true", help="Run validation only on a checkpoint")
    parser.add_argument("--eval-batches", type=int, default=None, help="Override validation batches; 0 means full val")
    parser.add_argument(
        "--force-random-masks",
        action="store_true",
        help="Ignore manifest masks and generate random free-form masks for the requested split.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    dist_ctx = init_distributed(args.device)
    try:
        if args.eval_only:
            run_eval_only(cfg, args, dist_ctx)
        else:
            train(cfg, args, dist_ctx)
    finally:
        destroy_distributed(dist_ctx)


if __name__ == "__main__":
    main()
