"""Paper-aligned RETHINED training entry point."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from data.dataset import get_dataloader
from device_utils import (
    empty_device_cache,
    get_autocast_device_type,
    get_device_name,
    get_peak_memory_allocated_gb,
    is_amp_enabled,
    resolve_device,
)
from discriminator import PatchDiscriminator
from hr import AttentionUpscaling
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
    generator_params = {
        k: v
        for k, v in cfg["model"]["generator"].items()
        if k != "copy_mode" and not k.startswith("transport_")
    }
    return {
        "coarse_model": {
            "class": coarse_cfg.get("class", "CoarseModel"),
            "parameters": {k: v for k, v in coarse_cfg.items() if k != "class"},
        },
        "generator": {
            "generator_class": "PatchInpainting",
            "params": generator_params,
        },
    }


def _collect_matcher_parameters(model: InpaintingModel) -> tuple[list[torch.nn.Parameter], set[int]]:
    generator = model.generator
    matcher_params: list[torch.nn.Parameter] = []
    matcher_param_ids: set[int] = set()

    def add_param(param: torch.nn.Parameter | None):
        if param is None or (not param.requires_grad) or id(param) in matcher_param_ids:
            return
        matcher_params.append(param)
        matcher_param_ids.add(id(param))

    matcher_module_names = (
        "query_context_encoder",
        "key_context_encoder",
        "query_context_descriptor_head",
        "shared_query_key_descriptor_head",
        "query_descriptor_head",
        "key_descriptor_head",
        "matching_descriptor_head",
        "patch_reranker",
        "pre_attention_norm",
        "multihead_attention",
    )
    for module_name in matcher_module_names:
        module = getattr(generator, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            add_param(param)

    for param_name in (
        "positionalencoding",
        "key_coarse_rgb_scale",
        "key_feature_scale",
        "query_context_scale",
    ):
        param = getattr(generator, param_name, None)
        if isinstance(param, torch.nn.Parameter):
            add_param(param)

    return matcher_params, matcher_param_ids


def build_generator_optimizer(
    model: InpaintingModel,
    *,
    base_lr: float,
    betas: tuple[float, float],
    matcher_lr_scale: float = 1.0,
) -> tuple[torch.optim.Optimizer, bool]:
    matcher_lr_scale = float(matcher_lr_scale)
    matcher_params, matcher_param_ids = _collect_matcher_parameters(model)
    base_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in matcher_param_ids
    ]
    param_groups: list[dict[str, object]] = []
    if base_params:
        param_groups.append(
            {
                "params": base_params,
                "lr": base_lr,
                "lr_scale": 1.0,
                "group_name": "base",
            }
        )
    if matcher_params:
        param_groups.append(
            {
                "params": matcher_params,
                "lr": base_lr * matcher_lr_scale,
                "lr_scale": matcher_lr_scale,
                "group_name": "matcher",
            }
        )
    optimizer = torch.optim.Adam(param_groups, lr=base_lr, betas=betas)
    return optimizer, bool(matcher_params)


def set_parameter_requires_grad(parameters: list[torch.nn.Parameter], requires_grad: bool):
    for param in parameters:
        param.requires_grad_(requires_grad)


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


def set_discriminator_requires_grad(discriminator: torch.nn.Module, enabled: bool):
    for param in discriminator.parameters():
        param.requires_grad = enabled


def save_checkpoint(
    model,
    discriminator,
    optimizer_g,
    optimizer_d,
    scaler,
    step,
    metrics,
    cfg,
    path,
):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def load_model_checkpoint(model, state_dict):
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if ".transport_" not in key and "transport_" not in key
    }
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    unexpected_non_transport = [key for key in unexpected_keys if "transport_" not in key]
    if missing_keys or unexpected_non_transport:
        raise RuntimeError(
            "Checkpoint is incompatible with the current attention-only model. "
            f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_non_transport}."
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
    discriminator,
    optimizer_g,
    optimizer_d,
    scaler,
    device,
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    required_keys = (
        "step",
        "model_state_dict",
        "discriminator_state_dict",
        "optimizer_g_state_dict",
        "optimizer_d_state_dict",
        "scaler_state_dict",
    )
    missing_keys = [key for key in required_keys if key not in ckpt]
    if missing_keys:
        raise KeyError(f"Expected a full training checkpoint, missing keys: {', '.join(missing_keys)}")
    load_model_checkpoint(model, ckpt["model_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"], strict=True)
    optimizer_g.load_state_dict(ckpt["optimizer_g_state_dict"])
    optimizer_d.load_state_dict(ckpt["optimizer_d_state_dict"])
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


def format_train_metric_snapshot(metrics):
    summary = (
        f"train g={metrics['generator_total']:.4f}, "
        f"d={metrics['discriminator_total']:.4f}, "
        f"l1={metrics['refined_l1']:.4f}, "
        f"ff={metrics['frequency']:.4f}, "
        f"perc={metrics['perceptual']:.4f}"
    )
    if "refined_query_patch_l1" in metrics:
        summary += f", qp={metrics['refined_query_patch_l1']:.4f}"
    if "retrieval_recall1" in metrics:
        summary += f", r1={metrics['retrieval_recall1']:.3f}"
    if "reranker_recall1" in metrics:
        summary += f", rr1={metrics['reranker_recall1']:.3f}"
    if "reranker_shortlist_acc" in metrics:
        summary += f", rsa={metrics['reranker_shortlist_acc']:.3f}"
    if "retrieval_recall8" in metrics:
        summary += f", r8={metrics['retrieval_recall8']:.3f}"
    if "retrieval_recall32" in metrics:
        summary += f", r32={metrics['retrieval_recall32']:.3f}"
    if "retrieval_coord_error" in metrics:
        summary += f", coord={metrics['retrieval_coord_error']:.3f}"
    if "boundary_identity_acc" in metrics:
        summary += f", bia={metrics['boundary_identity_acc']:.3f}"
    return summary


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
def validate_model(model, dataloader, device, use_amp, model_image_size, max_batches=8):
    model.eval()
    attn_upscaler = AttentionUpscaling(model.generator)
    amp_device_type = get_autocast_device_type(device)
    amp_enabled = is_amp_enabled(device, use_amp)

    lr_coarse_values = []
    lr_refined_values = []
    lr_gain_values = []
    hr_coarse_values = []
    hr_refined_values = []
    hr_gain_values = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
            break

        batch_views = prepare_multiscale_batch(
            batch,
            device,
            model_image_size,
            blur_layer=model.generator.final_gaussian_blur,
        )
        mask = batch_views["mask"]
        masked_image = batch_views["masked_image"]
        refine_target = batch_views["refine_target"]

        with torch.amp.autocast(amp_device_type, enabled=amp_enabled):
            refined_lr, attn_map, coarse_raw = model(
                masked_image,
                mask,
                value_image=refine_target,
            )

        refined_lr = refined_lr.clamp(0, 1)
        coarse_lr = composite_with_known(coarse_raw.clamp(0, 1), refine_target, mask)
        lr_coarse_err = masked_l1(coarse_lr, refine_target, mask)
        lr_refined_err = masked_l1(refined_lr, refine_target, mask)
        lr_coarse_values.extend(lr_coarse_err.tolist())
        lr_refined_values.extend(lr_refined_err.tolist())
        lr_gain_values.extend((lr_coarse_err - lr_refined_err).tolist())

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
            hr_coarse_values.extend(hr_coarse_err.tolist())
            hr_refined_values.extend(hr_refined_err.tolist())
            hr_gain_values.extend((hr_coarse_err - hr_refined_err).tolist())
        else:
            hr_coarse_values.extend(lr_coarse_err.tolist())
            hr_refined_values.extend(lr_refined_err.tolist())
            hr_gain_values.extend((lr_coarse_err - lr_refined_err).tolist())

    model.train()
    return {
        "masked_l1_lr_coarse": mean_metric(lr_coarse_values),
        "masked_l1_lr_refined": mean_metric(lr_refined_values),
        "lr_gain_abs": mean_metric(lr_gain_values),
        "lr_gain_pct": (
            100.0 * (mean_metric(lr_gain_values) / max(mean_metric(lr_coarse_values), 1e-8))
            if lr_coarse_values
            else None
        ),
        "masked_l1_hr_coarse_baseline": mean_metric(hr_coarse_values),
        "masked_l1_hr_refined": mean_metric(hr_refined_values),
        "hr_gain_abs": mean_metric(hr_gain_values),
        "hr_gain_pct": (
            100.0 * (mean_metric(hr_gain_values) / max(mean_metric(hr_coarse_values), 1e-8))
            if hr_coarse_values
            else None
        ),
    }


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


def build_train_loader(cfg, args):
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
        shuffle_override=(False if args.overfit else None),
    )


def build_eval_loader(cfg, args):
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
        shuffle_override=False,
    )


def run_eval_only(cfg, args):
    device = resolve_device(args.device)
    print_device_banner(device)
    seed_everything(cfg["training"]["seed"])

    model = InpaintingModel(build_model_config(cfg)).to(device)
    if not args.resume:
        raise ValueError("--eval-only requires --resume CHECKPOINT")
    checkpoint_step = load_eval_checkpoint(args.resume, model, device)
    model.generator.set_training_step(checkpoint_step)
    eval_loader = build_eval_loader(cfg, args)
    if eval_loader is None:
        raise ValueError("No evaluation loader configured.")

    health = validate_model(
        model,
        eval_loader,
        device,
        cfg["training"]["mixed_precision"],
        model.generator.image_size,
        max_batches=(args.eval_batches if args.eval_batches is not None else cfg.get("logging", {}).get("eval_batches", 8)),
    )
    print(json.dumps(health, indent=2))

    log_dir = Path(cfg["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "eval_only_full_val.json", "w", encoding="utf-8") as handle:
        json.dump(health, handle, indent=2)


def train(cfg, args):
    device = resolve_device(args.device)
    print_device_banner(device)
    seed_everything(cfg["training"]["seed"])

    model = InpaintingModel(build_model_config(cfg)).to(device)
    discriminator = PatchDiscriminator(**cfg["discriminator"]).to(device)
    loss_cfg = {k: v for k, v in cfg["loss"].items() if not k.startswith("transport_")}
    criterion = InpaintingLoss(**loss_cfg).to(device)
    optimizer_betas = tuple(cfg["training"].get("betas", [0.9, 0.999]))
    matcher_lr_scale = float(cfg["training"].get("matcher_lr_scale", 1.0))
    route_image_losses_away_from_matcher = bool(
        cfg["training"].get("route_image_losses_away_from_matcher", False)
    )

    optimizer_g, has_matcher_group = build_generator_optimizer(
        model,
        base_lr=cfg["training"]["lr"],
        betas=optimizer_betas,
        matcher_lr_scale=matcher_lr_scale,
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=cfg["training"].get("discriminator_lr", cfg["training"]["lr"]),
        betas=optimizer_betas,
    )

    use_amp = is_amp_enabled(device, cfg["training"]["mixed_precision"])
    if cfg["training"]["mixed_precision"] and not use_amp:
        print("Mixed precision requested, but no supported accelerator is active; disabling AMP.")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    train_loader = build_train_loader(cfg, args)
    eval_loader = build_eval_loader(cfg, args)
    print(f"Training images: {len(train_loader.dataset)}")
    if eval_loader is not None:
        print(f"Validation images: {len(eval_loader.dataset)}")

    log_cfg = cfg["logging"]
    total_steps = args.steps if args.steps else cfg["training"]["total_steps"]
    grad_accum = 1 if args.overfit else cfg["training"]["grad_accum_steps"]
    max_lr = cfg["training"]["lr"]
    min_lr = cfg["training"]["min_lr"]
    warmup_steps = cfg["training"]["warmup_steps"]
    grad_clip_g = cfg["training"].get("grad_clip", 1.0)
    grad_clip_d = cfg["training"].get("discriminator_grad_clip", grad_clip_g)
    model_image_size = model.generator.image_size

    print(f"Generator parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"Effective batch size: {cfg['data']['batch_size'] * grad_accum}")
    if has_matcher_group:
        print(f"Matcher/reranker LR scale: {matcher_lr_scale:.2f}x")
    if has_matcher_group and route_image_losses_away_from_matcher:
        print("Gradient routing enabled: image losses bypass matcher/reranker params")

    log_dir = Path(log_cfg["log_dir"])
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = create_summary_writer(log_dir / "tb")

    start_step = 0
    best_metric_name = log_cfg.get("save_best_metric", "masked_l1_hr_refined")
    best_metric_mode = log_cfg.get("save_best_mode", "min")
    save_best_checkpoint = log_cfg.get("save_best_checkpoint", True)
    best_metric_value = None
    best_metric_step = None
    if args.resume:
        resume_state = load_training_checkpoint(
            args.resume,
            model,
            discriminator,
            optimizer_g,
            optimizer_d,
            scaler,
            device,
        )
        start_step = resume_state["step"]
        if resume_state["best_metric_name"] == best_metric_name and resume_state["best_metric_mode"] == best_metric_mode:
            best_metric_value = resume_state["best_metric_value"]
            best_metric_step = resume_state["best_metric_step"]
        print(f"Resumed from step {start_step}")

    amp_device_type = get_autocast_device_type(device)
    data_iter = iter(train_loader)
    running_g = 0.0
    running_d = 0.0
    last_batch_views = None
    last_coarse = None
    last_refined = None
    metrics = {}
    matcher_group_lr = cfg["training"]["lr"] * matcher_lr_scale
    generator_base_params = []
    generator_matcher_params = []
    for param_group in optimizer_g.param_groups:
        if param_group.get("group_name") == "matcher":
            generator_matcher_params.extend(param_group["params"])
        else:
            generator_base_params.extend(param_group["params"])

    progress_bar = tqdm(range(start_step, total_steps), desc="Training", dynamic_ncols=True)
    for step_idx in progress_bar:
        step = step_idx + 1
        model.generator.set_training_step(step)
        criterion.set_training_step(step)
        lr_g = get_lr(step, warmup_steps, total_steps, max_lr, min_lr)
        lr_d = get_lr(
            step,
            warmup_steps,
            total_steps,
            cfg["training"].get("discriminator_lr", max_lr),
            cfg["training"].get("discriminator_min_lr", min_lr),
        )
        for pg in optimizer_g.param_groups:
            pg["lr"] = lr_g * float(pg.get("lr_scale", 1.0))
            if pg.get("group_name") == "matcher":
                matcher_group_lr = pg["lr"]
        for pg in optimizer_d.param_groups:
            pg["lr"] = lr_d

        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        metric_sums = defaultdict(float)
        step_has_nonfinite = False

        for _ in range(grad_accum):
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
            refine_target = batch_views["refine_target"]
            use_gradient_routing = route_image_losses_away_from_matcher and has_matcher_group

            with torch.amp.autocast(amp_device_type, enabled=use_amp):
                refined_raw, attn_map, coarse_raw, attention_aux = model(
                    masked_image,
                    mask,
                    value_image=refine_target,
                    return_aux=True,
                )
                refined_vis = refined_raw.clamp(0, 1) 
                coarse_vis = composite_with_known(coarse_raw.clamp(0, 1), refine_target, mask)

                set_discriminator_requires_grad(discriminator, True)
                real_logits = discriminator(refine_target)
                fake_logits_d = discriminator(refined_vis.detach()) 
                d_loss, d_metrics = criterion.discriminator_loss(real_logits, fake_logits_d)

            if not torch.isfinite(d_loss):
                step_has_nonfinite = True
                break

            scaler.scale(d_loss / grad_accum).backward()

            with torch.amp.autocast(amp_device_type, enabled=use_amp):
                set_discriminator_requires_grad(discriminator, False)
                fake_logits_g = discriminator(refined_vis) 
                g_loss_result = criterion.generator_loss(
                    coarse_raw,
                    refined_raw,
                    image,
                    refine_target,
                    mask,
                    fake_logits_g,
                    attention_aux=attention_aux,
                    return_components=use_gradient_routing,
                )
                if use_gradient_routing:
                    g_loss, g_metrics, g_components = g_loss_result
                else:
                    g_loss, g_metrics = g_loss_result

            if not torch.isfinite(g_loss):
                step_has_nonfinite = True
                break

            if use_gradient_routing:
                matcher_loss = (
                    g_components["refined_l1"]
                    + g_components["refined_query_patch_l1"]
                    + g_components["retrieval"]
                    + g_components["reranker"]
                    + g_components["boundary_identity"]
                    + g_components["coordinate"]
                    + g_components["coherence"]
                )
                base_loss = (
                    g_components["coarse_l2"]
                    + g_components["coarse_blur_l1"]
                    + g_components["coarse_gradient"]
                    + g_components["coarse_perceptual"]
                    + g_components["refined_l1"]
                    + g_components["refined_query_patch_l1"]
                    + g_components["frequency"]
                    + g_components["perceptual"]
                    + g_components["adversarial"]
                )
                if generator_matcher_params:
                    scaler.scale(matcher_loss / grad_accum).backward(
                        inputs=generator_matcher_params,
                        retain_graph=bool(generator_base_params),
                    )
                if generator_base_params:
                    scaler.scale(base_loss / grad_accum).backward(inputs=generator_base_params)
            else:
                scaler.scale(g_loss / grad_accum).backward()

            attn_metrics = model.generator.summarize_attention(
                attn_map.detach(),
                model.generator.flatten_query_mask(mask).detach(),
            )

            for key, value in g_metrics.items():
                metric_sums[key] += value
            for key, value in d_metrics.items():
                metric_sums[key] += value
            for key, value in attn_metrics.items():
                metric_sums[key] += value

            last_batch_views = batch_views
            last_coarse = coarse_vis.detach()
            last_refined = refined_vis.detach()

        if step_has_nonfinite:
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            progress_bar.write(f"Skipping non-finite step {step}")
            continue

        scaler.unscale_(optimizer_d)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip_d)
        scaler.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_g)
        scaler.step(optimizer_d)
        scaler.step(optimizer_g)
        scaler.update()
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)

        metrics = {key: value / grad_accum for key, value in metric_sums.items()}
        running_g = 0.9 * running_g + 0.1 * metrics["generator_total"]
        running_d = 0.9 * running_d + 0.1 * metrics["discriminator_total"]

        if step == 1 or step % log_cfg["log_interval"] == 0:
            writer.add_scalar("loss/coarse_l2", metrics["coarse_l2"], step)
            if "coarse_blur_l1" in metrics:
                writer.add_scalar("loss/coarse_blur_l1", metrics["coarse_blur_l1"], step)
            if "coarse_gradient" in metrics:
                writer.add_scalar("loss/coarse_gradient", metrics["coarse_gradient"], step)
            writer.add_scalar("loss/coarse_perceptual", metrics["coarse_perceptual"], step)
            writer.add_scalar("loss/refined_l1", metrics["refined_l1"], step)
            if "refined_query_patch_l1" in metrics:
                writer.add_scalar("loss/refined_query_patch_l1", metrics["refined_query_patch_l1"], step)
            if "retrieval_loss" in metrics:
                writer.add_scalar("loss/retrieval", metrics["retrieval_loss"], step)
            if "reranker_loss" in metrics:
                writer.add_scalar("loss/reranker", metrics["reranker_loss"], step)
            if "boundary_identity_loss" in metrics:
                writer.add_scalar("loss/boundary_identity", metrics["boundary_identity_loss"], step)
            if "coordinate_loss" in metrics:
                writer.add_scalar("loss/coordinate", metrics["coordinate_loss"], step)
            if "coherence_loss" in metrics:
                writer.add_scalar("loss/coherence", metrics["coherence_loss"], step)
            writer.add_scalar("loss/frequency", metrics["frequency"], step)
            writer.add_scalar("loss/perceptual", metrics["perceptual"], step)
            writer.add_scalar("loss/adversarial_g", metrics["adversarial_g"], step)
            writer.add_scalar("loss/adversarial_d_real", metrics["adversarial_d_real"], step)
            writer.add_scalar("loss/adversarial_d_fake", metrics["adversarial_d_fake"], step)
            writer.add_scalar("loss/generator_total", metrics["generator_total"], step)
            writer.add_scalar("loss/discriminator_total", metrics["discriminator_total"], step)
            writer.add_scalar("loss/running_generator", running_g, step)
            writer.add_scalar("loss/running_discriminator", running_d, step)
            writer.add_scalar("attention/top1", metrics["attention_top1"], step)
            writer.add_scalar("attention/top4", metrics["attention_top4"], step)
            writer.add_scalar("attention/entropy", metrics["attention_entropy"], step)
            writer.add_scalar("attention/masked_ratio", metrics["attention_masked_ratio"], step)
            if "retrieval_recall1" in metrics:
                writer.add_scalar("retrieval/recall1", metrics["retrieval_recall1"], step)
            if "reranker_recall1" in metrics:
                writer.add_scalar("retrieval/rerank_recall1", metrics["reranker_recall1"], step)
            if "retrieval_recall8" in metrics:
                writer.add_scalar("retrieval/recall8", metrics["retrieval_recall8"], step)
            if "retrieval_recall32" in metrics:
                writer.add_scalar("retrieval/recall32", metrics["retrieval_recall32"], step)
            if "reranker_shortlist_acc" in metrics:
                writer.add_scalar("retrieval/rerank_shortlist_acc", metrics["reranker_shortlist_acc"], step)
            if "retrieval_coord_error" in metrics:
                writer.add_scalar("retrieval/coord_error", metrics["retrieval_coord_error"], step)
            if "boundary_identity_acc" in metrics:
                writer.add_scalar("retrieval/boundary_identity_acc", metrics["boundary_identity_acc"], step)
            if "weight/retrieval_loss" in metrics:
                writer.add_scalar("loss_weight/retrieval", metrics["weight/retrieval_loss"], step)
            if "weight/boundary_identity" in metrics:
                writer.add_scalar("loss_weight/boundary_identity", metrics["weight/boundary_identity"], step)
            if "weight/coordinate" in metrics:
                writer.add_scalar("loss_weight/coordinate", metrics["weight/coordinate"], step)
            if "weight/coherence" in metrics:
                writer.add_scalar("loss_weight/coherence", metrics["weight/coherence"], step)
            if "weight/frequency" in metrics:
                writer.add_scalar("loss_weight/frequency", metrics["weight/frequency"], step)
            if "weight/perceptual" in metrics:
                writer.add_scalar("loss_weight/perceptual", metrics["weight/perceptual"], step)
            if "weight/adversarial" in metrics:
                writer.add_scalar("loss_weight/adversarial", metrics["weight/adversarial"], step)
            writer.add_scalar("lr/generator", lr_g, step)
            if has_matcher_group:
                writer.add_scalar("lr/generator_matcher", matcher_group_lr, step)
            writer.add_scalar("lr/discriminator", lr_d, step)
            peak_memory_gb = get_peak_memory_allocated_gb(device)
            if peak_memory_gb is not None:
                writer.add_scalar("accelerator_mem_gb", peak_memory_gb, step)
            write_status(log_cfg["log_dir"], step, total_steps, metrics, lr_g)
            if log_cfg.get("print_train_metrics", False):
                progress_bar.set_postfix(
                    g=f"{metrics['generator_total']:.4f}",
                    d=f"{metrics['discriminator_total']:.4f}",
                    l1=f"{metrics['refined_l1']:.4f}",
                    qp=(f"{metrics['refined_query_patch_l1']:.4f}" if "refined_query_patch_l1" in metrics else "n/a"),
                    rr1=(f"{metrics['reranker_recall1']:.3f}" if "reranker_recall1" in metrics else "n/a"),
                    rsa=(f"{metrics['reranker_shortlist_acc']:.3f}" if "reranker_shortlist_acc" in metrics else "n/a"),
                    r8=(f"{metrics['retrieval_recall8']:.3f}" if "retrieval_recall8" in metrics else "n/a"),
                    r32=(f"{metrics['retrieval_recall32']:.3f}" if "retrieval_recall32" in metrics else "n/a"),
                    coord=(f"{metrics['retrieval_coord_error']:.3f}" if "retrieval_coord_error" in metrics else "n/a"),
                    ff=f"{metrics['frequency']:.4f}",
                    refresh=False,
                )

        eval_interval = log_cfg.get("eval_interval", 0)
        if eval_loader is not None and eval_interval and step % eval_interval == 0:
            empty_device_cache(device)
            val_metrics = validate_model(
                model,
                eval_loader,
                device,
                cfg["training"]["mixed_precision"],
                model_image_size,
                max_batches=log_cfg.get("eval_batches", 8),
            )
            writer.add_scalar("val/lr_masked_l1_coarse", val_metrics["masked_l1_lr_coarse"], step)
            writer.add_scalar("val/lr_masked_l1_refined", val_metrics["masked_l1_lr_refined"], step)
            if val_metrics["lr_gain_pct"] is not None:
                writer.add_scalar("val/lr_gain_pct", val_metrics["lr_gain_pct"], step)
            writer.add_scalar("val/hr_masked_l1_coarse_baseline", val_metrics["masked_l1_hr_coarse_baseline"], step)
            writer.add_scalar("val/hr_masked_l1_refined", val_metrics["masked_l1_hr_refined"], step)
            if val_metrics["hr_gain_pct"] is not None:
                writer.add_scalar("val/hr_gain_pct", val_metrics["hr_gain_pct"], step)
            write_validation_history(log_cfg["log_dir"], step, val_metrics)
            progress_bar.write(
                f"Validation step {step}: "
                f"LR {val_metrics['masked_l1_lr_coarse']:.4f} -> {val_metrics['masked_l1_lr_refined']:.4f} "
                f"(gain {val_metrics['lr_gain_pct']:.2f}%) | "
                f"HR {val_metrics['masked_l1_hr_coarse_baseline']:.4f} -> {val_metrics['masked_l1_hr_refined']:.4f} "
                f"(gain {val_metrics['hr_gain_pct']:.2f}%)\n"
                f"  {format_train_metric_snapshot(metrics)}"
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
                    discriminator,
                    optimizer_g,
                    optimizer_d,
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

        if step % log_cfg["vis_interval"] == 0 and last_batch_views is not None:
            save_vis(writer, last_batch_views, last_coarse, last_refined, step, log_dir=log_cfg["log_dir"])

        checkpoint_steps = log_cfg.get("checkpoint_steps")
        should_save = False
        if checkpoint_steps:
            should_save = step in checkpoint_steps
        elif log_cfg.get("save_checkpoints", True):
            should_save = step % log_cfg["save_interval"] == 0

        if should_save:
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
                discriminator,
                optimizer_g,
                optimizer_d,
                scaler,
                step,
                checkpoint_metrics,
                cfg,
                ckpt_path,
            )
            progress_bar.write(f"Saved checkpoint: {ckpt_path}")

    final_path = ckpt_dir / f"step_{total_steps}.pth"
    if log_cfg.get("save_final_checkpoint", True):
        final_metrics = build_checkpoint_metrics(
            metrics,
            best_metric_name,
            best_metric_mode,
            best_metric_value,
            best_metric_step,
        )
        save_checkpoint(
            model,
            discriminator,
            optimizer_g,
            optimizer_d,
            scaler,
            total_steps,
            final_metrics,
            cfg,
            final_path,
        )
        progress_bar.write(f"Training complete. Final checkpoint: {final_path}")
    else:
        progress_bar.write("Training complete. Final checkpoint saving disabled.")
    progress_bar.close()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Paper-aligned RETHINED training")
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

    if args.eval_only:
        run_eval_only(cfg, args)
    else:
        train(cfg, args)


if __name__ == "__main__":
    main()
