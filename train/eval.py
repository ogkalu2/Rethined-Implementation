from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from losses import InpaintingLoss
from model import InpaintingModel
from upscale import AttentionUpscaling
from device_utils import get_autocast_device_type, is_amp_enabled
from distributed_utils import barrier, reduce_metrics, unwrap_model

from .checkpoints import load_eval_checkpoint
from .common import (
    build_model_config,
    composite_with_known,
    masked_l1,
    prepare_multiscale_batch,
    print_device_banner,
    seed_everything,
)
from .data import build_eval_loader


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


def run_eval_only(cfg, args, dist_ctx):
    device = dist_ctx.device
    if dist_ctx.is_main_process:
        print_device_banner(device)
    seed_everything(cfg["training"]["seed"])

    model = InpaintingModel(build_model_config(cfg)).to(device)
    if not args.resume:
        raise ValueError("--eval-only requires --resume CHECKPOINT")
    load_eval_checkpoint(args.resume, model, device)
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
