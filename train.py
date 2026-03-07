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

from model import InpaintingModel
from losses import InpaintingLoss
from data.dataset import get_dataloader


def build_model_config(cfg):
    """Build the model config dict from YAML config."""
    return {
        "coarse_model": {
            "class": "MobileOneCoarse",
            "parameters": {"variant": cfg["model"]["coarse_model"]["variant"]},
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


def composite_with_known(pred, target, mask):
    """Preserve known pixels exactly and only use predictions inside the hole."""
    return pred * mask + target * (1 - mask)


def set_parameter_trainability(model, train_mode: str, freeze_coarse: bool = False):
    """Configure which parts of the model should receive gradients."""
    coarse_param_ids = {id(param) for param in model.coarse_model.parameters()}
    for param in model.parameters():
        is_coarse = id(param) in coarse_param_ids
        if train_mode in ("joint", "refined_loss_only"):
            param.requires_grad = not (freeze_coarse and is_coarse)
        elif train_mode == "coarse_only":
            param.requires_grad = is_coarse
        elif train_mode == "refine_only":
            param.requires_grad = not is_coarse
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    if train_mode == "refine_only" or freeze_coarse:
        model.coarse_model.eval()


def compute_selective_gate_targets(model, target, margin: float = 0.0):
    """Build patch-level supervision for the learned refinement gate."""
    generator = getattr(model, "generator", None)
    if generator is None:
        return None

    gate = getattr(generator, "last_refinement_gate_map", None)
    candidate = getattr(generator, "last_candidate_patches_flat", None)
    base = getattr(generator, "last_base_patches_flat", None)
    pixel_mask = getattr(generator, "last_pixel_mask_flat", None)
    if gate is None or candidate is None or base is None or pixel_mask is None:
        return None

    with torch.no_grad():
        gt_patches, _ = generator.unfold_native(target, generator.kernel_size)
        gt_patches = gt_patches.flatten(start_dim=2).transpose(1, 2)

        pixel_mask_detached = pixel_mask.detach()
        valid_patches = pixel_mask_detached.sum(dim=-1) > 0
        if not valid_patches.any():
            return None

        mask_count = pixel_mask_detached.sum(dim=-1).clamp_min(1.0)
        base_patch_err = (torch.abs(base.detach() - gt_patches) * pixel_mask_detached).sum(dim=-1) / mask_count
        candidate_patch_err = (torch.abs(candidate.detach() - gt_patches) * pixel_mask_detached).sum(dim=-1) / mask_count
        gate_target = ((candidate_patch_err + margin) < base_patch_err).float()
        advantage = base_patch_err - candidate_patch_err

    gate_pred = (gate * pixel_mask).sum(dim=-1) / pixel_mask.sum(dim=-1).clamp_min(1.0)
    return {
        "gate_pred": gate_pred.float(),
        "gate_target": gate_target.float(),
        "valid_patches": valid_patches,
        "advantage": advantage.float(),
    }



def compute_selective_gate_loss(model, target, weight: float, margin: float):
    """Teach the gate to open only where the raw candidate beats coarse."""
    zero = target.new_zeros(())
    empty_metrics = {
        "gate_target_rate": None,
        "gate_pred_mean": None,
        "gate_accuracy": None,
        "gate_advantage_mean": None,
    }
    if weight <= 0:
        return zero, empty_metrics

    stats = compute_selective_gate_targets(model, target, margin=margin)
    if stats is None:
        return zero, empty_metrics

    valid_patches = stats["valid_patches"]
    gate_pred = stats["gate_pred"][valid_patches].clamp(1e-4, 1 - 1e-4)
    gate_target = stats["gate_target"][valid_patches]
    advantage = stats["advantage"][valid_patches]

    with torch.amp.autocast("cuda", enabled=False):
        gate_pred = gate_pred.float()
        gate_target = gate_target.float()
        advantage = advantage.float()
        raw_bce = F.binary_cross_entropy(gate_pred, gate_target, reduction="none")
        advantage_weight = (
            advantage.abs() / advantage.abs().mean().clamp_min(1e-6)
        ).detach().clamp(0.25, 4.0)
        gate_loss = (raw_bce * advantage_weight).mean()

    gate_binary = (gate_pred >= 0.5).float()
    gate_metrics = {
        "gate_target_rate": gate_target.mean().item(),
        "gate_pred_mean": gate_pred.mean().item(),
        "gate_accuracy": (gate_binary == gate_target).float().mean().item(),
        "gate_advantage_mean": advantage.mean().item(),
    }
    return gate_loss, gate_metrics


def compute_sample_improvement_loss(coarse, refined, target, mask, margin: float):
    """Encourage refined output to beat coarse on each sample, not only on average."""
    hole_denom = (mask.sum(dim=(1, 2, 3)) * target.shape[1]).clamp_min(1e-8)
    coarse_hole = (torch.abs(coarse.detach() - target) * mask).sum(dim=(1, 2, 3)) / hole_denom
    refined_hole = (torch.abs(refined - target) * mask).sum(dim=(1, 2, 3)) / hole_denom
    improvement_hinge = F.relu(refined_hole - coarse_hole + margin)
    metrics = {
        "sample_better_rate_batch": (refined_hole < coarse_hole).float().mean().item(),
        "sample_margin_violations": (improvement_hinge > 0).float().mean().item(),
        "sample_improvement_gap": (coarse_hole - refined_hole).mean().item(),
    }
    return improvement_hinge.mean(), metrics


def compute_selector_choice_loss(model, target, margin: float):
    """Supervise coarse-vs-head hypothesis selection in patch space."""
    generator = getattr(model, "generator", None)
    logits = getattr(generator, "last_selector_logits", None)
    candidate_bank = getattr(generator, "last_candidate_bank", None)
    pixel_mask = getattr(generator, "last_pixel_mask_flat", None)
    if generator is None or logits is None or candidate_bank is None or pixel_mask is None:
        zero = target.new_zeros(())
        empty_metrics = {
            "selector_accuracy": None,
            "selector_pred_noncoarse_rate": None,
            "selector_target_noncoarse_rate": None,
            "selector_advantage_mean": None,
        }
        return zero, empty_metrics

    with torch.no_grad():
        gt_patches, _ = generator.unfold_native(target, generator.kernel_size)
        gt_patches = gt_patches.flatten(start_dim=2).transpose(1, 2)

        pixel_mask_detached = pixel_mask.detach()
        valid_patches = pixel_mask_detached.sum(dim=-1) > 0
        if not valid_patches.any():
            zero = target.new_zeros(())
            empty_metrics = {
                "selector_accuracy": None,
                "selector_pred_noncoarse_rate": None,
                "selector_target_noncoarse_rate": None,
                "selector_advantage_mean": None,
            }
            return zero, empty_metrics

        mask_count = pixel_mask_detached.sum(dim=-1).clamp_min(1.0)
        candidate_err = (
            torch.abs(candidate_bank.detach() - gt_patches.unsqueeze(2)) * pixel_mask_detached.unsqueeze(2)
        ).sum(dim=-1) / mask_count.unsqueeze(-1)
        coarse_err = candidate_err[:, :, 0]
        best_noncoarse_err, best_noncoarse_idx = candidate_err[:, :, 1:].min(dim=-1)
        choose_noncoarse = (best_noncoarse_err + margin) < coarse_err
        target_idx = torch.zeros_like(best_noncoarse_idx)
        target_idx[choose_noncoarse] = best_noncoarse_idx[choose_noncoarse] + 1
        selector_advantage = coarse_err - best_noncoarse_err

    logits_valid = logits[valid_patches]
    target_valid = target_idx[valid_patches]
    selector_loss = F.cross_entropy(logits_valid.float(), target_valid, reduction="mean")

    pred_idx = logits_valid.argmax(dim=-1)
    metrics = {
        "selector_accuracy": (pred_idx == target_valid).float().mean().item(),
        "selector_pred_noncoarse_rate": (pred_idx > 0).float().mean().item(),
        "selector_target_noncoarse_rate": (target_valid > 0).float().mean().item(),
        "selector_advantage_mean": selector_advantage[valid_patches].mean().item(),
    }
    return selector_loss, metrics


def compute_head_oracle_loss(model, target, margin: float):
    """Train patch heads to offer at least one better-than-coarse hypothesis."""
    generator = getattr(model, "generator", None)
    candidate_bank = getattr(generator, "last_candidate_bank", None)
    base = getattr(generator, "last_base_patches_flat", None)
    pixel_mask = getattr(generator, "last_pixel_mask_flat", None)
    if generator is None or candidate_bank is None or base is None or pixel_mask is None or candidate_bank.size(2) <= 1:
        zero = target.new_zeros(())
        empty_metrics = {
            "head_oracle_success_rate": None,
            "head_oracle_gap": None,
            "head_oracle_margin_violations": None,
        }
        return zero, empty_metrics

    with torch.no_grad():
        gt_patches, _ = generator.unfold_native(target, generator.kernel_size)
        gt_patches = gt_patches.flatten(start_dim=2).transpose(1, 2)

        pixel_mask_detached = pixel_mask.detach()
        valid_patches = pixel_mask_detached.sum(dim=-1) > 0
        if not valid_patches.any():
            zero = target.new_zeros(())
            empty_metrics = {
                "head_oracle_success_rate": None,
                "head_oracle_gap": None,
                "head_oracle_margin_violations": None,
            }
            return zero, empty_metrics

    mask_count = pixel_mask.sum(dim=-1).clamp_min(1.0)
    coarse_err = (torch.abs(base.detach() - gt_patches) * pixel_mask).sum(dim=-1) / mask_count
    head_candidates = candidate_bank[:, :, 1:, :]
    head_err = (torch.abs(head_candidates - gt_patches.unsqueeze(2)) * pixel_mask.unsqueeze(2)).sum(dim=-1) / mask_count.unsqueeze(-1)
    best_head_err, _ = head_err.min(dim=-1)
    oracle_hinge = F.relu(best_head_err - coarse_err + margin)
    oracle_loss = oracle_hinge[valid_patches].mean()
    metrics = {
        "head_oracle_success_rate": ((best_head_err + margin) < coarse_err)[valid_patches].float().mean().item(),
        "head_oracle_gap": (coarse_err - best_head_err)[valid_patches].mean().item(),
        "head_oracle_margin_violations": (oracle_hinge[valid_patches] > 0).float().mean().item(),
    }
    return oracle_loss, metrics


def compute_oracle_patch_distill_loss(model, target, margin: float):
    """Teach the fused patch output to match the best available retrieved patch hypothesis."""
    generator = getattr(model, "generator", None)
    candidate_bank = getattr(generator, "last_candidate_bank", None)
    output_patches = getattr(generator, "last_output_patches_flat", None)
    pixel_mask = getattr(generator, "last_pixel_mask_flat", None)
    if generator is None or candidate_bank is None or output_patches is None or pixel_mask is None:
        zero = target.new_zeros(())
        empty_metrics = {
            "oracle_patch_noncoarse_rate": None,
            "oracle_patch_distill_gap": None,
            "oracle_patch_alignment": None,
        }
        return zero, empty_metrics

    with torch.no_grad():
        gt_patches, _ = generator.unfold_native(target, generator.kernel_size)
        gt_patches = gt_patches.flatten(start_dim=2).transpose(1, 2)

        pixel_mask_detached = pixel_mask.detach()
        valid_patches = pixel_mask_detached.sum(dim=-1) > 0
        if not valid_patches.any():
            zero = target.new_zeros(())
            empty_metrics = {
                "oracle_patch_noncoarse_rate": None,
                "oracle_patch_distill_gap": None,
                "oracle_patch_alignment": None,
            }
            return zero, empty_metrics

        mask_count = pixel_mask_detached.sum(dim=-1).clamp_min(1.0)
        candidate_err = (
            torch.abs(candidate_bank.detach() - gt_patches.unsqueeze(2)) * pixel_mask_detached.unsqueeze(2)
        ).sum(dim=-1) / mask_count.unsqueeze(-1)
        coarse_err = candidate_err[:, :, 0]
        best_err, best_idx = candidate_err.min(dim=-1)
        best_noncoarse_err, best_noncoarse_idx = candidate_err[:, :, 1:].min(dim=-1)
        use_noncoarse = (best_noncoarse_err + margin) < coarse_err
        best_idx = torch.where(use_noncoarse, best_noncoarse_idx + 1, torch.zeros_like(best_idx))
        oracle_target = torch.gather(
            candidate_bank.detach(),
            2,
            best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, candidate_bank.size(-1)),
        ).squeeze(2)
        oracle_advantage = coarse_err - best_noncoarse_err

    distill_l1 = (torch.abs(output_patches - oracle_target) * pixel_mask).sum(dim=-1) / pixel_mask.sum(dim=-1).clamp_min(1.0)
    oracle_patch_loss = distill_l1[valid_patches].mean()

    output_alignment = (
        (torch.abs(output_patches.detach() - oracle_target) * pixel_mask).sum(dim=-1) / pixel_mask.sum(dim=-1).clamp_min(1.0)
    )
    metrics = {
        "oracle_patch_noncoarse_rate": use_noncoarse[valid_patches].float().mean().item(),
        "oracle_patch_distill_gap": oracle_advantage[valid_patches].mean().item(),
        "oracle_patch_alignment": output_alignment[valid_patches].mean().item(),
    }
    return oracle_patch_loss, metrics


def compute_train_loss(criterion, model, coarse, refined, target, mask, train_mode: str):
    """Compute loss according to the current staged-training mode."""
    l1_coarse = criterion.masked_l1(coarse, target, mask)
    l1_refined = criterion.masked_l1(refined, target, mask)

    zero = refined.new_zeros(())
    perceptual = zero
    style = zero
    gate_selective = zero
    sample_improvement = zero
    selector_choice = zero
    head_oracle = zero
    coarse_anchor = zero
    oracle_patch = zero
    gate_metrics = {
        "gate_target_rate": None,
        "gate_pred_mean": None,
        "gate_accuracy": None,
        "gate_advantage_mean": None,
    }
    improvement_metrics = {
        "sample_better_rate_batch": None,
        "sample_margin_violations": None,
        "sample_improvement_gap": None,
    }
    selector_metrics = {
        "selector_accuracy": None,
        "selector_pred_noncoarse_rate": None,
        "selector_target_noncoarse_rate": None,
        "selector_advantage_mean": None,
    }
    head_oracle_metrics = {
        "head_oracle_success_rate": None,
        "head_oracle_gap": None,
        "head_oracle_margin_violations": None,
    }
    oracle_patch_metrics = {
        "oracle_patch_noncoarse_rate": None,
        "oracle_patch_distill_gap": None,
        "oracle_patch_alignment": None,
    }

    if train_mode in ("joint", "refine_only", "refined_loss_only") and criterion.perceptual_weight > 0:
        perceptual = criterion.perceptual_loss(refined, target)
    if train_mode in ("joint", "refine_only", "refined_loss_only") and criterion.style_weight > 0:
        style = criterion.style_loss(refined, target)
    if train_mode in ("joint", "refine_only", "refined_loss_only") and criterion.gate_selective_weight > 0:
        gate_selective, gate_metrics = compute_selective_gate_loss(
            model,
            target,
            criterion.gate_selective_weight,
            criterion.gate_selective_margin,
        )
    if train_mode in ("joint", "refine_only", "refined_loss_only") and criterion.sample_improvement_weight > 0:
        sample_improvement, improvement_metrics = compute_sample_improvement_loss(
            coarse,
            refined,
            target,
            mask,
            criterion.sample_improvement_margin,
        )
    if train_mode in ("joint", "refine_only", "refined_loss_only") and criterion.selector_choice_weight > 0:
        selector_choice, selector_metrics = compute_selector_choice_loss(
            model,
            target,
            criterion.selector_choice_margin,
        )
    if train_mode in ("joint", "refine_only", "refined_loss_only") and criterion.head_oracle_weight > 0:
        head_oracle, head_oracle_metrics = compute_head_oracle_loss(
            model,
            target,
            criterion.head_oracle_margin,
        )
    if train_mode in ("joint", "refine_only", "refined_loss_only") and criterion.coarse_anchor_weight > 0:
        coarse_anchor = l1_coarse
    if train_mode in ("joint", "refine_only", "refined_loss_only") and criterion.oracle_patch_weight > 0:
        oracle_patch, oracle_patch_metrics = compute_oracle_patch_distill_loss(
            model,
            target,
            criterion.oracle_patch_margin,
        )

    if train_mode == "joint":
        total = (
            l1_coarse
            + l1_refined
            + criterion.perceptual_weight * perceptual
            + criterion.style_weight * style
            + criterion.gate_selective_weight * gate_selective
            + criterion.sample_improvement_weight * sample_improvement
            + criterion.selector_choice_weight * selector_choice
            + criterion.head_oracle_weight * head_oracle
            + criterion.coarse_anchor_weight * coarse_anchor
            + criterion.oracle_patch_weight * oracle_patch
        )
    elif train_mode == "refined_loss_only":
        total = (
            l1_refined
            + criterion.perceptual_weight * perceptual
            + criterion.style_weight * style
            + criterion.gate_selective_weight * gate_selective
            + criterion.sample_improvement_weight * sample_improvement
            + criterion.selector_choice_weight * selector_choice
            + criterion.head_oracle_weight * head_oracle
            + criterion.coarse_anchor_weight * coarse_anchor
            + criterion.oracle_patch_weight * oracle_patch
        )
    elif train_mode == "coarse_only":
        total = l1_coarse
    elif train_mode == "refine_only":
        total = (
            l1_refined
            + criterion.perceptual_weight * perceptual
            + criterion.style_weight * style
            + criterion.gate_selective_weight * gate_selective
            + criterion.sample_improvement_weight * sample_improvement
            + criterion.selector_choice_weight * selector_choice
            + criterion.head_oracle_weight * head_oracle
            + criterion.coarse_anchor_weight * coarse_anchor
            + criterion.oracle_patch_weight * oracle_patch
        )
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    loss_dict = {
        "l1_coarse": l1_coarse.item(),
        "l1_refined": l1_refined.item(),
        "perceptual": perceptual.item(),
        "style": style.item(),
        "gate_selective": gate_selective.item(),
        "sample_improvement": sample_improvement.item(),
        "selector_choice": selector_choice.item(),
        "head_oracle": head_oracle.item(),
        "coarse_anchor": coarse_anchor.item(),
        "oracle_patch": oracle_patch.item(),
        "total": total.item(),
    }
    aux_metrics = {
        **gate_metrics,
        **improvement_metrics,
        **selector_metrics,
        **head_oracle_metrics,
        **oracle_patch_metrics,
    }
    return total, loss_dict, aux_metrics


def write_status(log_dir, step, total_steps, loss_dict, running_loss, lr, start_time, extra=None):
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
    train_mode="joint",
    freeze_coarse=False,
    gate_selective_margin=0.0,
    selector_choice_margin=0.0,
    head_oracle_margin=0.0,
):
    """Quick validation focused on whether refinement helps or hurts."""
    model.eval()

    stats = {
        "masked_l1_coarse": [],
        "masked_l1_refined": [],
        "valid_l1_coarse": [],
        "valid_l1_refined": [],
        "raw_valid_l1_coarse": [],
        "raw_valid_l1_refined": [],
        "refined_better_flags": [],
        "masked_delta_mean": [],
        "gate_target_rate": [],
        "gate_pred_mean": [],
        "gate_accuracy": [],
        "gate_advantage_mean": [],
        "selector_accuracy": [],
        "selector_pred_noncoarse_rate": [],
        "selector_target_noncoarse_rate": [],
        "selector_advantage_mean": [],
        "head_oracle_success_rate": [],
        "head_oracle_gap": [],
        "head_oracle_margin_violations": [],
    }
    gain_abs_values = []
    gain_pct_values = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
            break

        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        masked_image = batch["masked_image"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            refined_raw, _, coarse_raw = model(masked_image, mask)

        refined_raw = refined_raw.clamp(0, 1)
        coarse_raw = coarse_raw.clamp(0, 1)
        refined = composite_with_known(refined_raw, image, mask)
        coarse = composite_with_known(coarse_raw, image, mask)
        valid_mask = 1 - mask

        hole_denom = (mask.sum(dim=(1, 2, 3)) * image.shape[1]).clamp_min(1e-8)
        valid_denom = (valid_mask.sum(dim=(1, 2, 3)) * image.shape[1]).clamp_min(1e-8)

        coarse_hole = (torch.abs(coarse - image) * mask).sum(dim=(1, 2, 3)) / hole_denom
        refined_hole = (torch.abs(refined - image) * mask).sum(dim=(1, 2, 3)) / hole_denom
        coarse_valid = (torch.abs(coarse - image) * valid_mask).sum(dim=(1, 2, 3)) / valid_denom
        refined_valid = (torch.abs(refined - image) * valid_mask).sum(dim=(1, 2, 3)) / valid_denom
        raw_coarse_valid = (torch.abs(coarse_raw - image) * valid_mask).sum(dim=(1, 2, 3)) / valid_denom
        raw_refined_valid = (torch.abs(refined_raw - image) * valid_mask).sum(dim=(1, 2, 3)) / valid_denom
        masked_delta = (torch.abs(refined - coarse) * mask).sum(dim=(1, 2, 3)) / hole_denom
        gain_abs = coarse_hole - refined_hole
        gain_pct = gain_abs / coarse_hole.clamp_min(1e-8) * 100.0

        stats["masked_l1_coarse"].extend(coarse_hole.detach().cpu().tolist())
        stats["masked_l1_refined"].extend(refined_hole.detach().cpu().tolist())
        stats["valid_l1_coarse"].extend(coarse_valid.detach().cpu().tolist())
        stats["valid_l1_refined"].extend(refined_valid.detach().cpu().tolist())
        stats["raw_valid_l1_coarse"].extend(raw_coarse_valid.detach().cpu().tolist())
        stats["raw_valid_l1_refined"].extend(raw_refined_valid.detach().cpu().tolist())
        stats["masked_delta_mean"].extend(masked_delta.detach().cpu().tolist())
        stats["refined_better_flags"].extend((refined_hole < coarse_hole).detach().cpu().float().tolist())
        gain_abs_values.extend(gain_abs.detach().cpu().tolist())
        gain_pct_values.extend(gain_pct.detach().cpu().tolist())

        gate_stats = compute_selective_gate_targets(model, image, margin=gate_selective_margin)
        if gate_stats is not None:
            valid_patches = gate_stats["valid_patches"]
            gate_pred = gate_stats["gate_pred"][valid_patches]
            gate_target = gate_stats["gate_target"][valid_patches]
            gate_advantage = gate_stats["advantage"][valid_patches]
            gate_binary = (gate_pred >= 0.5).float()
            stats["gate_target_rate"].append(gate_target.mean().item())
            stats["gate_pred_mean"].append(gate_pred.mean().item())
            stats["gate_accuracy"].append((gate_binary == gate_target).float().mean().item())
            stats["gate_advantage_mean"].append(gate_advantage.mean().item())

        _, selector_metrics = compute_selector_choice_loss(model, image, margin=selector_choice_margin)
        if selector_metrics["selector_accuracy"] is not None:
            stats["selector_accuracy"].append(selector_metrics["selector_accuracy"])
            stats["selector_pred_noncoarse_rate"].append(selector_metrics["selector_pred_noncoarse_rate"])
            stats["selector_target_noncoarse_rate"].append(selector_metrics["selector_target_noncoarse_rate"])
            stats["selector_advantage_mean"].append(selector_metrics["selector_advantage_mean"])
        _, head_oracle_metrics = compute_head_oracle_loss(model, image, margin=head_oracle_margin)
        if head_oracle_metrics["head_oracle_success_rate"] is not None:
            stats["head_oracle_success_rate"].append(head_oracle_metrics["head_oracle_success_rate"])
            stats["head_oracle_gap"].append(head_oracle_metrics["head_oracle_gap"])
            stats["head_oracle_margin_violations"].append(head_oracle_metrics["head_oracle_margin_violations"])

    result = {}
    for key, values in stats.items():
        result[key] = float(sum(values) / len(values)) if values else None

    coarse_masked = result["masked_l1_coarse"]
    refined_masked = result["masked_l1_refined"]
    if coarse_masked is not None and refined_masked is not None and coarse_masked > 0:
        result["refinement_gain_pct"] = float((coarse_masked - refined_masked) / coarse_masked * 100.0)
    else:
        result["refinement_gain_pct"] = None

    better_rate = result["refined_better_flags"]
    result["refined_better_rate"] = better_rate
    result["refined_worse_rate"] = float(sum(1.0 for v in gain_abs_values if v < 0) / len(gain_abs_values)) if gain_abs_values else None
    add_distribution_metrics(result, gain_abs_values, "gain_abs")
    add_distribution_metrics(result, gain_pct_values, "gain_pct")

    warnings = []
    if result["refinement_gain_pct"] is not None and result["refinement_gain_pct"] < -2.0:
        warnings.append("refinement_worse_than_coarse")
    if better_rate is not None and better_rate < 0.45:
        warnings.append("refinement_rarely_beats_coarse")
    if (
        result["raw_valid_l1_coarse"] is not None and
        result["raw_valid_l1_refined"] is not None and
        result["raw_valid_l1_refined"] > result["raw_valid_l1_coarse"] * 1.1
    ):
        warnings.append("raw_refinement_hurts_valid_region")
    result["warnings"] = warnings

    model.train()
    set_parameter_trainability(model, train_mode, freeze_coarse=freeze_coarse)
    return result


def run_eval_only(cfg, args):
    """Run validation only on a checkpoint without training."""
    if not args.resume:
        raise ValueError("--eval-only requires --resume CHECKPOINT")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    torch.manual_seed(cfg["training"]["seed"])

    model_config = build_model_config(cfg)
    model = InpaintingModel(model_config).to(device)
    train_mode = cfg["training"].get("train_mode", "joint")
    freeze_coarse = cfg["training"].get("freeze_coarse", False)
    set_parameter_trainability(model, train_mode, freeze_coarse=freeze_coarse)

    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.resume}")

    criterion = InpaintingLoss(**cfg["loss"]).to(device)

    val_dir = cfg["data"].get("val_dir")
    manifest_path = cfg["data"].get("manifest_path")
    eval_loader = get_dataloader(
        root_dir=cfg["data"]["root_dir"],
        image_size=cfg["data"]["image_size"],
        split="val",
        batch_size=cfg["data"].get("eval_batch_size", cfg["data"]["batch_size"]),
        num_workers=max(1, min(2, cfg["data"]["num_workers"])),
        max_images=None,
        mask_min_coverage=cfg["data"]["mask_min_coverage"],
        mask_max_coverage=cfg["data"]["mask_max_coverage"],
        val_dir=val_dir,
        manifest_path=manifest_path,
    )
    print(f"Validation images: {len(eval_loader.dataset)}")

    if args.eval_batches is not None:
        max_batches = args.eval_batches if args.eval_batches > 0 else None
    else:
        max_batches = cfg.get("logging", {}).get("eval_batches", 8)
        if max_batches is not None and max_batches <= 0:
            max_batches = None

    health = evaluate_refinement_health(
        model,
        eval_loader,
        device,
        cfg["training"]["mixed_precision"],
        max_batches=max_batches,
        train_mode=train_mode,
        freeze_coarse=freeze_coarse,
        gate_selective_margin=criterion.gate_selective_margin,
        selector_choice_margin=criterion.selector_choice_margin,
        head_oracle_margin=criterion.head_oracle_margin,
    )

    print("\nFull validation results:")
    print(json.dumps(health, indent=2))

    log_dir = Path(cfg["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_path = log_dir / "eval_only_full_val.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2)
    print(f"Saved eval summary: {eval_path}")


def log_validation(writer, log_dir, step, total_steps, loss_dict, running_loss, lr, start_time, health):
    """Persist validation metrics and print a compact summary."""
    writer.add_scalar("val/masked_l1_coarse", health["masked_l1_coarse"], step)
    writer.add_scalar("val/masked_l1_refined", health["masked_l1_refined"], step)
    writer.add_scalar("val/valid_l1_coarse", health["valid_l1_coarse"], step)
    writer.add_scalar("val/valid_l1_refined", health["valid_l1_refined"], step)
    writer.add_scalar("val/raw_valid_l1_coarse", health["raw_valid_l1_coarse"], step)
    writer.add_scalar("val/raw_valid_l1_refined", health["raw_valid_l1_refined"], step)
    writer.add_scalar("val/refinement_gain_pct", health["refinement_gain_pct"], step)
    writer.add_scalar("val/refined_better_rate", health["refined_better_rate"], step)
    writer.add_scalar("val/refined_worse_rate", health["refined_worse_rate"], step)
    writer.add_scalar("val/masked_delta_mean", health["masked_delta_mean"], step)
    if health.get("gain_abs_p50") is not None:
        writer.add_scalar("val/gain_abs_p25", health["gain_abs_p25"], step)
        writer.add_scalar("val/gain_abs_p50", health["gain_abs_p50"], step)
        writer.add_scalar("val/gain_abs_p75", health["gain_abs_p75"], step)
    if health.get("gain_pct_p50") is not None:
        writer.add_scalar("val/gain_pct_p25", health["gain_pct_p25"], step)
        writer.add_scalar("val/gain_pct_p50", health["gain_pct_p50"], step)
        writer.add_scalar("val/gain_pct_p75", health["gain_pct_p75"], step)
    if health.get("gate_target_rate") is not None:
        writer.add_scalar("val/gate_target_rate", health["gate_target_rate"], step)
    if health.get("gate_pred_mean") is not None:
        writer.add_scalar("val/gate_pred_mean", health["gate_pred_mean"], step)
    if health.get("gate_accuracy") is not None:
        writer.add_scalar("val/gate_accuracy", health["gate_accuracy"], step)
    if health.get("gate_advantage_mean") is not None:
        writer.add_scalar("val/gate_advantage_mean", health["gate_advantage_mean"], step)
    if health.get("selector_accuracy") is not None:
        writer.add_scalar("val/selector_accuracy", health["selector_accuracy"], step)
    if health.get("selector_pred_noncoarse_rate") is not None:
        writer.add_scalar("val/selector_pred_noncoarse_rate", health["selector_pred_noncoarse_rate"], step)
    if health.get("selector_target_noncoarse_rate") is not None:
        writer.add_scalar("val/selector_target_noncoarse_rate", health["selector_target_noncoarse_rate"], step)
    if health.get("selector_advantage_mean") is not None:
        writer.add_scalar("val/selector_advantage_mean", health["selector_advantage_mean"], step)
    if health.get("head_oracle_success_rate") is not None:
        writer.add_scalar("val/head_oracle_success_rate", health["head_oracle_success_rate"], step)
    if health.get("head_oracle_gap") is not None:
        writer.add_scalar("val/head_oracle_gap", health["head_oracle_gap"], step)
    if health.get("head_oracle_margin_violations") is not None:
        writer.add_scalar("val/head_oracle_margin_violations", health["head_oracle_margin_violations"], step)
    write_status(
        log_dir,
        step,
        total_steps,
        loss_dict,
        running_loss,
        lr,
        start_time,
        extra={"validation": health},
    )
    append_validation_history(log_dir, step, health)

    gate_suffix = ""
    if health.get("gate_accuracy") is not None:
        gate_suffix = (
            f", gate_acc={health['gate_accuracy']:.2f}"
            f", gate_on={health['gate_pred_mean']:.2f}"
            f", gate_target={health['gate_target_rate']:.2f}"
        )
    selector_suffix = ""
    if health.get("selector_accuracy") is not None:
        selector_suffix = (
            f", sel_acc={health['selector_accuracy']:.2f}"
            f", sel_on={health['selector_pred_noncoarse_rate']:.2f}"
            f", sel_target={health['selector_target_noncoarse_rate']:.2f}"
        )
    oracle_suffix = ""
    if health.get("head_oracle_success_rate") is not None:
        oracle_suffix = (
            f", oracle_ok={health['head_oracle_success_rate']:.2f}"
            f", oracle_gap={health['head_oracle_gap']:.4f}"
        )
    warning_suffix = f" warnings={','.join(health['warnings'])}" if health["warnings"] else ""
    print(
        f"\nValidation step {step}: "
        f"masked L1 coarse={health['masked_l1_coarse']:.4f}, "
        f"refined={health['masked_l1_refined']:.4f}, "
        f"gain={health['refinement_gain_pct']:.2f}%, "
        f"better_rate={health['refined_better_rate']:.2f}, "
        f"worse_rate={health['refined_worse_rate']:.2f}, "
        f"valid coarse={health['valid_l1_coarse']:.4f}, "
        f"valid refined={health['valid_l1_refined']:.4f}, "
        f"raw valid refined={health['raw_valid_l1_refined']:.4f}, "
        f"gain_p50={health['gain_pct_p50']:.2f}% "
        f"(p25={health['gain_pct_p25']:.2f}%, p75={health['gain_pct_p75']:.2f}%){gate_suffix}{selector_suffix}{oracle_suffix}{warning_suffix}"
    )


def train(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Seed
    torch.manual_seed(cfg["training"]["seed"])

    # Model
    model_config = build_model_config(cfg)
    model = InpaintingModel(model_config).to(device)
    train_mode = cfg["training"].get("train_mode", "joint")
    freeze_coarse = cfg["training"].get("freeze_coarse", False)
    set_parameter_trainability(model, train_mode, freeze_coarse=freeze_coarse)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Train mode: {train_mode}")
    if train_mode == "refine_only" or freeze_coarse:
        frozen_params = sum(p.numel() for p in model.coarse_model.parameters())
        print(f"Frozen coarse parameters: {frozen_params:,}")

    # Loss
    criterion = InpaintingLoss(**cfg["loss"]).to(device)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=cfg["training"]["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["mixed_precision"])

    # Data
    max_images = args.overfit if args.overfit else None
    val_dir = cfg["data"].get("val_dir")
    manifest_path = cfg["data"].get("manifest_path")
    train_loader = get_dataloader(
        root_dir=cfg["data"]["root_dir"],
        image_size=cfg["data"]["image_size"],
        split="train",
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        max_images=max_images,
        mask_min_coverage=cfg["data"]["mask_min_coverage"],
        mask_max_coverage=cfg["data"]["mask_max_coverage"],
        val_dir=val_dir,
        manifest_path=manifest_path,
    )
    print(f"Training images: {len(train_loader.dataset)}")

    eval_loader = None
    eval_batches = cfg["logging"].get("eval_batches", 8)
    if log_cfg := cfg.get("logging"):
        pass
    if cfg["logging"].get("eval_interval", 0):
        eval_loader = get_dataloader(
            root_dir=cfg["data"]["root_dir"],
            image_size=cfg["data"]["image_size"],
            split="val",
            batch_size=cfg["data"].get("eval_batch_size", cfg["data"]["batch_size"]),
            num_workers=max(1, min(2, cfg["data"]["num_workers"])),
            max_images=max_images,
            mask_min_coverage=cfg["data"]["mask_min_coverage"],
            mask_max_coverage=cfg["data"]["mask_max_coverage"],
            val_dir=val_dir,
            manifest_path=manifest_path,
        )
        print(f"Validation images: {len(eval_loader.dataset)}")

    # Training params
    total_steps = args.steps if args.steps else cfg["training"]["total_steps"]
    grad_accum = 1 if args.overfit else cfg["training"]["grad_accum_steps"]
    max_lr = cfg["training"]["lr"]
    min_lr = cfg["training"]["min_lr"]
    warmup_steps = cfg["training"]["warmup_steps"]
    grad_clip = cfg["training"]["grad_clip"]
    use_amp = cfg["training"]["mixed_precision"]

    # Logging
    log_cfg = cfg["logging"]
    eval_interval = log_cfg.get("eval_interval", 0)
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
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    set_parameter_trainability(model, train_mode, freeze_coarse=freeze_coarse)
    data_iter = iter(train_loader)
    optimizer.zero_grad()

    pbar = tqdm(range(start_step, total_steps), desc="Training", dynamic_ncols=True)
    running_loss = 0.0
    train_start_time = time.time()

    for step in pbar:
        # Get batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        masked_image = batch["masked_image"].to(device, non_blocking=True)

        # Forward
        with torch.amp.autocast("cuda", enabled=use_amp):
            refined_raw, attn, coarse_raw = model(masked_image, mask)
            coarse = composite_with_known(coarse_raw, image, mask)
            refined = composite_with_known(refined_raw, image, mask)
            loss, loss_dict, gate_metrics = compute_train_loss(
                criterion,
                model,
                coarse,
                refined,
                image,
                mask,
                train_mode=train_mode,
            )
            loss = loss / grad_accum

        # Backward
        scaler.scale(loss).backward()

        # Optimizer step (every grad_accum steps)
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update LR
            opt_step = (step + 1) // grad_accum
            lr = get_lr(opt_step, warmup_steps, total_steps // grad_accum, max_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        running_loss = 0.9 * running_loss + 0.1 * loss_dict["total"]
        pbar.set_postfix(loss=f"{running_loss:.4f}", l1r=f"{loss_dict['l1_refined']:.4f}")

        # Logging
        if step % log_cfg["log_interval"] == 0:
            writer.add_scalar("loss/total", loss_dict["total"], step)
            writer.add_scalar("loss/l1_coarse", loss_dict["l1_coarse"], step)
            writer.add_scalar("loss/l1_refined", loss_dict["l1_refined"], step)
            writer.add_scalar("loss/perceptual", loss_dict["perceptual"], step)
            writer.add_scalar("loss/style", loss_dict["style"], step)
            writer.add_scalar("loss/gate_selective", loss_dict["gate_selective"], step)
            writer.add_scalar("loss/sample_improvement", loss_dict["sample_improvement"], step)
            writer.add_scalar("loss/selector_choice", loss_dict["selector_choice"], step)
            writer.add_scalar("loss/head_oracle", loss_dict["head_oracle"], step)
            writer.add_scalar("loss/coarse_anchor", loss_dict["coarse_anchor"], step)
            writer.add_scalar("loss/oracle_patch", loss_dict["oracle_patch"], step)
            if gate_metrics.get("gate_target_rate") is not None:
                writer.add_scalar("train/gate_target_rate", gate_metrics["gate_target_rate"], step)
                writer.add_scalar("train/gate_pred_mean", gate_metrics["gate_pred_mean"], step)
                writer.add_scalar("train/gate_accuracy", gate_metrics["gate_accuracy"], step)
                writer.add_scalar("train/gate_advantage_mean", gate_metrics["gate_advantage_mean"], step)
            if gate_metrics.get("sample_better_rate_batch") is not None:
                writer.add_scalar("train/sample_better_rate_batch", gate_metrics["sample_better_rate_batch"], step)
                writer.add_scalar("train/sample_margin_violations", gate_metrics["sample_margin_violations"], step)
                writer.add_scalar("train/sample_improvement_gap", gate_metrics["sample_improvement_gap"], step)
            if gate_metrics.get("selector_accuracy") is not None:
                writer.add_scalar("train/selector_accuracy", gate_metrics["selector_accuracy"], step)
                writer.add_scalar("train/selector_pred_noncoarse_rate", gate_metrics["selector_pred_noncoarse_rate"], step)
                writer.add_scalar("train/selector_target_noncoarse_rate", gate_metrics["selector_target_noncoarse_rate"], step)
                writer.add_scalar("train/selector_advantage_mean", gate_metrics["selector_advantage_mean"], step)
            if gate_metrics.get("head_oracle_success_rate") is not None:
                writer.add_scalar("train/head_oracle_success_rate", gate_metrics["head_oracle_success_rate"], step)
                writer.add_scalar("train/head_oracle_gap", gate_metrics["head_oracle_gap"], step)
                writer.add_scalar("train/head_oracle_margin_violations", gate_metrics["head_oracle_margin_violations"], step)
            if gate_metrics.get("oracle_patch_noncoarse_rate") is not None:
                writer.add_scalar("train/oracle_patch_noncoarse_rate", gate_metrics["oracle_patch_noncoarse_rate"], step)
                writer.add_scalar("train/oracle_patch_distill_gap", gate_metrics["oracle_patch_distill_gap"], step)
                writer.add_scalar("train/oracle_patch_alignment", gate_metrics["oracle_patch_alignment"], step)
            if hasattr(model.generator, "refinement_gate"):
                writer.add_scalar("model/refinement_gate", torch.sigmoid(model.generator.refinement_gate).item(), step)
            opt_step = (step + 1) // grad_accum
            lr = get_lr(opt_step, warmup_steps, total_steps // grad_accum, max_lr, min_lr)
            writer.add_scalar("lr", lr, step)

            if device == "cuda":
                writer.add_scalar("vram_gb", torch.cuda.max_memory_allocated() / 1e9, step)

            # Write status file for remote monitoring
            write_status(log_cfg["log_dir"], step, total_steps, loss_dict, running_loss, lr, train_start_time)

        if eval_loader is not None and step > 0 and step % eval_interval == 0:
            health = evaluate_refinement_health(
                model,
                eval_loader,
                device,
                use_amp,
                max_batches=eval_batches,
                train_mode=train_mode,
                freeze_coarse=freeze_coarse,
                gate_selective_margin=criterion.gate_selective_margin,
                selector_choice_margin=criterion.selector_choice_margin,
                head_oracle_margin=criterion.head_oracle_margin,
            )
            log_validation(
                writer,
                log_cfg["log_dir"],
                step,
                total_steps,
                loss_dict,
                running_loss,
                lr,
                train_start_time,
                health,
            )

        # Visualization
        if step % log_cfg["vis_interval"] == 0:
            save_vis(writer, batch, coarse, refined, step, log_dir=log_cfg["log_dir"])

        # Checkpoint at specific steps
        checkpoint_steps = log_cfg.get("checkpoint_steps", None)
        save_checkpoints = log_cfg.get("save_checkpoints", True)
        should_save = False
        if save_checkpoints and checkpoint_steps and step in checkpoint_steps:
            should_save = True
        elif save_checkpoints and not checkpoint_steps and step > 0 and step % log_cfg["save_interval"] == 0:
            should_save = True

        if should_save:
            ckpt_path = ckpt_dir / f"step_{step}.pth"
            if save_checkpoint(model, optimizer, scaler, step, loss_dict, cfg, ckpt_path):
                print(f"\nSaved checkpoint: {ckpt_path}")
                opt_step_ckpt = (step + 1) // grad_accum
                lr_ckpt = get_lr(opt_step_ckpt, warmup_steps, total_steps // grad_accum, max_lr, min_lr)
                write_status(log_cfg["log_dir"], step, total_steps, loss_dict, running_loss, lr_ckpt, train_start_time,
                             extra={"event": "checkpoint", "checkpoint_path": str(ckpt_path)})

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
    final_lr = get_lr(max((total_steps + grad_accum - 1) // grad_accum, 1), warmup_steps, total_steps // grad_accum, max_lr, min_lr)
    if eval_loader is not None:
        final_health = evaluate_refinement_health(
            model,
            eval_loader,
            device,
            use_amp,
            max_batches=eval_batches,
            train_mode=train_mode,
            freeze_coarse=freeze_coarse,
            gate_selective_margin=criterion.gate_selective_margin,
            selector_choice_margin=criterion.selector_choice_margin,
            head_oracle_margin=criterion.head_oracle_margin,
        )
        log_validation(
            writer,
            log_cfg["log_dir"],
            total_steps,
            total_steps,
            loss_dict,
            running_loss,
            final_lr,
            train_start_time,
            final_health,
        )
    if save_final_checkpoint:
        print(f"\nTraining complete. Final checkpoint: {final_path}")
    else:
        print("\nTraining complete. Final checkpoint saving disabled for this run.")
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="RETHINED Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--overfit", type=int, default=None, help="Overfit on N images (sanity check)")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--eval-only", action="store_true", help="Run validation only on a checkpoint")
    parser.add_argument("--eval-batches", type=int, default=None, help="Override eval batches; use 0 for full validation set")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.eval_only:
        run_eval_only(cfg, args)
    else:
        train(cfg, args)


if __name__ == "__main__":
    main()
