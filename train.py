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
        if train_mode == "joint":
            param.requires_grad = not (freeze_coarse and is_coarse)
        elif train_mode == "coarse_only":
            param.requires_grad = is_coarse
        elif train_mode == "refine_only":
            param.requires_grad = not is_coarse
        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    if train_mode == "refine_only" or freeze_coarse:
        model.coarse_model.eval()


def compute_train_loss(criterion, coarse, refined, target, mask, train_mode: str):
    """Compute loss according to the current staged-training mode."""
    l1_coarse = criterion.masked_l1(coarse, target, mask)
    l1_refined = criterion.masked_l1(refined, target, mask)

    zero = refined.new_zeros(())
    perceptual = zero
    style = zero

    if train_mode in ("joint", "refine_only") and criterion.perceptual_weight > 0:
        perceptual = criterion.perceptual_loss(refined, target)
    if train_mode in ("joint", "refine_only") and criterion.style_weight > 0:
        style = criterion.style_loss(refined, target)

    if train_mode == "joint":
        total = (
            l1_coarse
            + l1_refined
            + criterion.perceptual_weight * perceptual
            + criterion.style_weight * style
        )
    elif train_mode == "coarse_only":
        total = l1_coarse
    elif train_mode == "refine_only":
        total = (
            l1_refined
            + criterion.perceptual_weight * perceptual
            + criterion.style_weight * style
        )
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    loss_dict = {
        "l1_coarse": l1_coarse.item(),
        "l1_refined": l1_refined.item(),
        "perceptual": perceptual.item(),
        "style": style.item(),
        "total": total.item(),
    }
    return total, loss_dict


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
def evaluate_refinement_health(model, dataloader, device, use_amp, max_batches=8, train_mode="joint", freeze_coarse=False):
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
    }

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
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

        stats["masked_l1_coarse"].extend(coarse_hole.detach().cpu().tolist())
        stats["masked_l1_refined"].extend(refined_hole.detach().cpu().tolist())
        stats["valid_l1_coarse"].extend(coarse_valid.detach().cpu().tolist())
        stats["valid_l1_refined"].extend(refined_valid.detach().cpu().tolist())
        stats["raw_valid_l1_coarse"].extend(raw_coarse_valid.detach().cpu().tolist())
        stats["raw_valid_l1_refined"].extend(raw_refined_valid.detach().cpu().tolist())
        stats["masked_delta_mean"].extend(masked_delta.detach().cpu().tolist())
        stats["refined_better_flags"].extend((refined_hole < coarse_hole).detach().cpu().float().tolist())

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
    writer.add_scalar("val/masked_delta_mean", health["masked_delta_mean"], step)
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

    warning_suffix = f" warnings={','.join(health['warnings'])}" if health["warnings"] else ""
    print(
        f"\nValidation step {step}: "
        f"masked L1 coarse={health['masked_l1_coarse']:.4f}, "
        f"refined={health['masked_l1_refined']:.4f}, "
        f"gain={health['refinement_gain_pct']:.2f}%, "
        f"better_rate={health['refined_better_rate']:.2f}, "
        f"valid coarse={health['valid_l1_coarse']:.4f}, "
        f"valid refined={health['valid_l1_refined']:.4f}, "
        f"raw valid refined={health['raw_valid_l1_refined']:.4f}{warning_suffix}"
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
            loss, loss_dict = compute_train_loss(criterion, coarse, refined, image, mask, train_mode=train_mode)
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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, args)


if __name__ == "__main__":
    main()
