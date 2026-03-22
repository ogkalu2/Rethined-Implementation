from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from distributed_utils import unwrap_model


_LEGACY_OPTIONAL_MISSING_KEYS = {
    "inpainter.transport_candidate_refine_head.0.weight",
    "inpainter.transport_candidate_refine_head.2.weight",
    "inpainter.transport_candidate_refine_head.4.weight",
    "inpainter.transport_candidate_refine_head.4.bias",
    "inpainter.transport_candidate_score_head.0.weight",
    "inpainter.transport_candidate_score_head.2.weight",
    "inpainter.transport_candidate_score_head.4.weight",
    "inpainter.transport_candidate_score_head.4.bias",
    "hr_upscaler.patch_inpainting.transport_candidate_refine_head.0.weight",
    "hr_upscaler.patch_inpainting.transport_candidate_refine_head.2.weight",
    "hr_upscaler.patch_inpainting.transport_candidate_refine_head.4.weight",
    "hr_upscaler.patch_inpainting.transport_candidate_refine_head.4.bias",
    "hr_upscaler.patch_inpainting.transport_candidate_score_head.0.weight",
    "hr_upscaler.patch_inpainting.transport_candidate_score_head.2.weight",
    "hr_upscaler.patch_inpainting.transport_candidate_score_head.4.weight",
    "hr_upscaler.patch_inpainting.transport_candidate_score_head.4.bias",
}


def save_checkpoint(
    model,
    optimizer_g,
    scaler,
    step,
    epoch,
    metrics,
    cfg,
    path,
):
    raw_model = unwrap_model(model)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def prune_checkpoints(checkpoint_dir, keep_last_checkpoints, prefix="step_", preserve_paths=()):
    if keep_last_checkpoints is None:
        return []
    keep_last_checkpoints = int(keep_last_checkpoints)
    if keep_last_checkpoints <= 0:
        return []

    checkpoint_dir = Path(checkpoint_dir)
    preserve_paths = {Path(path).resolve() for path in preserve_paths}
    numbered_checkpoints = []
    for path in checkpoint_dir.glob(f"{prefix}*.pth"):
        try:
            step = int(path.stem[len(prefix):])
        except (IndexError, ValueError):
            continue
        numbered_checkpoints.append((step, path))

    numbered_checkpoints.sort(key=lambda item: (item[0], item[1].name))
    kept_paths = {path.resolve() for _, path in numbered_checkpoints[-keep_last_checkpoints:]}
    removed_paths = []
    for _, path in numbered_checkpoints:
        resolved_path = path.resolve()
        if resolved_path in kept_paths or resolved_path in preserve_paths:
            continue
        path.unlink(missing_ok=True)
        removed_paths.append(path)
    return removed_paths


def load_model_checkpoint(model, state_dict):
    raw_model = unwrap_model(model)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    missing_keys, unexpected_keys = raw_model.load_state_dict(state_dict, strict=False)
    ignored_missing_keys = sorted(key for key in missing_keys if key in _LEGACY_OPTIONAL_MISSING_KEYS)
    missing_keys = [key for key in missing_keys if key not in _LEGACY_OPTIONAL_MISSING_KEYS]
    if ignored_missing_keys:
        warnings.warn(
            "Loading checkpoint with randomly initialized legacy transport candidate heads: "
            f"{ignored_missing_keys}",
            RuntimeWarning,
        )
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
        "epoch": ckpt.get("epoch"),
        "best_metric_name": metrics.get("best_metric_name"),
        "best_metric_mode": metrics.get("best_metric_mode"),
        "best_metric_value": metrics.get("best_metric_value"),
        "best_metric_step": metrics.get("best_metric_step"),
    }


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
    if "transport_selection_loss" in metrics:
        summary += f", tsel={metrics['transport_selection_loss']:.4f}"
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


def write_status(log_dir, step, total_steps, epoch, total_epochs, metrics, lr):
    status = {
        "step": step,
        "total_steps": total_steps,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "progress_pct": round(100.0 * step / max(total_steps, 1), 2),
        "lr": lr,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    status_path = Path(log_dir) / "training_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)


def write_validation_history(log_dir, step, epoch, metrics):
    history_path = Path(log_dir) / "validation_history.json"
    entry = {
        "step": step,
        "epoch": epoch,
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
