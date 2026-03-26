from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from device_utils import (
    empty_device_cache,
    get_autocast_device_type,
    get_peak_memory_allocated_gb,
    is_amp_enabled,
)
from distributed_utils import barrier, reduce_metrics, reduce_scalar
from losses import InpaintingLoss
from model import InpaintingModel

from .checkpoints import (
    build_checkpoint_metrics,
    format_train_metric_snapshot,
    format_val_selection_snapshot,
    load_training_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    save_vis,
    write_status,
    write_validation_history,
)
from .common import (
    NullSummaryWriter,
    build_model_config,
    create_summary_writer,
    epochs_from_steps,
    get_lr,
    is_better_metric,
    metric_to_float,
    next_epoch_interval_target,
    prepare_multiscale_batch,
    print_device_banner,
    seed_everything,
    steps_from_epochs,
)
from .data import build_eval_loader, build_train_loader
from .eval import render_visualization_batch, validate_model


def _tensor_range(tensor: torch.Tensor) -> tuple[float, float]:
    detached = tensor.detach().float()
    finite_mask = torch.isfinite(detached)
    if not bool(finite_mask.any()):
        return float("nan"), float("nan")
    finite_values = detached[finite_mask]
    return float(finite_values.min().item()), float(finite_values.max().item())


def _find_nonfinite_named_parameters(model: torch.nn.Module, limit: int = 8) -> list[str]:
    bad_names = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param.detach()).all():
            bad_names.append(name)
            if len(bad_names) >= limit:
                break
    return bad_names


def _summarize_nonfinite_step(
    *,
    step: int,
    accum_idx: int,
    batch: dict,
    batch_views: dict[str, torch.Tensor],
    coarse_raw: torch.Tensor,
    refined_raw: torch.Tensor,
    g_loss: torch.Tensor,
    g_metrics: dict[str, torch.Tensor | float],
    device: torch.device,
    use_amp: bool,
) -> list[str]:
    image_paths = batch.get("image_path") or []
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    mask_paths = batch.get("mask_path") or []
    if isinstance(mask_paths, str):
        mask_paths = [mask_paths]
    sources = batch.get("source") or []
    if isinstance(sources, str):
        sources = [sources]

    mask = batch_views["mask"]
    image = batch_views["image"]
    masked_image = batch_views["masked_image"]
    refine_target = batch_views["refine_target"]
    mask_coverage = mask.detach().float().mean(dim=(1, 2, 3))
    metric_snapshot = {
        key: metric_to_float(value)
        for key, value in g_metrics.items()
    }
    nonfinite_metrics = [
        key for key, value in metric_snapshot.items()
        if not torch.isfinite(torch.tensor(value))
    ]

    lines = [
        f"Skipping non-finite step {step} (microbatch {accum_idx + 1})",
        f"  device={device.type} amp={use_amp}",
        (
            "  mask_coverage="
            f"{', '.join(f'{float(value):.4f}' for value in mask_coverage.cpu())}"
        ),
        (
            "  loss="
            f"{float(g_loss.detach().float().item()) if torch.isfinite(g_loss.detach()).all() else 'non-finite'}"
        ),
        f"  image_range={_tensor_range(image)} masked_range={_tensor_range(masked_image)}",
        f"  refine_target_range={_tensor_range(refine_target)}",
        f"  coarse_range={_tensor_range(coarse_raw)} refined_range={_tensor_range(refined_raw)}",
        (
            "  nonfinite_metrics="
            f"{', '.join(nonfinite_metrics) if nonfinite_metrics else '<loss only>'}"
        ),
    ]
    if sources:
        lines.append(f"  sources={', '.join(str(source) for source in sources)}")
    if image_paths:
        lines.append(f"  image_paths={', '.join(str(path) for path in image_paths)}")
    nonempty_mask_paths = [str(path) for path in mask_paths if str(path)]
    if nonempty_mask_paths:
        lines.append(f"  mask_paths={', '.join(nonempty_mask_paths)}")
    if use_amp and device.type == "xpu":
        lines.append("  note=XPU AMP is active; rerun with training.mixed_precision: false to rule out precision instability.")
    return lines


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
        "query_context_descriptor_head", "key_context_descriptor_head", "query_context_scale",
        "key_coarse_rgb_scale", "key_feature_scale",
        "pre_attention_norm", "multihead_attention",
    }
    scorer_params = []
    base_params = []
    for name, param in raw_model.named_parameters():
        parts = name.split(".")
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
    if cfg["training"]["mixed_precision"] and not use_amp and dist_ctx.is_main_process:
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
    grad_accum = 1 if args.overfit else cfg["training"]["grad_accum_steps"]
    steps_per_epoch = len(train_loader) / max(grad_accum, 1)
    total_epochs_cfg = cfg["training"].get("total_epochs")
    if args.steps is not None:
        total_steps = args.steps
        total_epochs = epochs_from_steps(total_steps, steps_per_epoch)
    elif args.epochs is not None:
        total_epochs = float(args.epochs)
        total_steps = steps_from_epochs(total_epochs, steps_per_epoch)
    elif total_epochs_cfg is not None:
        total_epochs = float(total_epochs_cfg)
        total_steps = steps_from_epochs(total_epochs, steps_per_epoch)
    else:
        total_steps = cfg["training"]["total_steps"]
        total_epochs = epochs_from_steps(total_steps, steps_per_epoch)

    max_lr = cfg["training"]["lr"]
    min_lr = cfg["training"]["min_lr"]
    warmup_epochs_cfg = cfg["training"].get("warmup_epochs")
    if warmup_epochs_cfg is not None:
        warmup_steps = steps_from_epochs(float(warmup_epochs_cfg), steps_per_epoch)
    else:
        warmup_steps = cfg["training"]["warmup_steps"]
    grad_clip_g = cfg["training"].get("grad_clip", 1.0)
    model_image_size = raw_model.inpainter.image_size

    eval_interval_epochs = log_cfg.get("eval_interval_epochs")
    vis_interval_epochs = log_cfg.get("vis_interval_epochs")
    save_interval_epochs = log_cfg.get("save_interval_epochs")
    checkpoint_steps = set(log_cfg.get("checkpoint_steps") or [])
    checkpoint_epochs = log_cfg.get("checkpoint_epochs") or []
    checkpoint_steps.update(steps_from_epochs(epoch, steps_per_epoch) for epoch in checkpoint_epochs)
    eval_interval_steps = 0 if eval_interval_epochs is not None else log_cfg.get("eval_interval", 0)
    vis_interval_steps = 0 if vis_interval_epochs is not None else log_cfg.get("vis_interval", 0)
    save_interval_steps = 0 if save_interval_epochs is not None else log_cfg.get("save_interval", 0)
    next_eval_epoch = (
        next_epoch_interval_target(start_step, steps_per_epoch, float(eval_interval_epochs))
        if eval_loader is not None and eval_interval_epochs
        else None
    )
    next_vis_epoch = (
        next_epoch_interval_target(start_step, steps_per_epoch, float(vis_interval_epochs))
        if vis_interval_epochs
        else None
    )
    next_save_epoch = (
        next_epoch_interval_target(start_step, steps_per_epoch, float(save_interval_epochs))
        if save_interval_epochs
        else None
    )

    if dist_ctx.is_main_process:
        print(f"Inpainter parameters: {sum(p.numel() for p in raw_model.parameters()):,}")
        print(f"  Base parameters: {sum(p.numel() for p in base_params):,}")
        print(f"  Scorer parameters: {sum(p.numel() for p in scorer_params):,} (LR: {scorer_lr})")
        print(f"Per-rank effective batch size: {cfg['data']['batch_size'] * grad_accum}")
        print(f"Global effective batch size: {cfg['data']['batch_size'] * grad_accum * dist_ctx.world_size}")
        print(
            f"Training schedule: {total_epochs:.2f} epochs "
            f"(~{total_steps} steps, {steps_per_epoch:.2f} steps/epoch)"
        )
        print(
            f"Warmup: {epochs_from_steps(warmup_steps, steps_per_epoch):.2f} epochs "
            f"({warmup_steps} steps)"
        )

    log_dir = Path(log_cfg["log_dir"])
    ckpt_dir = log_dir / "models"
    if dist_ctx.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = create_summary_writer(log_dir / "tb") if dist_ctx.is_main_process else NullSummaryWriter()
    if args.resume and dist_ctx.is_main_process:
        print(f"Resumed from step {start_step} (~epoch {epochs_from_steps(start_step, steps_per_epoch):.2f})")

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
        epoch = epochs_from_steps(step, steps_per_epoch)
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
                    if dist_ctx.is_main_process:
                        for line in _summarize_nonfinite_step(
                            step=step,
                            accum_idx=accum_idx,
                            batch=batch,
                            batch_views=batch_views,
                            coarse_raw=coarse_raw,
                            refined_raw=refined_raw,
                            g_loss=g_loss,
                            g_metrics=g_metrics,
                            device=device,
                            use_amp=use_amp,
                        ):
                            progress_bar.write(line)
                    break

                scaler.scale(g_loss / grad_accum).backward()

            if step_has_nonfinite:
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
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip_g)
        grad_norm_value = float(grad_norm.detach().float().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        if not torch.isfinite(torch.tensor(grad_norm_value)):
            optimizer_g.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Gradient norm became non-finite at step {step} before optimizer step. "
                "Lower the learning rate or disable mixed precision."
            )
        scaler.step(optimizer_g)
        scaler.update()
        bad_param_names = _find_nonfinite_named_parameters(raw_model)
        if bad_param_names:
            optimizer_g.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Model parameters became non-finite immediately after optimizer step {step}. "
                f"First affected parameters: {', '.join(bad_param_names)}. "
                "Resume from the last good checkpoint and consider lowering the learning rate "
                "or disabling mixed precision."
            )
        optimizer_g.zero_grad(set_to_none=True)

        metrics = {key: value / grad_accum for key, value in metric_sums.items()}
        metrics = reduce_metrics(metrics, dist_ctx, average=True)
        running_g = 0.9 * running_g + 0.1 * metrics["inpainter_total"]
        if dist_ctx.is_main_process:
            postfix = {"epoch": f"{epoch:.2f}/{total_epochs:.2f}"}
            if log_cfg.get("print_train_metrics", False):
                postfix["i"] = f"{metrics['inpainter_total']:.4f}"
                postfix["l1"] = f"{metrics['refined_l1']:.4f}"
                postfix["freq"] = f"{metrics['focal_frequency']:.4f}"
                postfix["perc"] = f"{metrics['perceptual']:.4f}"
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
                if "transport_selection_loss" in metrics:
                    postfix["tsel"] = f"{metrics['transport_selection_loss']:.4f}"
            progress_bar.set_postfix(postfix, refresh=False)

        if dist_ctx.is_main_process and (step == 1 or step % log_cfg["log_interval"] == 0):
            writer.add_scalar("loss/coarse_l2", metrics["coarse_l2"], step)
            writer.add_scalar("loss/refined_l1", metrics["refined_l1"], step)
            writer.add_scalar("loss/focal_frequency", metrics["focal_frequency"], step)
            if "refined_query_patch_l1" in metrics:
                writer.add_scalar("loss/refined_query_patch_l1", metrics["refined_query_patch_l1"], step)
            if "retrieval_loss" in metrics:
                writer.add_scalar("loss/retrieval", metrics["retrieval_loss"], step)
            if "retrieval_coherence_loss" in metrics:
                writer.add_scalar("loss/retrieval_coherence", metrics["retrieval_coherence_loss"], step)
            if "transport_patch" in metrics:
                writer.add_scalar("loss/transport_patch", metrics["transport_patch"], step)
            if "transport_selection_loss" in metrics:
                writer.add_scalar("loss/transport_selection", metrics["transport_selection_loss"], step)
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
            write_status(log_cfg["log_dir"], step, total_steps, epoch, total_epochs, metrics, lr_g)

        should_eval = False
        if eval_loader is not None:
            if next_eval_epoch is not None and epoch + 1e-9 >= next_eval_epoch:
                should_eval = True
                while epoch + 1e-9 >= next_eval_epoch:
                    next_eval_epoch += float(eval_interval_epochs)
            elif eval_interval_steps and step % eval_interval_steps == 0:
                should_eval = True

        if should_eval:
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
                write_validation_history(log_cfg["log_dir"], step, epoch, val_metrics)
                retrieval_snapshot = format_val_selection_snapshot(
                    val_metrics,
                    retrieval_margin_label=retrieval_margin_label,
                )
                retrieval_suffix = f" | val {retrieval_snapshot}" if retrieval_snapshot else ""
                progress_bar.write(
                    f"Validation epoch {epoch:.2f} (step {step}): "
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
                        epoch,
                        best_metrics,
                        cfg,
                        best_path,
                    )
                    progress_bar.write(
                        f"Saved best checkpoint: {best_path} "
                        f"({best_metric_name}={best_metric_value:.6f} at step {best_metric_step}, epoch {epoch:.2f})"
                    )
                keep_last_eval_checkpoints = log_cfg.get("keep_last_checkpoints")
                if keep_last_eval_checkpoints:
                    eval_ckpt_path = ckpt_dir / f"eval_step_{step}.pth"
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
                        epoch,
                        checkpoint_metrics,
                        cfg,
                        eval_ckpt_path,
                    )
                    progress_bar.write(f"Saved eval checkpoint: {eval_ckpt_path} (epoch {epoch:.2f})")
                    removed_paths = prune_checkpoints(
                        ckpt_dir,
                        keep_last_eval_checkpoints,
                        prefix="eval_step_",
                        preserve_paths=(eval_ckpt_path,),
                    )
                    if removed_paths:
                        removed_names = ", ".join(path.name for path in removed_paths)
                        progress_bar.write(f"Pruned old eval checkpoints: {removed_names}")
            empty_device_cache(device)
            barrier(dist_ctx)

        should_render_vis = False
        if next_vis_epoch is not None and epoch + 1e-9 >= next_vis_epoch:
            should_render_vis = True
            while epoch + 1e-9 >= next_vis_epoch:
                next_vis_epoch += float(vis_interval_epochs)
        elif vis_interval_steps:
            should_render_vis = step % vis_interval_steps == 0

        if dist_ctx.is_main_process and should_render_vis and last_batch_views is not None:
            vis_coarse, vis_refined = render_visualization_batch(
                model,
                last_batch_views,
                device,
                cfg["training"]["mixed_precision"],
            )
            save_vis(writer, last_batch_views, vis_coarse, vis_refined, step, log_dir=log_cfg["log_dir"])

        should_save = False
        if checkpoint_steps:
            should_save = step in checkpoint_steps
        elif log_cfg.get("save_checkpoints", True):
            if next_save_epoch is not None and epoch + 1e-9 >= next_save_epoch:
                should_save = True
                while epoch + 1e-9 >= next_save_epoch:
                    next_save_epoch += float(save_interval_epochs)
            elif save_interval_steps:
                should_save = step % save_interval_steps == 0

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
                epoch,
                checkpoint_metrics,
                cfg,
                ckpt_path,
            )
            progress_bar.write(f"Saved checkpoint: {ckpt_path} (epoch {epoch:.2f})")

    final_path = ckpt_dir / f"step_{total_steps}.pth"
    if dist_ctx.is_main_process and log_cfg.get("save_final_checkpoint", True):
        final_epoch = epochs_from_steps(total_steps, steps_per_epoch)
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
            final_epoch,
            final_metrics,
            cfg,
            final_path,
        )
        progress_bar.write(f"Training complete at epoch {final_epoch:.2f}. Final checkpoint: {final_path}")
    elif dist_ctx.is_main_process:
        progress_bar.write("Training complete. Final checkpoint saving disabled.")
    progress_bar.close()
    writer.close()
    barrier(dist_ctx)
