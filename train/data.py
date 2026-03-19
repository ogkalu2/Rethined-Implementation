from __future__ import annotations

from data.dataset import get_dataloader


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
    eval_interval_epochs = cfg.get("logging", {}).get("eval_interval_epochs", 0)
    if eval_interval <= 0 and eval_interval_epochs <= 0 and not args.eval_only:
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
