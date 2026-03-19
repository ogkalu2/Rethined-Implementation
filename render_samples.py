"""Render a few dataset samples from a checkpoint for visual inspection."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image

from data.dataset import InpaintingDataset
from device_utils import resolve_device
from model import InpaintingModel
from train.checkpoints import load_model_checkpoint
from train.common import build_model_config, composite_with_known


def load_model_and_cfg(checkpoint_path: Path, config_path: Path, device: torch.device):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = InpaintingModel(build_model_config(cfg)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_model_checkpoint(model, ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def make_dataset_sample(
    dataset: InpaintingDataset,
    idx: int,
    random_masks: bool,
    random_mask_seed: int | None = None,
):
    sample = dataset.samples[idx]
    dataset_sample = dataset[idx]
    image = dataset_sample["image"]

    if random_masks:
        rng = None
        if random_mask_seed is not None:
            rng = np.random.RandomState(random_mask_seed + idx)
        mask = torch.from_numpy(dataset.mask_gen(rng=rng)).unsqueeze(0)
        masked_image = image * (1 - mask)
    else:
        mask = dataset_sample["mask"]
        masked_image = dataset_sample["masked_image"]

    return {
        "image": image,
        "mask": mask,
        "masked_image": masked_image,
        "source": dataset_sample["source"],
        "image_path": str(sample["image_path"]),
    }


@torch.no_grad()
def predict_with_intermediates(model: InpaintingModel, image: torch.Tensor, mask: torch.Tensor):
    masked_image = image * (1 - mask)
    model_image_size = int(model.inpainter.image_size)
    if image.shape[-2:] == (model_image_size, model_image_size):
        refined, attn_map, coarse_raw = model.inpainter(
            masked_image,
            mask,
            value_image=image,
            return_aux=False,
        )
        refined = refined.clamp(0, 1)
        coarse_vis = composite_with_known(coarse_raw.clamp(0, 1), image, mask)
        refined_vis = refined
        return refined, coarse_vis, refined_vis

    known_hr = image
    refine_target_lr = F.interpolate(
        known_hr,
        size=(model_image_size, model_image_size),
        mode="bicubic",
        align_corners=False,
    )
    image_lr = model._prefilter_downsample(known_hr, model_image_size)
    mask_lr = F.interpolate(mask, size=(model_image_size, model_image_size), mode="nearest")
    mask_lr = (mask_lr > 0.5).to(image_lr.dtype)
    masked_lr = image_lr * (1 - mask_lr)

    refined_lr, attn_map, coarse_raw = model.inpainter(
        masked_lr,
        mask_lr,
        value_image=refine_target_lr,
        return_aux=False,
    )
    refined_lr = refined_lr.clamp(0, 1)
    coarse_lr = composite_with_known(coarse_raw.clamp(0, 1), refine_target_lr, mask_lr)

    masked_hr = known_hr * (1 - mask)
    final_hr = model.hr_upscaler(masked_hr, refined_lr, attn_map, mask_hr=mask).clamp(0, 1)
    coarse_hr = composite_with_known(
        F.interpolate(coarse_lr, size=known_hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1),
        known_hr,
        mask,
    )
    lr_refined_hr = composite_with_known(
        F.interpolate(refined_lr, size=known_hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1),
        known_hr,
        mask,
    )
    return final_hr, coarse_hr, lr_refined_hr


@torch.no_grad()
def render_sample(model, batch):
    image_hr = batch["image"].unsqueeze(0)
    mask_hr = batch["mask"].unsqueeze(0)
    masked_hr = batch["masked_image"].unsqueeze(0)

    hr_final, coarse_vis, lr_refined_vis = predict_with_intermediates(model, image_hr, mask_hr)
    mask_rgb = mask_hr.repeat(1, 3, 1, 1)

    return torch.cat(
        [
            image_hr.cpu(),
            masked_hr.cpu(),
            coarse_vis.cpu(),
            lr_refined_vis.cpu(),
            hr_final.cpu(),
            mask_rgb.cpu(),
        ],
        dim=3,
    ).squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Render sample outputs from a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_masks", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fixed_mask_seed", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg = load_model_and_cfg(Path(args.checkpoint), Path(args.config), device)

    dataset = InpaintingDataset(
        root_dir=cfg["data"].get("root_dir"),
        image_size=cfg["data"]["image_size"],
        split=args.split,
        mask_min_coverage=cfg["data"]["mask_min_coverage"],
        mask_max_coverage=cfg["data"]["mask_max_coverage"],
        max_images=args.max_images,
        val_dir=cfg["data"].get("val_dir"),
        manifest_path=cfg["data"].get("manifest_path"),
        deterministic=args.deterministic,
        fixed_mask_seed=args.fixed_mask_seed,
        mask_generator_kwargs=cfg["data"].get("mask_generator"),
    )
    if args.random_masks and any(sample["mask_path"] is not None for sample in dataset.samples):
        print(
            "Warning: --random_masks overrides manifest masks, so renders will not match paired-mask training/eval."
        )

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = [
        f"checkpoint: {args.checkpoint}",
        f"config: {args.config}",
        f"device: {device}",
        f"split: {args.split}",
        f"random_masks: {args.random_masks}",
        f"deterministic: {args.deterministic}",
        f"fixed_mask_seed: {args.fixed_mask_seed}",
        f"max_images: {args.max_images}",
        f"indices: {indices}",
        "Columns: ground_truth | masked_input | coarse_x2 | lr_refined_x2 | predict_final | mask",
        "",
    ]

    panels = []
    for slot, idx in enumerate(indices):
        batch = make_dataset_sample(
            dataset,
            idx,
            random_masks=args.random_masks,
            random_mask_seed=(args.fixed_mask_seed if args.deterministic else None),
        )
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        panel = render_sample(model, batch)
        panel_name = f"sample_{slot:02d}_idx_{idx}.png"
        save_image(panel, output_dir / panel_name)
        panels.append(panel)
        metadata.extend(
            [
                f"sample_{slot:02d}:",
                f"  idx: {idx}",
                f"  source: {batch['source']}",
                f"  image_path: {batch['image_path']}",
                f"  panel: {panel_name}",
                "",
            ]
        )

    if panels:
        save_image(torch.cat(panels, dim=1), output_dir / "grid.png")

    with open(output_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))

    print(f"Saved {len(panels)} sample panels to {output_dir}")
    print(f"Grid: {output_dir / 'grid.png'}")


if __name__ == "__main__":
    main()
