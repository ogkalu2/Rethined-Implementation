"""Paper-aligned RETHINED evaluation."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from cleanfid import fid
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import save_image
from tqdm import tqdm

from data.dataset import get_dataloader
from device_utils import empty_device_cache, get_device_name, is_amp_enabled, resolve_device, time_device_call
from hr import AttentionUpscaling
from model import InpaintingModel
from train import build_model_config, composite_with_known, load_model_checkpoint, prepare_multiscale_batch


def load_model(checkpoint_path, cfg, device, random_init=False):
    model = InpaintingModel(build_model_config(cfg)).to(device)
    if not random_init:
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required unless random_init=True")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        load_model_checkpoint(model, state_dict)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, attn_upscaler, batch_views, device, amp_enabled):
    with torch.amp.autocast(device.type, enabled=amp_enabled):
        refined_lr, attn_map, coarse_raw = model(batch_views["masked_image"], batch_views["mask"])

    refined_lr = refined_lr.clamp(0, 1)
    coarse_lr = composite_with_known(coarse_raw.clamp(0, 1), batch_views["image"], batch_views["mask"])

    if batch_views["has_hr_target"]:
        target = batch_views["image_hr"]
        eval_mask = batch_views["mask_hr"]
        coarse_eval = composite_with_known(
            F.interpolate(coarse_lr, size=target.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1),
            target,
            eval_mask,
        )
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            refined_hr = attn_upscaler(
                batch_views["masked_image_hr"],
                refined_lr,
                attn_map,
                mask_hr=batch_views["mask_hr"],
            ).clamp(0, 1)
        refined_eval = composite_with_known(refined_hr, target, eval_mask)
    else:
        target = batch_views["image"]
        eval_mask = batch_views["mask"]
        coarse_eval = coarse_lr
        refined_eval = refined_lr

    return coarse_eval, refined_eval, target, eval_mask


def save_eval_image(tensor, path):
    save_image(tensor.clamp(0, 1), path)


@torch.no_grad()
def evaluate_quality(model, dataloader, device, model_image_size, num_images=200):
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    attn_upscaler = AttentionUpscaling(model.generator)
    amp_enabled = is_amp_enabled(device, True)

    metrics = {
        "l1_coarse": [],
        "l1_refined": [],
        "ssim_coarse": [],
        "ssim_refined": [],
        "lpips_coarse": [],
        "lpips_refined": [],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        coarse_dir = tmp_root / "coarse"
        refined_dir = tmp_root / "refined"
        target_dir = tmp_root / "target"
        coarse_dir.mkdir(parents=True, exist_ok=True)
        refined_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for batch in tqdm(dataloader, desc="Quality eval"):
            if count >= num_images:
                break

            batch_views = prepare_multiscale_batch(
                batch,
                device,
                model_image_size,
                blur_layer=model.generator.final_gaussian_blur,
            )
            coarse_eval, refined_eval, target, eval_mask = run_inference(
                model,
                attn_upscaler,
                batch_views,
                device,
                amp_enabled,
            )

            for sample_idx in range(target.shape[0]):
                if count >= num_images:
                    break

                gt = target[sample_idx]
                coarse = coarse_eval[sample_idx]
                refined = refined_eval[sample_idx]
                mask = eval_mask[sample_idx]

                if mask.sum() > 0:
                    metrics["l1_coarse"].append(F.l1_loss(coarse * mask, gt * mask).item() / mask.mean().item())
                    metrics["l1_refined"].append(F.l1_loss(refined * mask, gt * mask).item() / mask.mean().item())
                else:
                    metrics["l1_coarse"].append(0.0)
                    metrics["l1_refined"].append(0.0)

                gt_np = gt.cpu().permute(1, 2, 0).numpy()
                coarse_np = coarse.cpu().permute(1, 2, 0).numpy()
                refined_np = refined.cpu().permute(1, 2, 0).numpy()
                metrics["ssim_coarse"].append(ssim(gt_np, coarse_np, channel_axis=2, data_range=1.0))
                metrics["ssim_refined"].append(ssim(gt_np, refined_np, channel_axis=2, data_range=1.0))

                gt_lpips = gt.unsqueeze(0) * 2 - 1
                coarse_lpips = coarse.unsqueeze(0) * 2 - 1
                refined_lpips = refined.unsqueeze(0) * 2 - 1
                metrics["lpips_coarse"].append(lpips_fn(coarse_lpips, gt_lpips).item())
                metrics["lpips_refined"].append(lpips_fn(refined_lpips, gt_lpips).item())

                save_eval_image(coarse, coarse_dir / f"{count:06d}.png")
                save_eval_image(refined, refined_dir / f"{count:06d}.png")
                save_eval_image(gt, target_dir / f"{count:06d}.png")
                count += 1

        results = {}
        for key, values in metrics.items():
            results[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

        results["fid_coarse"] = float(fid.compute_fid(str(coarse_dir), str(target_dir), device=device, num_workers=0))
        results["fid_refined"] = float(fid.compute_fid(str(refined_dir), str(target_dir), device=device, num_workers=0))
        return results


@torch.no_grad()
def benchmark_speed(model, device, num_runs=50, warmup=10):
    resolution = model.generator.image_size
    amp_enabled = is_amp_enabled(device, True)
    dummy_img = torch.randn(1, 3, resolution, resolution, device=device)
    dummy_mask = torch.zeros(1, 1, resolution, resolution, device=device)
    dummy_mask[:, :, resolution // 4:3 * resolution // 4, resolution // 4:3 * resolution // 4] = 1.0
    dummy_masked = dummy_img * (1 - dummy_mask)

    for _ in range(warmup):
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            _ = model(dummy_masked, dummy_mask)

    latencies = []
    for _ in range(num_runs):
        latencies.append(time_device_call(lambda: model(dummy_masked, dummy_mask), device))

    latencies = np.array(latencies)
    return {
        str(resolution): {
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "throughput_fps": float(1000.0 / np.mean(latencies)),
        }
    }


@torch.no_grad()
def benchmark_upscaling(model, device, hr_resolutions, num_runs=30, warmup=5):
    lr_res = model.generator.image_size
    attn_upscaler = AttentionUpscaling(model.generator)
    amp_enabled = is_amp_enabled(device, True)

    dummy_lr = torch.randn(1, 3, lr_res, lr_res, device=device)
    dummy_mask = torch.zeros(1, 1, lr_res, lr_res, device=device)
    dummy_mask[:, :, lr_res // 4:3 * lr_res // 4, lr_res // 4:3 * lr_res // 4] = 1.0
    dummy_masked = dummy_lr * (1 - dummy_mask)
    with torch.amp.autocast(device.type, enabled=amp_enabled):
        refined_lr, attn_map, _ = model(dummy_masked, dummy_mask)
    refined_lr = refined_lr.clamp(0, 1)

    results = {}
    for hr_res in hr_resolutions:
        try:
            dummy_hr = torch.randn(1, 3, hr_res, hr_res, device=device)
            dummy_hr_mask = torch.zeros(1, 1, hr_res, hr_res, device=device)
            dummy_hr_mask[:, :, hr_res // 4:3 * hr_res // 4, hr_res // 4:3 * hr_res // 4] = 1.0
            dummy_hr_masked = dummy_hr * (1 - dummy_hr_mask)

            for _ in range(warmup):
                with torch.amp.autocast(device.type, enabled=amp_enabled):
                    _ = attn_upscaler(dummy_hr_masked, refined_lr, attn_map, mask_hr=dummy_hr_mask)

            latencies = []
            output = None
            for _ in range(num_runs):
                def run():
                    nonlocal output
                    with torch.amp.autocast(device.type, enabled=amp_enabled):
                        output = attn_upscaler(dummy_hr_masked, refined_lr, attn_map, mask_hr=dummy_hr_mask)

                latencies.append(time_device_call(run, device))

            latencies = np.array(latencies)
            results[str(hr_res)] = {
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "output_shape": list(output.shape),
            }
            empty_device_cache(device)
        except RuntimeError as exc:
            results[str(hr_res)] = {"error": str(exc)}
            empty_device_cache(device)

    return results


@torch.no_grad()
def test_upscaling_quality(model, dataloader, device, hr_res=2048, num_images=50):
    attn_upscaler = AttentionUpscaling(model.generator)
    amp_enabled = is_amp_enabled(device, True)
    l1_values = []
    count = 0

    for batch in tqdm(dataloader, desc=f"Upscaling quality {hr_res}"):
        if count >= num_images:
            break

        batch_views = prepare_multiscale_batch(
            batch,
            device,
            model.generator.image_size,
            blur_layer=model.generator.final_gaussian_blur,
        )
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            refined_lr, attn_map, _ = model(batch_views["masked_image"], batch_views["mask"])
        refined_lr = refined_lr.clamp(0, 1)

        image_hr = F.interpolate(batch_views["image_hr"], size=(hr_res, hr_res), mode="bicubic", align_corners=False)
        mask_hr = F.interpolate(batch_views["mask_hr"], size=(hr_res, hr_res), mode="nearest")
        masked_hr = image_hr * (1 - mask_hr)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            refined_hr = attn_upscaler(masked_hr, refined_lr, attn_map, mask_hr=mask_hr).clamp(0, 1)
        refined_hr = composite_with_known(refined_hr, image_hr, mask_hr)

        for sample_idx in range(image_hr.shape[0]):
            if count >= num_images:
                break
            mask = mask_hr[sample_idx]
            if mask.sum() > 0:
                l1_values.append(
                    F.l1_loss(refined_hr[sample_idx] * mask, image_hr[sample_idx] * mask).item() / mask.mean().item()
                )
            count += 1

    return {
        "hr_resolution": hr_res,
        "num_images": len(l1_values),
        "l1_mean": float(np.mean(l1_values)) if l1_values else None,
        "l1_std": float(np.std(l1_values)) if l1_values else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Paper-aligned RETHINED evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--speed_only", action="store_true")
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--reparameterize", action="store_true")
    parser.add_argument("--upscale_test", action="store_true")
    parser.add_argument("--speed_runs", type=int, default=50)
    parser.add_argument("--speed_warmup", type=int, default=10)
    parser.add_argument("--upscale_runs", type=int, default=30)
    parser.add_argument("--upscale_warmup", type=int, default=5)
    parser.add_argument("--hr_resolutions", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if not args.random_init and not args.checkpoint:
        parser.error("--checkpoint is required unless --random_init is set")

    device = resolve_device(args.device)
    print(f"Device: {device}")
    if device.type in {"cuda", "xpu"}:
        print(f"Accelerator: {get_device_name(device)}")

    model = load_model(args.checkpoint, cfg, device, random_init=args.random_init)
    if args.reparameterize:
        model.reparameterize()
        model.eval()
        print("Applied inference reparameterization.")

    results = {
        "checkpoint": args.checkpoint,
        "random_init": args.random_init,
        "reparameterized": args.reparameterize,
        "params": sum(p.numel() for p in model.parameters()),
    }

    print("\n--- Speed Benchmark (base model) ---")
    speed = benchmark_speed(model, device, num_runs=args.speed_runs, warmup=args.speed_warmup)
    results["speed"] = speed
    for res, info in speed.items():
        print(f"  {res}x{res}: {info['mean_ms']:.2f}ms mean, {info['throughput_fps']:.1f} FPS")

    print("\n--- AttentionUpscaling Speed ---")
    if args.hr_resolutions:
        hr_resolutions = tuple(int(item.strip()) for item in args.hr_resolutions.split(",") if item.strip())
    else:
        hr_resolutions = tuple(dict.fromkeys([cfg["data"]["image_size"], 2048, 4096]))
    up_speed = benchmark_upscaling(
        model,
        device,
        hr_resolutions=hr_resolutions,
        num_runs=args.upscale_runs,
        warmup=args.upscale_warmup,
    )
    results["upscaling_speed"] = up_speed
    for res, info in up_speed.items():
        if "mean_ms" in info:
            print(f"  LR={model.generator.image_size} -> HR={res}: {info['mean_ms']:.2f}ms")
        else:
            print(f"  LR={model.generator.image_size} -> HR={res}: {info}")

    if args.speed_only:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nResults saved to {args.output}")
        return

    print("\n--- Quality Metrics ---")
    val_loader = get_dataloader(
        root_dir=cfg["data"]["root_dir"],
        image_size=cfg["data"]["image_size"],
        split="val",
        batch_size=cfg["data"].get("eval_batch_size", cfg["data"]["batch_size"]),
        num_workers=max(1, min(2, cfg["data"]["num_workers"])),
        persistent_workers=cfg["data"].get("persistent_workers"),
        prefetch_factor=cfg["data"].get("prefetch_factor"),
        mask_min_coverage=cfg["data"]["mask_min_coverage"],
        mask_max_coverage=cfg["data"]["mask_max_coverage"],
        val_dir=cfg["data"].get("val_dir"),
        manifest_path=cfg["data"].get("manifest_path"),
        deterministic=cfg["data"].get("force_random_masks_eval", False),
        fixed_mask_seed=cfg["training"]["seed"],
        force_random_masks=cfg["data"].get("force_random_masks_eval", False),
        shuffle_override=False,
    )
    quality = evaluate_quality(model, val_loader, device, model.generator.image_size, num_images=args.num_images)
    results["quality"] = quality
    print(f"  {'Metric':<20} {'Coarse':>10} {'Refined':>10} {'Improvement':>12}")
    print(f"  {'-' * 58}")
    for base in ["l1", "ssim", "lpips", "fid"]:
        coarse_key = f"{base}_coarse"
        refined_key = f"{base}_refined"
        coarse_value = quality[coarse_key]["mean"] if isinstance(quality[coarse_key], dict) else quality[coarse_key]
        refined_value = quality[refined_key]["mean"] if isinstance(quality[refined_key], dict) else quality[refined_key]
        if base == "ssim":
            improvement = f"+{(refined_value - coarse_value) * 100:.1f}%"
        else:
            improvement = f"-{(coarse_value - refined_value) / coarse_value * 100:.1f}%" if coarse_value > 0 else "N/A"
        print(f"  {base.upper():<20} {coarse_value:>10.4f} {refined_value:>10.4f} {improvement:>12}")

    if args.upscale_test or not args.speed_only:
        print("\n--- AttentionUpscaling Quality ---")
        for hr_res in hr_resolutions:
            if hr_res <= model.generator.image_size:
                continue
            up_quality = test_upscaling_quality(model, val_loader, device, hr_res=hr_res, num_images=min(args.num_images, 50))
            results[f"upscaling_quality_{hr_res}"] = up_quality
            if up_quality["l1_mean"] is not None:
                print(f"  HR={hr_res}: L1={up_quality['l1_mean']:.4f} (n={up_quality['num_images']})")
            else:
                print(f"  HR={hr_res}: Failed")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
