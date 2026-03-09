"""RETHINED Phase 005: Claim Verification.

Evaluates quality metrics, speed benchmarks, coarse vs refined comparison,
and multi-resolution AttentionUpscaling.

Usage:
    # Full evaluation
    python evaluate.py --checkpoint logs/train_256/checkpoints/step_10000.pth \
        --config configs/train_celeba_256.yaml --num_images 200

    # Speed benchmark only
    python evaluate.py --checkpoint logs/train_256/checkpoints/step_10000.pth \
        --config configs/train_celeba_256.yaml --speed_only

    # Multi-resolution test
    python evaluate.py --checkpoint logs/train_256/checkpoints/step_10000.pth \
        --config configs/train_celeba_256.yaml --upscale_test
"""

import argparse
import json
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from data.dataset import get_dataloader
from device_utils import (
    empty_device_cache,
    get_device_name,
    is_amp_enabled,
    resolve_device,
    time_device_call,
)
from model import InpaintingModel, AttentionUpscaling
from train import load_model_checkpoint


def build_model_config(cfg):
    coarse_cfg = cfg["model"]["coarse_model"]
    return {
        "coarse_model": {
            "class": coarse_cfg.get("class", "PaperCoarse"),
            "parameters": {k: v for k, v in coarse_cfg.items() if k != "class"},
        },
        "generator": {
            "generator_class": "PatchInpainting",
            "params": {k: v for k, v in cfg["model"]["generator"].items()},
        },
    }


def load_model(checkpoint_path, cfg, device):
    model_config = build_model_config(cfg)
    model = InpaintingModel(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_model_checkpoint(model, ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_quality(model, dataloader, device, num_images=200):
    """Compute L1, SSIM, LPIPS on validation set."""
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    amp_enabled = is_amp_enabled(device, True)

    metrics = {
        "l1_coarse": [], "l1_refined": [],
        "ssim_coarse": [], "ssim_refined": [],
        "lpips_coarse": [], "lpips_refined": [],
    }

    count = 0
    for batch in tqdm(dataloader, desc="Quality eval"):
        if count >= num_images:
            break

        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        masked_image = batch["masked_image"].to(device)
        bs = image.shape[0]

        with torch.amp.autocast(device.type, enabled=amp_enabled):
            refined, attn, coarse = model(masked_image, mask)

        refined = refined.clamp(0, 1) * mask + image * (1 - mask)
        coarse = coarse.clamp(0, 1) * mask + image * (1 - mask)

        for i in range(bs):
            if count >= num_images:
                break

            gt = image[i]
            ref = refined[i]
            crs = coarse[i]

            # L1 (masked region only)
            m = mask[i]  # (1, H, W)
            if m.sum() > 0:
                metrics["l1_refined"].append(F.l1_loss(ref * m, gt * m).item() / m.mean().item())
                metrics["l1_coarse"].append(F.l1_loss(crs * m, gt * m).item() / m.mean().item())
            else:
                metrics["l1_refined"].append(0.0)
                metrics["l1_coarse"].append(0.0)

            # SSIM (full image)
            gt_np = gt.cpu().permute(1, 2, 0).numpy()
            ref_np = ref.cpu().permute(1, 2, 0).numpy()
            crs_np = crs.cpu().permute(1, 2, 0).numpy()
            metrics["ssim_refined"].append(ssim(gt_np, ref_np, channel_axis=2, data_range=1.0))
            metrics["ssim_coarse"].append(ssim(gt_np, crs_np, channel_axis=2, data_range=1.0))

            # LPIPS (full image)
            gt_lpips = gt.unsqueeze(0) * 2 - 1  # [0,1] -> [-1,1]
            ref_lpips = ref.unsqueeze(0) * 2 - 1
            crs_lpips = crs.unsqueeze(0) * 2 - 1
            metrics["lpips_refined"].append(lpips_fn(ref_lpips, gt_lpips).item())
            metrics["lpips_coarse"].append(lpips_fn(crs_lpips, gt_lpips).item())

            count += 1

    results = {}
    for k, v in metrics.items():
        results[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}

    return results


@torch.no_grad()
def benchmark_speed(model, device, resolutions=(256, 512), num_runs=50, warmup=10):
    """Benchmark inference speed at different resolutions."""
    results = {}
    amp_enabled = is_amp_enabled(device, True)

    for res in resolutions:
        # Check if resolution fits in VRAM
        try:
            dummy_img = torch.randn(1, 3, res, res, device=device)
            dummy_mask = torch.zeros(1, 1, res, res, device=device)
            dummy_mask[:, :, res//4:3*res//4, res//4:3*res//4] = 1.0
        except RuntimeError:
            results[str(res)] = {"error": "OOM"}
            continue

        # Need a model that works at this resolution
        if res != model.generator.image_size:
            # Skip non-native resolutions for base model — test those with upscaling
            results[str(res)] = {"note": "non-native, use AttentionUpscaling"}
            del dummy_img, dummy_mask
            empty_device_cache(device)
            continue

        # Warmup
        for _ in range(warmup):
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                _ = model(dummy_img, dummy_mask)

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            latencies.append(time_device_call(lambda: model(dummy_img, dummy_mask), device))

        latencies = np.array(latencies)
        results[str(res)] = {
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput_fps": float(1000.0 / np.mean(latencies)),
        }

        del dummy_img, dummy_mask
        empty_device_cache(device)

    return results


@torch.no_grad()
def benchmark_upscaling(
    model,
    device,
    lr_res=256,
    hr_resolutions=(512, 1024),
    disable_hr_residual_refiner=False,
    num_runs=30,
    warmup=5,
):
    """Benchmark AttentionUpscaling speed at higher resolutions."""
    attn_upscaler = AttentionUpscaling(model.generator)
    results = {}
    amp_enabled = is_amp_enabled(device, True)
    original_hr_refiner = model.generator.hr_residual_refiner
    if disable_hr_residual_refiner:
        model.generator.hr_residual_refiner = None

    try:
        # Generate LR input + attention map
        dummy_lr = torch.randn(1, 3, lr_res, lr_res, device=device)
        dummy_mask = torch.zeros(1, 1, lr_res, lr_res, device=device)
        dummy_mask[:, :, lr_res//4:3*lr_res//4, lr_res//4:3*lr_res//4] = 1.0

        # Run LR inference to get attention map
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            refined, attn_map, coarse = model(dummy_lr, dummy_mask)

        for hr_res in hr_resolutions:
            try:
                dummy_hr = torch.randn(1, 3, hr_res, hr_res, device=device)
                dummy_hr_mask = torch.zeros(1, 1, hr_res, hr_res, device=device)
                dummy_hr_mask[:, :, hr_res // 4:3 * hr_res // 4, hr_res // 4:3 * hr_res // 4] = 1.0

                # Warmup
                for _ in range(warmup):
                    with torch.amp.autocast(device.type, enabled=amp_enabled):
                        _ = attn_upscaler(dummy_hr, refined, attn_map, mask_hr=dummy_hr_mask)

                # Benchmark
                latencies = []
                for _ in range(num_runs):
                    hr_out = None

                    def run_upscaler():
                        nonlocal hr_out
                        with torch.amp.autocast(device.type, enabled=amp_enabled):
                            hr_out = attn_upscaler(dummy_hr, refined, attn_map, mask_hr=dummy_hr_mask)

                    latencies.append(time_device_call(run_upscaler, device))

                latencies = np.array(latencies)
                results[str(hr_res)] = {
                    "mean_ms": float(np.mean(latencies)),
                    "p50_ms": float(np.percentile(latencies, 50)),
                    "p95_ms": float(np.percentile(latencies, 95)),
                    "output_shape": list(hr_out.shape),
                }

                del dummy_hr, dummy_hr_mask, hr_out
                empty_device_cache(device)

            except RuntimeError as e:
                results[str(hr_res)] = {"error": str(e)}
                empty_device_cache(device)

        del dummy_lr, dummy_mask, refined, attn_map, coarse
        empty_device_cache(device)
    finally:
        model.generator.hr_residual_refiner = original_hr_refiner

    return results


@torch.no_grad()
def test_upscaling_quality(model, dataloader, device, hr_res=512, num_images=50):
    """Test if AttentionUpscaling maintains quality at higher resolution."""
    attn_upscaler = AttentionUpscaling(model.generator)
    amp_enabled = is_amp_enabled(device, True)

    l1_values = []
    count = 0

    for batch in tqdm(dataloader, desc=f"Upscaling quality {hr_res}"):
        if count >= num_images:
            break

        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        masked_image = batch["masked_image"].to(device)

        with torch.amp.autocast(device.type, enabled=amp_enabled):
            refined, attn_map, coarse = model(masked_image, mask)

        # Upscale original image to HR for comparison
        image_hr = F.interpolate(image, size=(hr_res, hr_res), mode="bicubic", align_corners=False)
        mask_hr = F.interpolate(mask, size=(hr_res, hr_res), mode="nearest")

        # Create HR masked input (bicubic upsample of original)
        masked_hr = image_hr * (1 - mask_hr)

        try:
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                hr_output = attn_upscaler(masked_hr, refined.clamp(0, 1), attn_map, mask_hr=mask_hr)
            hr_output = hr_output.clamp(0, 1)

            # L1 in masked region
            for i in range(image.shape[0]):
                if count >= num_images:
                    break
                m = mask_hr[i]
                if m.sum() > 0:
                    l1_values.append(
                        F.l1_loss(hr_output[i] * m, image_hr[i] * m).item() / m.mean().item()
                    )
                count += 1

        except RuntimeError:
            break

    return {
        "hr_resolution": hr_res,
        "num_images": len(l1_values),
        "l1_mean": float(np.mean(l1_values)) if l1_values else None,
        "l1_std": float(np.std(l1_values)) if l1_values else None,
    }


def main():
    parser = argparse.ArgumentParser(description="RETHINED Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--speed_only", action="store_true")
    parser.add_argument("--compare_hr_refiner", action="store_true")
    parser.add_argument("--upscale_test", action="store_true")
    parser.add_argument("--speed_runs", type=int, default=50)
    parser.add_argument("--speed_warmup", type=int, default=10)
    parser.add_argument("--upscale_runs", type=int, default=30)
    parser.add_argument("--upscale_warmup", type=int, default=5)
    parser.add_argument("--hr_resolutions", type=str, default=None,
                        help="Comma-separated HR resolutions for upscaling benchmark, e.g. 512,1024")
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    if device.type in {"cuda", "xpu"}:
        print(f"Accelerator: {get_device_name(device)}")

    model = load_model(args.checkpoint, cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    results = {"checkpoint": args.checkpoint, "params": total_params}

    # Speed benchmark (always run)
    native_lr_res = model.generator.image_size
    print("\n--- Speed Benchmark (base model) ---")
    speed = benchmark_speed(
        model,
        device,
        resolutions=(native_lr_res,),
        num_runs=args.speed_runs,
        warmup=args.speed_warmup,
    )
    results["speed"] = speed
    for res, data in speed.items():
        if "mean_ms" in data:
            print(f"  {res}x{res}: {data['mean_ms']:.2f}ms mean, {data['throughput_fps']:.1f} FPS")
        else:
            print(f"  {res}x{res}: {data}")

    # Upscaling speed benchmark
    print("\n--- AttentionUpscaling Speed ---")
    data_hr_res = cfg["data"]["image_size"]
    if args.hr_resolutions:
        hr_resolutions = tuple(int(x.strip()) for x in args.hr_resolutions.split(",") if x.strip())
    else:
        hr_resolutions = tuple(dict.fromkeys([data_hr_res, 1024, 2048]))
    upscale_speed = benchmark_upscaling(
        model,
        device,
        lr_res=native_lr_res,
        hr_resolutions=hr_resolutions,
        num_runs=args.upscale_runs,
        warmup=args.upscale_warmup,
    )
    results["upscaling_speed"] = upscale_speed
    for res, data in upscale_speed.items():
        if "mean_ms" in data:
            print(f"  LR={native_lr_res} -> HR={res}: {data['mean_ms']:.2f}ms (upscale only)")
        else:
            print(f"  LR={native_lr_res} -> HR={res}: {data}")

    if args.compare_hr_refiner and model.generator.hr_residual_refiner is not None:
        print("\n--- AttentionUpscaling Speed (HRResidualRefiner disabled) ---")
        upscale_speed_no_hr = benchmark_upscaling(
            model,
            device,
            lr_res=native_lr_res,
            hr_resolutions=hr_resolutions,
            disable_hr_residual_refiner=True,
            num_runs=args.upscale_runs,
            warmup=args.upscale_warmup,
        )
        results["upscaling_speed_no_hr_refiner"] = upscale_speed_no_hr
        for res, data in upscale_speed_no_hr.items():
            if "mean_ms" in data:
                full_ms = upscale_speed.get(res, {}).get("mean_ms")
                delta_suffix = ""
                if full_ms is not None:
                    delta_suffix = f", delta={full_ms - data['mean_ms']:.2f}ms"
                print(
                    f"  LR={native_lr_res} -> HR={res}: {data['mean_ms']:.2f}ms "
                    f"(upscale only, no HR refiner){delta_suffix}"
                )
            else:
                print(f"  LR={native_lr_res} -> HR={res}: {data}")

    if args.speed_only:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
        return

    # Quality metrics
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
    )
    quality = evaluate_quality(model, val_loader, device, num_images=args.num_images)
    results["quality"] = quality
    print(f"  {'Metric':<20} {'Coarse':>10} {'Refined':>10} {'Improvement':>12}")
    print(f"  {'-'*52}")
    for base in ["l1", "ssim", "lpips"]:
        c = quality[f"{base}_coarse"]["mean"]
        r = quality[f"{base}_refined"]["mean"]
        if base == "ssim":
            imp = f"+{(r-c)*100:.1f}%"
        elif base in ("l1", "lpips"):
            imp = f"-{(c-r)/c*100:.1f}%" if c > 0 else "N/A"
        else:
            imp = ""
        print(f"  {base.upper():<20} {c:>10.4f} {r:>10.4f} {imp:>12}")

    # Upscaling quality test
    if args.upscale_test or not args.speed_only:
        print("\n--- AttentionUpscaling Quality ---")
        for hr_res in (512, 1024):
            uq = test_upscaling_quality(model, val_loader, device, hr_res=hr_res, num_images=50)
            results[f"upscaling_quality_{hr_res}"] = uq
            if uq["l1_mean"] is not None:
                print(f"  HR={hr_res}: L1={uq['l1_mean']:.4f} (n={uq['num_images']})")
            else:
                print(f"  HR={hr_res}: Failed")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
