"""Dataset loaders for V6 training.

Supports:
- plain image-folder datasets with generated masks
- manifest-driven mixed datasets with optional paired masks per sample
"""

import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from data.masks import FreeFormMaskGenerator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _scan_images(root_dir: Path) -> list[Path]:
    """Scan directory for image files, using cached file list if available."""
    cache_path = root_dir / ".filelist.txt"

    if cache_path.exists():
        paths = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    paths.append(Path(line))
        if paths and paths[0].exists():
            print(f"Loaded {len(paths):,} images from cache: {cache_path}")
            return paths
        print(f"Cache stale, rescanning...")

    print(f"Scanning {root_dir} for images...")
    t0 = time.time()
    paths = sorted([
        p for p in root_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    elapsed = time.time() - t0
    print(f"Found {len(paths):,} images in {elapsed:.1f}s")

    if paths:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                for p in paths:
                    f.write(f"{p}\n")
        except OSError:
            pass

    return paths


def _load_manifest(manifest_path: Path, split: str) -> list[dict]:
    """Load sample metadata from CSV manifest."""
    samples = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "image_path", "mask_path", "source"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        for row in reader:
            if row["split"] != split:
                continue
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"]) if row["mask_path"] else None
            samples.append({
                "image_path": image_path,
                "mask_path": mask_path,
                "source": row["source"],
            })
    return samples


class InpaintingDataset(Dataset):
    """Image dataset with either paired masks or generated masks."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        image_size: int = 512,
        split: str = "train",
        mask_min_coverage: float = 0.3,
        mask_max_coverage: float = 0.5,
        max_images: Optional[int] = None,
        val_dir: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ):
        self.image_size = image_size
        self.split = split
        self.manifest_path = Path(manifest_path) if manifest_path else None

        if self.manifest_path is not None:
            self.samples = _load_manifest(self.manifest_path, split)
            if not self.samples:
                raise FileNotFoundError(f"No samples found for split '{split}' in {self.manifest_path}")
        else:
            if root_dir is None:
                raise ValueError("root_dir is required when manifest_path is not provided")

            if val_dir and split in ("val", "test"):
                scan_dir = Path(val_dir)
            else:
                scan_dir = Path(root_dir)

            image_paths = _scan_images(scan_dir)
            if not image_paths:
                raise FileNotFoundError(f"No images found in {scan_dir}")

            if val_dir is None:
                n = len(image_paths)
                if split == "train":
                    image_paths = image_paths[:int(0.7 * n)]
                elif split == "test":
                    image_paths = image_paths[int(0.7 * n):int(0.9 * n)]
                elif split == "val":
                    image_paths = image_paths[int(0.9 * n):]

            self.samples = [{
                "image_path": p,
                "mask_path": None,
                "source": "generated",
            } for p in image_paths]

        if max_images is not None:
            self.samples = self.samples[:max_images]

        self.mask_gen = FreeFormMaskGenerator(
            image_size=image_size,
            min_coverage=mask_min_coverage,
            max_coverage=mask_max_coverage,
        )

    def _resize_if_needed(self, image: Image.Image, mask: Optional[Image.Image]):
        w, h = image.size
        if min(w, h) >= self.image_size:
            return image, mask

        scale = self.image_size / min(w, h)
        new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
        image = image.resize((new_w, new_h), Image.BICUBIC)
        if mask is not None:
            mask = mask.resize((new_w, new_h), Image.NEAREST)
        return image, mask

    def _crop_pair(self, image: Image.Image, mask: Optional[Image.Image]):
        image, mask = self._resize_if_needed(image, mask)
        w, h = image.size

        if self.split == "train":
            x = np.random.randint(0, w - self.image_size + 1)
            y = np.random.randint(0, h - self.image_size + 1)
        else:
            x = (w - self.image_size) // 2
            y = (h - self.image_size) // 2

        box = (x, y, x + self.image_size, y + self.image_size)
        image = image.crop(box)
        if mask is not None:
            mask = mask.crop(box)

        if self.split == "train" and np.random.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return image, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        mask_img = None
        if sample["mask_path"] is not None:
            mask_img = Image.open(sample["mask_path"]).convert("L")
            if mask_img.size != image.size:
                mask_img = mask_img.resize(image.size, Image.NEAREST)

        image, mask_img = self._crop_pair(image, mask_img)
        image = TF.to_tensor(image)

        if mask_img is not None:
            mask = TF.to_tensor(mask_img)
            mask = (mask > 0.5).float()
        else:
            mask_np = self.mask_gen()
            mask = torch.from_numpy(mask_np).unsqueeze(0)

        masked_image = image * (1 - mask)

        return {
            "image": image,
            "mask": mask,
            "masked_image": masked_image,
            "source": sample["source"],
        }


def get_dataloader(
    root_dir: Optional[str] = None,
    image_size: int = 512,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    max_images: Optional[int] = None,
    mask_min_coverage: float = 0.3,
    mask_max_coverage: float = 0.5,
    val_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
):
    """Create a DataLoader for inpainting training/evaluation."""
    dataset = InpaintingDataset(
        root_dir=root_dir,
        image_size=image_size,
        split=split,
        mask_min_coverage=mask_min_coverage,
        mask_max_coverage=mask_max_coverage,
        max_images=max_images,
        val_dir=val_dir,
        manifest_path=manifest_path,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": (split == "train"),
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": (split == "train"),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = (
            persistent_workers if persistent_workers is not None else True
        )
        loader_kwargs["prefetch_factor"] = (
            prefetch_factor if prefetch_factor is not None else 2
        )

    return torch.utils.data.DataLoader(**loader_kwargs)
