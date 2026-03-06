"""Build a SuperCAF-only manifest for V6 training."""

import argparse
import csv
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_images(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def split_counts(total: int) -> tuple[int, int, int]:
    train = int(total * 0.7)
    test = int(total * 0.2)
    val = total - train - test
    return train, test, val


def main():
    parser = argparse.ArgumentParser(description="Prepare SuperCAF-only manifest")
    parser.add_argument("--supercaf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir = args.supercaf_root / "images"
    masks_dir = args.supercaf_root / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("SuperCAF root must contain images/ and masks/ directories")

    samples = []
    for image_path in scan_images(images_dir):
        rel = image_path.relative_to(images_dir)
        mask_path = masks_dir / rel
        if not mask_path.exists():
            alt = mask_path.with_suffix(".png")
            if alt.exists():
                mask_path = alt
            else:
                raise FileNotFoundError(f"Missing mask for {image_path}")
        samples.append({
            "image_path": str(image_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "source": "supercaf",
        })

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    train_n, test_n, val_n = split_counts(len(samples))

    rows = []
    for idx, sample in enumerate(samples):
        if idx < train_n:
            split = "train"
        elif idx < train_n + test_n:
            split = "test"
        else:
            split = "val"
        rows.append({
            "split": split,
            **sample,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image_path", "mask_path", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manifest: {args.output}")
    print(f"Total samples: {len(samples)}")
    print(f"Splits: train={train_n}, test={test_n}, val={val_n}")


if __name__ == "__main__":
    main()
