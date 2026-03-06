"""Build a mixed SuperCAF + DF2K manifest for V6 training.

SuperCAF samples keep their paired masks.
DF2K samples use generated masks at training time.
"""

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


def paired_supercaf_samples(supercaf_root: Path) -> list[dict]:
    images_dir = supercaf_root / "images"
    masks_dir = supercaf_root / "masks"
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
                raise FileNotFoundError(f"Missing SuperCAF mask for {image_path}")
        samples.append({
            "image_path": image_path.resolve(),
            "mask_path": mask_path.resolve(),
            "source": "supercaf",
        })
    return samples


def df2k_samples(df2k_root: Path) -> list[dict]:
    return [{
        "image_path": p.resolve(),
        "mask_path": "",
        "source": "df2k",
    } for p in scan_images(df2k_root)]


def main():
    parser = argparse.ArgumentParser(description="Prepare SuperCAF + DF2K manifest")
    parser.add_argument("--supercaf-root", type=Path, required=True)
    parser.add_argument("--df2k-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-total", type=int, default=2850)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    supercaf = paired_supercaf_samples(args.supercaf_root)
    needed_df2k = max(args.target_total - len(supercaf), 0)

    df2k = df2k_samples(args.df2k_root)
    if needed_df2k > len(df2k):
        raise ValueError(f"Requested {needed_df2k} DF2K samples but only found {len(df2k)}")

    rng.shuffle(df2k)
    selected = supercaf + df2k[:needed_df2k]
    rng.shuffle(selected)

    train_n, test_n, val_n = split_counts(len(selected))
    rows = []
    for idx, sample in enumerate(selected):
        if idx < train_n:
            split = "train"
        elif idx < train_n + test_n:
            split = "test"
        else:
            split = "val"
        rows.append({
            "split": split,
            "image_path": str(sample["image_path"]),
            "mask_path": str(sample["mask_path"]),
            "source": sample["source"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image_path", "mask_path", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manifest: {args.output}")
    print(f"SuperCAF samples: {len(supercaf)}")
    print(f"DF2K samples: {needed_df2k}")
    print(f"Total samples: {len(selected)}")
    print(f"Splits: train={train_n}, test={test_n}, val={val_n}")


if __name__ == "__main__":
    main()
