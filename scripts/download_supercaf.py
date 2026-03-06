"""Download and extract the SuperCAF test set into the expected layout."""

import argparse
import shutil
import zipfile
from pathlib import Path

import gdown


SUPERCAF_URL = "https://drive.google.com/uc?id=1w4qwHhDb25r0RIWwPjgOp86G6SQnnBMO"


def find_dataset_root(extract_dir: Path) -> Path:
    for candidate in [extract_dir, *extract_dir.iterdir()]:
        if candidate.is_dir() and (candidate / "images").exists() and (candidate / "masks").exists():
            return candidate
    raise FileNotFoundError("Could not find extracted SuperCAF folder with images/ and masks/")


def main():
    parser = argparse.ArgumentParser(description="Download SuperCAF dataset")
    parser.add_argument("--output-root", type=Path, default=Path("datasets/supercaf"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset")
    args = parser.parse_args()

    output_root = args.output_root
    output_root.parent.mkdir(parents=True, exist_ok=True)

    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} already exists. Use --force to replace it.")
        shutil.rmtree(output_root)

    zip_path = output_root.parent / "SuperCAF_testset.zip"
    extract_dir = output_root.parent / "_supercaf_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SuperCAF to {zip_path} ...")
    gdown.download(SUPERCAF_URL, str(zip_path), quiet=False)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    dataset_root = find_dataset_root(extract_dir)
    shutil.move(str(dataset_root), str(output_root))

    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)

    print(f"SuperCAF ready at {output_root}")
    print(f"Images dir: {output_root / 'images'}")
    print(f"Masks dir: {output_root / 'masks'}")


if __name__ == "__main__":
    main()
