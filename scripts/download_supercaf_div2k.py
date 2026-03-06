"""Download SuperCAF and DIV2K into the expected local dataset layout.

SuperCAF is downloaded from Google Drive via gdown.
DIV2K is downloaded from Kaggle via the Kaggle API package.
"""

import argparse
import shutil
import zipfile
from pathlib import Path


SUPERCAF_URL = "https://drive.google.com/uc?id=1w4qwHhDb25r0RIWwPjgOp86G6SQnnBMO"
DIV2K_DATASET = "soumikrakshit/div2k-high-resolution-images"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_module(name: str, install_hint: str):
    try:
        return __import__(name)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency '{name}'. Install it with: {install_hint}") from exc


def normalize_supercaf_root(extract_root: Path, final_root: Path) -> None:
    candidates = [p for p in extract_root.rglob("*") if p.is_dir() and p.name.lower() == "images"]
    for images_dir in candidates:
        parent = images_dir.parent
        masks_dir = parent / "masks"
        if masks_dir.exists():
            if final_root.exists():
                shutil.rmtree(final_root)
            shutil.move(str(parent), str(final_root))
            return
    raise FileNotFoundError("Could not find extracted SuperCAF images/ and masks/ directories")


def download_supercaf(datasets_root: Path) -> None:
    gdown = require_module("gdown", "pip install gdown")
    zip_path = datasets_root / "SuperCAF_testset.zip"
    extract_root = datasets_root / "_supercaf_extract"
    final_root = datasets_root / "supercaf"

    if final_root.exists():
        print(f"SuperCAF already exists at {final_root}, skipping download")
        return

    ensure_dir(datasets_root)
    print("Downloading SuperCAF...")
    gdown.download(SUPERCAF_URL, str(zip_path), quiet=False, fuzzy=True)

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    print("Extracting SuperCAF...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    normalize_supercaf_root(extract_root, final_root)
    zip_path.unlink(missing_ok=True)
    shutil.rmtree(extract_root, ignore_errors=True)
    print(f"SuperCAF ready at {final_root}")


def download_div2k(datasets_root: Path) -> None:
    kaggle_mod = require_module("kaggle", "pip install kaggle")
    final_root = datasets_root / "div2k"

    if final_root.exists():
        print(f"DIV2K already exists at {final_root}, skipping download")
        return

    ensure_dir(final_root)
    print("Authenticating Kaggle API...")
    api = kaggle_mod.api
    api.authenticate()

    print(f"Downloading DIV2K dataset '{DIV2K_DATASET}'...")
    api.dataset_download_files(DIV2K_DATASET, path=str(final_root), unzip=True, quiet=False)
    print(f"DIV2K ready at {final_root}")


def main():
    parser = argparse.ArgumentParser(description="Download SuperCAF and DIV2K")
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--skip-supercaf", action="store_true")
    parser.add_argument("--skip-div2k", action="store_true")
    args = parser.parse_args()

    datasets_root = args.datasets_root.resolve()
    ensure_dir(datasets_root)

    if not args.skip_supercaf:
        download_supercaf(datasets_root)
    if not args.skip_div2k:
        download_div2k(datasets_root)

    print("Dataset download step complete.")


if __name__ == "__main__":
    main()
