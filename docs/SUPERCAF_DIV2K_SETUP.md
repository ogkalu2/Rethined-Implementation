# SuperCAF + DIV2K Setup

This path avoids the larger DF2K download and uses all DIV2K images directly.

## Download

From the repo root:

```bash
python v6/scripts/download_supercaf_div2k.py
```

Requirements:

- `gdown` for SuperCAF
- `kaggle` for DIV2K
- Kaggle API credentials configured for the Kaggle download

The script writes to:

```text
datasets/
  supercaf/
    images/
    masks/
  div2k/
```

## Build the Mixed Manifest

```bash
python v6/scripts/prepare_supercaf_div2k_manifest.py \
  --supercaf-root datasets/supercaf \
  --div2k-root datasets/div2k \
  --output v6/dataset/manifests/supercaf_div2k_manifest.csv
```

## Train

```bash
cd v6
python train.py --config configs/train_supercaf_div2k_512.yaml
```
