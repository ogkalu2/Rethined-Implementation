# SuperCAF + DF2K Setup

## What to Download

For DF2K, download only the HR folders:

- `DF2K_train_HR`
- `DF2K_valid_HR`

Ignore the LR folders.

For SuperCAF, use the provided archive with `images/` and `masks/`.

## Suggested Paths

```text
datasets/
  supercaf/
    images/
    masks/
  df2k/
    DF2K_train_HR/
    DF2K_valid_HR/
```

## Build the Mixed Manifest

```bash
python v6/scripts/prepare_supercaf_df2k_manifest.py \
  --supercaf-root datasets/supercaf \
  --df2k-root datasets/df2k \
  --output v6/dataset/manifests/supercaf_df2k_manifest.csv
```

This will:

- keep all SuperCAF paired-mask samples
- randomly sample the remaining count from DF2K to reach 2,850 total
- create deterministic `train/val/test` splits with seed 42

## Train

```bash
cd v6
python train.py --config configs/train_supercaf_df2k_512.yaml
```
