# SuperCAF-Only Training

## Prepare the manifest

```bash
python v6/scripts/prepare_supercaf_manifest.py \
  --supercaf-root datasets/supercaf \
  --output v6/dataset/manifests/supercaf_manifest.csv
```

## Train

```bash
cd v6
python train.py --config configs/train_supercaf_512.yaml
```

## Early dead-end signals

The V6 training loop now logs validation health metrics during training:

- `val/masked_l1_coarse`
- `val/masked_l1_refined`
- `val/refinement_gain_pct`
- `val/refined_better_rate`
- `val/valid_l1_coarse`
- `val/valid_l1_refined`
- `val/masked_delta_mean`

Interpretation:

- good sign: `masked_l1_refined < masked_l1_coarse`
- good sign: `refinement_gain_pct > 0`
- good sign: `refined_better_rate > 0.5`
- bad sign: refinement worsens masked L1 for many checkpoints in a row
- bad sign: valid-region L1 rises noticeably over coarse

Warnings are also written into `training_status.json` when refinement appears to
be a dead end.
