# DF8K Surrogate Dataset Plan

This plan creates a reproducible surrogate for the paper's DF8K-Inpainting
training set using the information that is actually available:

- Paper: DF8K-Inpainting has 2,850 images total
- Sources: DF2K + CAFHQ
- Content: no persons
- Splits: 70/20/10
- Masks: free-form, random for train/val, fixed for test
- Coverage: 30%-50%

## Target Counts

- Train: 1,995
- Val: 285
- Test: 570
- Total: 2,850

## Recommended Folder Layout

```text
datasets/
  df8k_surrogate/
    raw/
      df2k/
      cafhq/
    curated/
      images/
    manifests/
      master_images.csv
      excluded_images.csv
      candidate_review.csv
    splits/
      train.txt
      val.txt
      test.txt
    masks/
      test/
```

## Build Procedure

1. Download source datasets into `datasets/df8k_surrogate/raw/`.
2. Inventory all images into `manifests/master_images.csv`.
3. Remove images with people, portraits, or obvious face-centric composition.
4. Keep high-resolution landscape, architecture, object, and outdoor scenes.
5. Copy selected final images into `curated/images/`.
6. Freeze the curated count at 2,850 images.
7. Create deterministic splits:
   - first 1,995 to `train.txt`
   - next 285 to `val.txt`
   - final 570 to `test.txt`
8. Put fixed test masks into `masks/test/`.

## Selection Rules

Keep:

- outdoor scenes
- indoor scenes without people
- objects, products, furniture, architecture, nature
- images with clean high-resolution detail

Exclude:

- any visible person, face, or body part
- portraits or people-centric scenes
- collages, screenshots, heavy text overlays
- very low-information images or corrupt files

## Determinism

To keep the surrogate stable across runs:

- sort all candidate paths lexicographically
- record every exclusion in `excluded_images.csv`
- record any ambiguous manual review in `candidate_review.csv`
- never resample splits once `curated/images/` is frozen

## Masks

Use fixed masks for test and random masks for train/val.

The official repo thanks LaMa for mask generation, so the closest public
description is "LaMa-style irregular free-form masks". The local generator in
`v6/data/masks.py` is usable, but it is an approximation rather than proof of
exact parity with the authors' generator.

## Current Recommendation

For the first V6 training run:

- train on the DF8K surrogate if CAFHQ is available
- otherwise start on DF2K only with the same split logic
- keep the split manifests from day one so evaluation stays comparable
