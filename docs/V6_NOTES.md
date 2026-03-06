# V6 Notes

V6 continues from the `prod_v3` codebase and fixes the issues that made the
refinement branch destructive:

1. Patch masks now use `any` semantics instead of sampling a single pixel.
2. Valid regions keep full patches, not high-frequency-only patches.
3. Attention applies `k_mask`, so masked patches are not used as source patches.
4. The attention residual connection is enabled.
5. Refinement reconstructs patches as `LF + refined_HF` instead of subtracting LF.
6. VGG perceptual/style losses now backpropagate through the prediction branch.
7. Style loss is disabled by default; masked L1 + perceptual are the default training guess.

This is an engineering continuation, not a claim of exact paper reproduction.
