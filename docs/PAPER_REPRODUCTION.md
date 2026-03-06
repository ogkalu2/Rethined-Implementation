# RETHINED Prod V3 — Exact Paper Reproduction

## Goal
Reproduce the paper's results exactly by matching the official GitHub code (`github.com/CrisalixSA/rethined`) and paper (arXiv:2503.14757v1, WACV 2025 Oral).

## Paper Reference
- **Title**: RETHINED: A New Benchmark and Baseline for Real-Time High-Resolution Image Inpainting On Edge Devices
- **Venue**: WACV 2025 (Oral)
- **arXiv**: 2503.14757v1 (March 2025)
- **GitHub**: https://github.com/CrisalixSA/rethined
- **Project Page**: https://crisalixsa.github.io/rethined/
- **Authors**: Marcelo Sanchez (Crisalix), Gil Triginer (Crisalix), Ignacio Sarasua (NVIDIA), Lara Raad (UdelaR), Coloma Ballester (UPF)
- **Note**: Author confirmed no training code or pretrained weights will be released (GitHub Issue #5, Jan 2026)

### Paper Target Metrics (Table 1)

| Dataset | Resolution | LPIPS | L1 | FID | SSIM | Time (ms) |
|---------|-----------|-------|------|------|------|-----------|
| CelebA-HQ | 1024x1024 | 0.032 | 0.003 | 12.954 | 0.970 | - |
| DIV8K | 1024x1024 | 0.030 | 0.026 | 12.717 | 0.987 | 17.59 |
| DIV8K | 2048x2048 | 0.031 | 0.031 | 12.5 | 0.975 | 34.33 |
| DIV8K | 4096x4096 | 0.031 | 0.029 | 13.37 | 0.971 | 39.43 |

### Parameter Counts (Tables 3-5, INCONSISTENT in paper)

| Source | Params | Notes |
|--------|--------|-------|
| Table 3 | **4.3M** | Main comparison table (iPad Pro benchmark) |
| Table 4 (P=8) | **16.9M** | Patch size ablation |
| Table 5 (dk=2048, feat.cond.) | **18.3M** | Embedding dim ablation (best config) |

The 4.3M likely refers to post-reparameterization or coarse-only. The 18.3M is the full model with dk=2048 + feature conditioning, which is the configuration that achieves the best results.

---

## 1. Architecture — Complete Specification

### 1.1 Coarse Encoder: MobileOne-S4

**Source**: Paper Section 3.2 references [47] (MobileOne, CVPR 2023). Official code confirms `mobileone(variant='s4')`.

```python
from mobileone import mobileone
encoder = mobileone(variant='s4')
```

**MobileOne-S4 Stage Specification** (from Apple ml-mobileone):

| Stage | Base Ch | Width Mult | Output Channels | Num Blocks | Stride | SE Blocks |
|-------|---------|-----------|-----------------|-----------|--------|-----------|
| stage0 | 64 | 1.0 | **64** | 1 | 2x | 0 |
| stage1 | 64 | 3.0 | **192** | 2 | 2x | 0 |
| stage2 | 128 | 3.5 | **448** | 8 | 2x | 0 |
| stage3 | 256 | 3.5 | **896** | 10 | 2x | 5 (last 5 blocks) |
| stage4 | 512 | 4.0 | **2048** | 1 | 2x | 1 |

- Total downsampling: 32x (5 stages, each stride 2)
- Convolutions: 3x3 depthwise + 1x1 pointwise + BatchNorm (Paper Sec 3.2)
- Reparameterizable at inference: multi-branch conv+BN → single conv (Paper Sec 3.5)
- Feature list: `[x0, x1, x2, x3, x4]` with channels `[64, 192, 448, 896, 2048]`

**Spatial dimensions at 512x512 input:**
| Stage | Spatial Size | Channels |
|-------|-------------|----------|
| Input | 512x512 | 3 |
| stage0 | 256x256 | 64 |
| stage1 | 128x128 | 192 |
| stage2 | 64x64 | 448 |
| stage3 | 32x32 | 896 |
| stage4 | 16x16 | 2048 |

### 1.2 Coarse Decoder

**Source**: Official GitHub `model.py`, class `MobileOneCoarse`

The official code defines the decoder as:
```python
self.d4 = nn.ConvTranspose2d(2048, 1792, kernel_size=4, stride=2, padding=1)
self.d3 = nn.ConvTranspose2d(1792 + 1792, 896, kernel_size=4, stride=2, padding=1)
self.d2 = nn.ConvTranspose2d(896 + 896, 384, kernel_size=4, stride=2, padding=1)
self.d1 = nn.ConvTranspose2d(384 + 384, 64, kernel_size=4, stride=2, padding=1)
self.d0 = nn.ConvTranspose2d(64 + 64, 3, kernel_size=4, stride=2, padding=1)
```

**KNOWN BUG in official code**: The skip connection input channels DON'T match MobileOne-S4 outputs.

| Layer | Official In Channels | Skip Source | Skip Channels (S4) | Actual Cat | Official Expects | MISMATCH |
|-------|---------------------|-------------|-------------------|-----------|-----------------|----------|
| d4 | 2048 | — | — | 2048 | 2048 | OK |
| d3 | 3584 (1792+1792) | x3 (stage3) | 896 | 1792+896=**2688** | 3584 | **YES** |
| d2 | 1792 (896+896) | x2 (stage2) | 448 | 896+448=**1344** | 1792 | **YES** |
| d1 | 768 (384+384) | x1 (stage1) | 192 | 384+192=**576** | 768 | **YES** |
| d0 | 128 (64+64) | x0 (stage0) | 64 | 64+64=**128** | 128 | OK |

The official code would **crash at runtime** with standard MobileOne-S4. This means either:
1. The authors use a modified MobileOne with different channel widths
2. The released code is intentionally obfuscated/incomplete
3. Only d0 (correct) represents the actual architecture

**Our V3 Implementation**: Use decoder channels that match standard MobileOne-S4 skip connections:

```python
# CORRECTED decoder matching MobileOne-S4 encoder channels
self.d4 = nn.ConvTranspose2d(2048, 896, kernel_size=4, stride=2, padding=1)   # 2048→896
self.d3 = nn.ConvTranspose2d(896+896, 448, kernel_size=4, stride=2, padding=1) # 1792→448, cat(896,x3=896)
self.d2 = nn.ConvTranspose2d(448+448, 192, kernel_size=4, stride=2, padding=1) # 896→192, cat(448,x2=448)
self.d1 = nn.ConvTranspose2d(192+192, 64, kernel_size=4, stride=2, padding=1)  # 384→64, cat(192,x1=192)
self.d0 = nn.ConvTranspose2d(64+64, 3, kernel_size=4, stride=2, padding=1)     # 128→3, cat(64,x0=64)
```

**Forward pass** (from official code):
```python
def forward(self, x):
    features = []
    x0 = self.model.stage0(x);  features.append(x0)   # 256x256, 64ch
    x1 = self.model.stage1(x0); features.append(x1)    # 128x128, 192ch
    x2 = self.model.stage2(x1); features.append(x2)    # 64x64, 448ch
    x3 = self.model.stage3(x2); features.append(x3)    # 32x32, 896ch
    x4 = self.model.stage4(x3); features.append(x4)    # 16x16, 2048ch

    out = self.relu(self.d4(x4))          # 32x32
    out = torch.cat([out, x3], dim=1)     # cat with stage3
    out = self.relu(self.d3(out))          # 64x64
    out = torch.cat([out, x2], dim=1)     # cat with stage2
    out = self.relu(self.d2(out))          # 128x128
    out = torch.cat([out, x1], dim=1)     # cat with stage1
    out = self.relu(self.d1(out))          # 256x256
    out = torch.cat([out, x0], dim=1)     # cat with stage0
    out = self.sigmoid(self.d0(out))       # 512x512, 3ch

    return out, features  # returns (coarse_output, encoder_features_list)
```

- Activations: **ReLU** on all decoder layers, **Sigmoid** on final output (confirmed in code)
- Output range: **[0, 1]** (Sigmoid)

### 1.3 NeuralPatchMatch Refinement Module (g_phi)

**Source**: Official GitHub `model.py`, class `PatchInpainting`

#### Default Configuration (from official `__main__` block):

```python
config = {
    'generator': {
        'params': {
            'kernel_size': 8,          # P = patch size
            'nheads': 1,               # single attention head
            'stem_out_stride': 1,      # no output stride
            'stem_out_channels': 3,    # RGB output
            'merge_mode': 'all',       # sum merge
            'image_size': 512,         # operating resolution
            'embed_dim': 576,          # d_qk projection dim (NOTE: not 2048)
            'use_qpos': None,          # positional encoding OFF
            'use_kpos': None,          # positional encoding OFF
            'dropout': 0.1,            # attention dropout
            'feature_i': 2,            # feature conditioning from features[2] (stage2)
            'concat_features': True,   # concatenate encoder features to patches
            'final_conv': True,        # coherence layer ON
            'feature_dim': 896,        # expected feature channel dim
            'attention_type': 'MultiHeadAttention',
            'compute_v': False,        # no V computation
            'use_argmax': False,       # soft attention (not hard)
        }
    }
}
```

#### KNOWN INCONSISTENCY: feature_i vs feature_dim

The config says `feature_i=2` (stage2 = 448 channels) but `feature_dim=896` (stage3 channels). This means:
- `features[2]` = stage2 output = **448 channels**
- But the MHA input dim is computed as `stem_out_channels*kernel_size**2 + feature_dim = 3*64 + 896 = 1088`
- After F.interpolate (spatial only) and torch.cat, actual channels = `192 + 448 = 640`, NOT 1088
- This would cause a **runtime dimension mismatch** in the Linear layers

**Our V3 Options** (choose ONE):
- **Option A** (recommended): `feature_i=3, feature_dim=896` — stage3 outputs 896ch, dims match
- **Option B**: `feature_i=2, feature_dim=448` — respects code's stage choice, fixes dim

We use **Option A** (`feature_i=3, feature_dim=896`) because it is dimensionally correct AND matches our validated prod_v1 implementation.

#### embed_dim: Code=576, Paper Best=2048

The official code uses `embed_dim=576`, but the paper's Table 5 shows `dk=2048` achieves the best results (FID=12.5, LPIPS=0.031). The code likely represents a fast deployment config, while the paper reports the quality-optimal config.

**Our V3**: Use `embed_dim=2048` to match paper's best results.

#### 1.3.1 Patch Extraction (img2col via F.conv2d)

The official code uses a CoreML-compatible img2col via grouped convolution with identity kernels:

```python
def _compute_unfolding_weights(self, kernel_size, channels):
    weights = torch.eye(kernel_size * kernel_size).reshape(
        kernel_size * kernel_size, 1, kernel_size, kernel_size)
    weights = weights.repeat(channels, 1, 1, 1)
    return weights

def unfolding_coreml(self, feature_map, weights, kernel_size):
    batch_size, in_channels, img_h, img_w = feature_map.shape
    patches = F.conv2d(feature_map, weights, bias=None,
                       stride=(kernel_size, kernel_size),
                       padding=0, dilation=1, groups=in_channels)
    return patches, (img_h, img_w)
```

- Three sets of unfolding weights registered as buffers (non-persistent):
  - `unfolding_weights`: for stem_out_channels (3ch RGB)
  - `unfolding_weights_image`: for 3ch image
  - `unfolding_weights_mask`: for 1ch mask
- At 512x512 with P=8: produces `N = (512/8)^2 = 4096` patches
- Each patch: `3 * 8 * 8 = 192` values (flattened)

#### 1.3.2 High-Frequency Extraction

```python
self.final_gaussian_blur = GaussianBlur2d((7,7), sigma=(2.01, 2.01), separable=False)
```

Forward:
```python
image_blurred = self.final_gaussian_blur(image)
image_as_patches_blurred, _ = self.unfolding_coreml(image_blurred, self.unfolding_weights, self.kernel_size)
image_as_patches, sizes = self.unfolding_coreml(image, self.unfolding_weights, self.kernel_size)
image_as_patches = image_as_patches - image_as_patches_blurred  # HF = patches - blurred_patches
```

- **Gaussian kernel**: (7, 7)
- **Sigma**: (2.01, 2.01)
- **Separable**: False
- **Library**: kornia.filters.GaussianBlur2d
- Paper Eq(3): `p_HF_i = p_i - p^sigma_i`

#### 1.3.3 Mask Compositing

Before refinement, coarse output is composited with original:
```python
if self.mask_inpainting:
    image = image_coarse_inpainting * mask + image * (1 - mask)
```
Where `mask=1` in corrupted regions. This means: use coarse prediction where corrupted, keep original where uncorrupted.

Patch-level mask:
```python
mask_same_res_as_features_pooled, _ = self.unfolding_coreml(mask, self.unfolding_weights_mask, self.kernel_size)
mask_same_res_as_features_pooled = mask_same_res_as_features_pooled[:, 0:1, :, :]  # take first channel only
mask_same_res_as_features_pooled = mask_same_res_as_features_pooled.flatten(start_dim=2).unsqueeze(-1)
```
This gives a per-patch binary mask indicating if **any pixel** in that patch is corrupted.

#### 1.3.4 Feature Conditioning

```python
if self.concat_features:
    features_to_concat = features[self.feature_i]  # features[2] in official code
    features_to_concat = F.interpolate(features_to_concat,
                                        size=image_as_patches.shape[-2:],
                                        mode='bilinear', align_corners=False)
    input_attn = torch.cat([image_as_patches, features_to_concat], dim=1)
    input_attn = input_attn.flatten(start_dim=2).transpose(1, 2)
```

- Interpolation: **bilinear** with `align_corners=False`
- Concatenation along channel dimension
- Result shape: `(B, N, 192 + feature_dim)`

#### 1.3.5 Self-Attention Masking (ADDITIVE, despite paper saying multiplicative)

**Paper (Section 3.3)** describes multiplicative binary masking: `M_T = A * M_D` (element-wise product after softmax).

**Official code** implements **additive masking before softmax**:

```python
# Identity matrix buffer (registered at init)
self.register_buffer('qk_mask', 1e4 * torch.eye(N).unsqueeze(0).unsqueeze(0))

# In forward():
qk_mask = -1e4 * self.qk_mask.repeat(B, 1, 1, 1) + \
          2e4 * ((1 - mask_same_res_as_features_pooled) * self.qk_mask)
k_mask = -1e4 * mask_same_res_as_features_pooled  # COMPUTED BUT NEVER USED
```

Breaking down the `qk_mask` formula:
- `self.qk_mask` = `1e4 * I` (identity matrix scaled by 1e4)
- Term 1: `-1e4 * (1e4 * I)` = diagonal of `-1e8` (but this seems wrong, let me recheck)

Actually, `self.qk_mask` stores `1e4 * I`. Then:
- `-1e4 * self.qk_mask` = diagonal entries get `-1e4 * 1e4 = -1e8`...

Wait, `self.qk_mask.repeat(B,1,1,1)` just repeats along batch dim, the values are still `1e4` on diagonal, `0` elsewhere.

So: `qk_mask = -1e4 * qk_mask_eye + 2e4 * ((1-mask) * qk_mask_eye)`

For **unmasked patches** (mask=0): `-1e4 * 1e4 + 2e4 * (1 * 1e4)` = `-1e8 + 2e8` = `+1e8` on diagonal → strong self-attention
For **masked patches** (mask=1): `-1e4 * 1e4 + 2e4 * (0 * 1e4)` = `-1e8 + 0` = `-1e8` on diagonal → suppress self-attention

Wait, that's scaled by the qk_mask values. Let me re-read:

```python
self.qk_mask = 1e4 * torch.eye(N)  # shape (1, 1, N, N), diagonal = 1e4
```

Then in forward:
```python
qk_mask = -1e4 * self.qk_mask + 2e4 * ((1 - mask) * self.qk_mask)
```

Expanding on diagonal entries (off-diagonal = 0 throughout):
- Diagonal value of `self.qk_mask` = 1e4
- Term 1: `-1e4 * 1e4` = this is scalar mult... no wait.

`-1e4` is a scalar, `self.qk_mask` has values 1e4 on diagonal. So:
- Term 1 diagonal: `-1e4 * 1e4` = **-1e8**? No! `-1e4` is just a Python float multiplied element-wise:
  - `-1e4 * 1e4` = `-1e8`

Hmm, that's very large. Let me re-examine:

Actually `self.qk_mask` = `1e4 * torch.eye(N)` means the diagonal values are `10000.0`. Then:

For an **unmasked** patch i (mask_i = 0):
- `qk_mask[i,i] = -1e4 * 1e4 + 2e4 * (1-0) * 1e4 = -1e8 + 2e8 = +1e8`

Hmm, +1e8 before softmax makes that token attend ONLY to itself. For a **masked** patch:
- `qk_mask[i,i] = -1e4 * 1e4 + 2e4 * (1-1) * 1e4 = -1e8 + 0 = -1e8`

-1e8 before softmax suppresses self-attention for masked patches, forcing them to attend to other patches.

**Net effect** (identical to paper's semantic intent):
- Unmasked patches → attend to themselves (preserved unchanged)
- Masked patches → attend to OTHER patches (especially unmasked ones)
- Off-diagonal entries remain 0 (no explicit suppression of attending to masked patches from off-diagonal)

**k_mask**: Computed as `-1e4 * mask` but **NEVER APPLIED** in MultiHeadAttention.forward(). It is passed as a parameter but the function ignores it. This is a **bug in the official code** — the k_mask was intended to suppress attention TO masked patches (columns), but was never used.

**Our V3**: Implement the SAME additive masking as the official code (not the paper's multiplicative description), including the k_mask bug (compute but don't use).

#### 1.3.6 Multi-Head Attention

**Source**: Official GitHub `model.py`, class `MultiHeadAttention`

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_v, n_head, split, dropout, d_qk, compute_v, use_argmax=False):
        self.w_qs = nn.Linear(embed_dim, n_head * d_qk, bias=False)  # Q projection
        self.w_ks = nn.Linear(embed_dim, n_head * d_qk, bias=False)  # K projection
        self.w_vs = nn.Linear(d_v, n_head * d_v, bias=False)         # V projection
        self.fc   = nn.Linear(n_head * d_v, d_v, bias=False)         # output projection
        self.dropout = nn.Dropout(dropout)
```

**Dimension analysis** (with our corrected config: feature_i=3, feature_dim=896, embed_dim=2048):

| Projection | Input Dim | Output Dim | Parameters |
|-----------|-----------|-----------|-----------|
| w_qs (Q) | embed_dim=1088 | n_head * d_qk = 1 * 2048 | Linear(1088, 2048, bias=False) |
| w_ks (K) | embed_dim=1088 | n_head * d_qk = 1 * 2048 | Linear(1088, 2048, bias=False) |
| w_vs (V) | d_v=192 | n_head * d_v = 1 * 192 | Linear(192, 192, bias=False) |
| fc (out) | n_head * d_v = 192 | d_v=192 | Linear(192, 192, bias=False) |

Where:
- `embed_dim` (MHA input) = `stem_out_channels * kernel_size^2 + feature_dim` = `3*64 + 896 = 1088`
- `d_v` = `stem_out_channels * kernel_size^2` = `3*64 = 192` (raw HF patch tokens)
- `d_qk` = `embed_dim` config param = `2048` (for paper best) or `576` (official code default)
- `n_head` = `1`

**Forward pass**:
```python
def forward(self, q, k, v, qpos, kpos, qk_mask=None, k_mask=None):
    q = self.w_qs(q).view(B, N, n_head, d_k).transpose(1, 2)   # (B, 1, N, 2048)
    k = self.w_ks(k).view(B, N, n_head, d_k).transpose(1, 2)   # (B, 1, N, 2048)
    v = self.w_vs(v).view(B, N, n_head, d_v).transpose(1, 2)    # (B, 1, N, 192)

    attn = torch.matmul(q / self.d_k**0.5, k.transpose(2, 3))   # (B, 1, N, N)

    if qk_mask is not None:
        attn += qk_mask                                          # additive masking

    attn = F.softmax(attn, dim=-1)                               # NOTE: no float32 cast in official code
    attn = self.dropout(attn)

    output = torch.matmul(attn, v)                               # (B, 1, N, 192)
    output = output.transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, 192)
    output = self.dropout(self.fc(output))                        # (B, N, 192)
    # output += residual                                          # COMMENTED OUT in official code

    return output, attn
```

**Key observations from official code**:
1. **No float32 cast** on softmax (our V3 should add this for FP16 safety)
2. **Residual connection is COMMENTED OUT** (`# output += residual`)
3. **k_mask is accepted but never used** (bug)
4. **Dropout applied twice**: once after softmax, once after output projection
5. **Q and K use the SAME input** (self-attention): `self.multihead_attention(input_attn, input_attn, ...)`
6. **V uses raw HF patches** (not the feature-concatenated input): `image_as_patches`
7. **All Linear layers have `bias=False`**
8. **use_argmax=False**: soft attention (hard attention would scatter argmax indices)

#### 1.3.7 Post-Attention Processing

```python
# Subtract blurred patches from attention output (additional HF extraction)
out = out - image_as_patches_blurred.flatten(start_dim=2).transpose(1, 2)

# Apply mask: only modify corrupted patches, keep uncorrupted unchanged
mask = mask_same_res_as_features_pooled.squeeze(1).squeeze(-1).unsqueeze(-1)  # (B, N, 1)
out = out * mask + image_as_patches * (1 - mask)
```

#### 1.3.8 Coherence Layer (final_conv)

```python
self.final_conv = nn.Sequential(
    nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
    nn.Sigmoid()
)
```

Applied in patch-grid space:
```python
# In folding_coreml, when use_final_conv=True:
patches = rearrange(patches, 'b (p1 p2) c -> b c p1 p2', p1=window_size, p2=window_size)
patches = self.final_conv(patches)
patches = rearrange(patches, 'b c p1 p2 -> b (p1 p2) c')
```

- Input/output channels: `192` (= 3 * P^2 = 3 * 64)
- Kernel: **3x3**
- Padding: **1, reflect** mode
- Activation: **Sigmoid** (gates the patch features)
- Operates on the `(window_size x window_size)` patch grid (64x64 at 512px)

#### 1.3.9 Patch Reassembly

Uses `einops.rearrange` (NOT torch PixelShuffle despite defining it):
```python
final_image = rearrange(patches, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                h=output_size[0]//kernel_size, w=output_size[1]//kernel_size,
                p1=kernel_size, p2=kernel_size)
```
This converts `(B, N, 3*P^2)` → `(B, 3, H, W)`

#### 1.3.10 Positional Encoding

The official code config sets `use_qpos=None, use_kpos=None`, which means:
```python
self.positionalencoding = None  # when use_kpos or use_qpos is falsy
```
**Positional encoding is DISABLED** in the default configuration.

A `PositionalEncoding` class is defined (sinusoidal) but never instantiated with the default config.

### 1.4 Attention Upscaling (inference only)

**Source**: Official GitHub `model.py`, class `AttentionUpscaling`

Process:
1. Bicubic upsample LR inpainted image to HR: `F.interpolate(x_lr, size=(hr_h, hr_w), mode='bicubic', align_corners=False)`
2. Compute HR patch size: `hr_patch_size = P * (hr_h // lr_h)` (e.g., 8 * 2 = 16 for 2x)
3. Extract HR patches using same img2col with larger kernel
4. Compute HR HF patches: `hr_hf = hr_patches - blur(hr_patches)`
5. Apply LR attention map to HR HF patches: `torch.matmul(attn_map.squeeze(1), hr_hf_patches)`
6. Fold back: `folding_coreml(reconstructed, (hr_h, hr_w), hr_patch_size, use_final_conv=False)`
   - **Note**: `use_final_conv=False` — coherence layer is NOT applied at HR
7. Add to bicubic upscale: `final = x_hr_base + reconstructed_hr_hf`

### 1.5 Reparameterization (Paper Section 3.5)

- Fuse BatchNorm into Conv: `W_hat = W * gamma / sigma`
- Fuse skip connections: 1x1 identity conv padded by `S-1` zeros, then merged
- Applied to coarse encoder only (MobileOne reparameterization)
- References: RepVGG [14], MobileFormer [8]

---

## 2. Dataset — DF8K-Inpainting

**Source**: Paper Section 4

### 2.1 Composition
- **Total images**: 2,850
- **Sources**:
  - **DF2K**: DIV2K [45] + Flickr2K [27] merged
  - **CAFHQ**: From [58]
- **Content**: Outdoor scenes, wide variety of entities from human-made objects to nature landscapes
- **Restriction**: **No persons** in any image
- **Resolutions**: 2K, 4K, and 8K (mixed)
- **Format**: PNG (lossless, no JPEG artifacts)

### 2.2 Split
- **Train**: 70% → ~1,995 images
- **Test**: 20% → ~570 images (with fixed test masks)
- **Validation**: 10% → ~285 images

### 2.3 Mask Generation
- **Style**: Irregular free-form masks, similar to LaMa [43] (Suvorov et al., WACV 2022)
- **Training**: Random mask generated at each iteration
- **Testing**: Fixed set of test masks (for reproducibility)
- **Coverage**: 30%-50% of image area

### 2.4 Available Sources

| Dataset | Images | Resolution | Source |
|---------|--------|-----------|--------|
| DF2K (DIV2K + Flickr2K) | 3,450 | 2K | https://www.kaggle.com/datasets/anvu1204/df2kdata/data |
| CAFHQ | ~400-500? | 2K-8K | Part of DF8K-Inpainting (may need to be sourced separately) |
| DIV8K | 1,500 | 8K | NTIRE challenge dataset |
| LSDIR | 84,991 | HQ PNG | https://data.vision.ee.ethz.ch/yawli/index.html |

**Note**: The exact CAFHQ composition and where to download it is unclear. The paper says DF8K-Inpainting = DF2K + CAFHQ = 2,850 images. Since DF2K alone has 3,450, this suggests the 2,850 is a curated subset of DF2K+CAFHQ combined.

---

## 3. Training Recipe

**Source**: Paper Section 5.1

### 3.1 Confirmed Settings

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | **Adam** | Paper Sec 5.1 |
| Learning rate | **0.001** (10^-3) | Paper Sec 5.1 |
| LR schedule | **Warmup + progressive cosine decay** | Paper Sec 5.1 |
| Total steps | **600,000** | Paper Sec 5.1 |
| Effective batch size | **128** | Paper Sec 5.1 |
| Hardware | **NVIDIA RTX 4090** | Paper Sec 5.1 |
| Training mode | **Joint single-stage** (end-to-end) | Paper Sec 5.1 ("jointly in a single stage") |
| Training resolution | **512x512** | Official code `image_size=512` |
| Input masking | Masked input: `x = y * m` | Paper Sec 3.1 |

### 3.2 Unknown Settings (deferred to supplementary, unavailable)

| Parameter | Our Best Guess | Reasoning |
|-----------|---------------|-----------|
| Adam betas | (0.9, 0.999) | PyTorch default, standard for inpainting |
| Adam weight_decay | 0.0 | PyTorch default |
| Adam epsilon | 1e-8 | PyTorch default |
| Warmup steps | 1,000 | Common choice, our prod_v1 used this |
| Min LR (cosine) | 1e-6 | Standard for cosine decay |
| Gradient clipping | 1.0 | Standard for attention models |
| Mixed precision | FP16 | Implied by mobile focus, code uses fp16 mention |
| Data augmentation | None | Not mentioned in paper |
| Gradient accumulation | 32 (with batch=4) | 128 effective / 4 per-GPU = 32 accum |

### 3.3 Loss Functions (COMPLETELY UNSPECIFIED in paper)

The paper says: "More information about training recipe can be found in supplementary."
No supplementary material is available. No loss code is in the official repo.

**Our best guess (from standard inpainting literature + our prod_v1)**:

| Loss | Weight | Details |
|------|--------|---------|
| **L1 (hole)** | 6.0 | L1 on masked region only |
| **L1 (valid)** | 1.0 | L1 on unmasked region |
| **Perceptual (VGG19)** | 0.1 | Features from relu1_2, relu2_2, relu3_2, relu4_2 |
| **Style (Gram)** | 250.0 | Gram matrix from same VGG layers |
| **Adversarial** | **0.0** | Paper does NOT use GAN |

**VGG normalization**: Standard practice is to normalize inputs with ImageNet mean/std before VGG feature extraction. Our prod_v1 was missing this. V3 should add it:
```python
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
x_normalized = (x - mean) / std
```

---

## 4. Complete Configuration for V3

### 4.1 Model Config (matching paper's best: dk=2048, P=8, feat.cond.=ON)

```yaml
model:
  coarse_model:
    variant: "s4"                    # MobileOne-S4, confirmed in official code
  generator:
    kernel_size: 8                   # P=8, Paper Table 4 best
    nheads: 1                        # Single head, official code
    stem_out_stride: 1               # Official code
    stem_out_channels: 3             # RGB, official code
    merge_mode: "all"                # Official code
    image_size: 512                  # Official code __main__
    embed_dim: 2048                  # Paper Table 5 best (code=576)
    use_qpos: null                   # Positional encoding OFF, official code
    use_kpos: null                   # Positional encoding OFF, official code
    dropout: 0.1                     # Official code
    feature_i: 3                     # stage3=896ch (code says 2 but dims don't match)
    concat_features: true            # Official code
    final_conv: true                 # Coherence layer ON, official code
    feature_dim: 896                 # Official code
    attention_type: "MultiHeadAttention"
    compute_v: false                 # Official code
    use_argmax: false                # Soft attention, official code
    attention_masking: true          # Self-attention masking ON (default)
    mask_inpainting: true            # Mask compositing ON (default)
```

### 4.2 Training Config

```yaml
data:
  root_dir: "datasets/df8k/train"
  val_dir: "datasets/df8k/val"
  image_size: 512                    # Training at 512x512
  batch_size: 4
  num_workers: 4
  mask_min_coverage: 0.30
  mask_max_coverage: 0.50

training:
  total_steps: 600000
  grad_accum_steps: 32               # 4 * 32 = 128 effective batch
  lr: 0.001                          # Paper: 10^-3
  warmup_steps: 1000                 # Best guess (not in paper)
  min_lr: 0.000001                   # Best guess
  grad_clip: 1.0                     # Best guess
  mixed_precision: true
  seed: 42

loss:
  l1_hole_weight: 6.0               # Best guess (not in paper)
  l1_valid_weight: 1.0              # Best guess
  perceptual_weight: 0.1            # Best guess
  style_weight: 250.0               # Best guess
  vgg_normalize: true               # Add ImageNet normalization

logging:
  log_dir: "logs/train_df8k_512"
  log_interval: 100
  vis_interval: 5000
  eval_interval: 10000
  save_interval: 50000
  keep_last_checkpoints: 5
```

---

## 5. Differences: Official Code vs Our Implementation

### 5.1 Changes We MUST Make (vs our prod_v1)

| # | Change | From (prod_v1) | To (V3) | Reason |
|---|--------|---------------|---------|--------|
| 1 | Training resolution | 256x256 | **512x512** | Official code image_size=512 |
| 2 | embed_dim (dk) | 576 | **2048** | Paper Table 5 best config |
| 3 | Dataset | Places2 (JPEG, 256px) | **DF8K (PNG, 2K+)** | Paper Sec 4 |
| 4 | Learning rate | 0.0001 | **0.001** | Paper Sec 5.1 |
| 5 | VGG normalization | Missing | **Add ImageNet mean/std** | Standard practice |

### 5.2 Things We Keep the Same (already match official code)

| Parameter | Value | Confirmed By |
|-----------|-------|-------------|
| Backbone | MobileOne-S4 | Official code |
| Patch size P | 8 | Official code + Paper Table 4 |
| nheads | 1 | Official code |
| dropout | 0.1 | Official code |
| Gaussian blur | (7,7), sigma=(2.01, 2.01) | Official code |
| Coherence layer | Conv2d(192,192,k=3,reflect) + Sigmoid | Official code |
| Output activation | Sigmoid | Official code |
| Mask compositing | coarse * mask + orig * (1-mask) | Official code |
| feature_dim | 896 | Official code |
| concat_features | True | Official code |
| use_argmax | False | Official code |
| Positional encoding | Disabled (use_qpos=None, use_kpos=None) | Official code |
| Attention residual | Disabled (commented out) | Official code |

### 5.3 Deliberate Deviations from Official Code (with justification)

| # | Official Code | Our V3 | Reason |
|---|---------------|--------|--------|
| 1 | feature_i=2, feature_dim=896 | **feature_i=3**, feature_dim=896 | Dimension mismatch: stage2=448ch ≠ 896. Using stage3=896ch makes dims match. |
| 2 | embed_dim=576 | **embed_dim=2048** | Paper Table 5 says dk=2048 gives best results (LPIPS=0.031) |
| 3 | Decoder channels buggy (3584, 1792, 768) | **Corrected** (1792, 896, 384) | Official decoder in_channels don't match MobileOne-S4 skip connections |
| 4 | kornia GaussianBlur2d | **Custom Gaussian** or torch-native | Kornia has .detach().cpu() that breaks CUDA graphs / torch.compile |
| 5 | einops.rearrange | **Native torch view/permute** | einops causes torch.compile graph breaks |
| 6 | No float32 cast on softmax | **Add float32 cast** | Prevents FP16 overflow in attention |
| 7 | k_mask unused | **Keep unused** (match official bug) | Matches official behavior exactly |

---

## 6. Evaluation Protocol

### 6.1 Metrics (Paper Section 5.1)
- **L1** distance
- **SSIM** [49]
- **FID** (Frechet Inception Distance) [21]
- **LPIPS** (Learned Perceptual Image Patch Similarity) [60]

### 6.2 Evaluation Resolutions
- 1024x1024 (CelebA-HQ, DIV8K)
- 2048x2048 (DIV8K)
- 4096x4096 (DIV8K)

### 6.3 Test Masks
- **Fixed set of test masks** (not random) for reproducibility
- Generated similar to LaMa [43], 30-50% coverage
- Same masks used across all methods for fair comparison

### 6.4 Baselines to Compare
- CoordFill [29]
- MI-GAN [41]

---

## 7. Implementation Plan

### Phase 1: Architecture & Setup
- [ ] Create rethined_prod_v3 codebase (copy from prod, modify)
- [ ] Change image_size to 512
- [ ] Change embed_dim to 2048
- [ ] Change decoder channels to match MobileOne-S4 (corrected)
- [ ] Keep feature_i=3, feature_dim=896 (dimensionally correct)
- [ ] Replace kornia GaussianBlur2d with torch-native
- [ ] Replace einops with native torch ops
- [ ] Add float32 cast on attention softmax
- [ ] Add ImageNet normalization to VGG perceptual loss
- [ ] Verify model param count (~18.3M target with dk=2048)
- [ ] Local sanity check: 50 steps, verify loss decreases
- [ ] Git commit

### Phase 2: Dataset
- [ ] Download DF2K from Kaggle (~28GB, 3,450 images)
- [ ] Source CAFHQ dataset
- [ ] Curate combined DF8K-Inpainting subset (~2,850 images)
- [ ] Create 70/20/10 train/test/val split
- [ ] Generate fixed test masks for evaluation
- [ ] Verify: images load correctly at 512x512, no JPEG artifacts

### Phase 3: Training (RunPod RTX 4090)
- [ ] Upload code + dataset to RunPod
- [ ] Train with LR=0.001, 600K steps, batch 128
- [ ] Save checkpoints every 50K steps
- [ ] Monitor: if catastrophic forgetting at LR=0.001, try 0.0005 then 0.0001
- [ ] Evaluate at each checkpoint

### Phase 4: Evaluation
- [ ] Evaluate on DF8K-Inpainting test set (with fixed masks)
- [ ] Evaluate on CelebA-HQ at 1024x1024
- [ ] Evaluate at 2048x2048 and 4096x4096 using Attention Upscaling
- [ ] Compare LPIPS/SSIM/FID/L1 against paper Table 1
- [ ] Speed benchmark with reparameterization

### Phase 5: Analysis & Ablation
- [ ] If results match paper: success
- [ ] If not: systematically ablate each change:
  - embed_dim 576 vs 2048
  - feature_i 2 vs 3
  - Training resolution 256 vs 512
  - Learning rate 0.001 vs 0.0001
  - Dataset DF8K vs Places2

---

## 8. Key References

| Ref # | Paper | Used For |
|-------|-------|----------|
| [47] | MobileOne (Vasu et al., CVPR 2023) | Encoder backbone |
| [43] | LaMa (Suvorov et al., WACV 2022) | Mask generation strategy |
| [39] | U-Net (Ronneberger et al., MICCAI 2015) | Encoder-decoder structure |
| [48] | Attention Is All You Need (Vaswani et al., NeurIPS 2017) | Self-attention mechanism |
| [6] | PatchMatch (Barnes et al., ACM TOG 2009) | Patch-based matching inspiration |
| [16] | ViT (Dosovitskiy et al., 2020) | Patch embedding approach |
| [14] | RepVGG (Ding et al., CVPR 2021) | Skip connection reparameterization |
| [42] | PixelShuffle (Shi et al., CVPR 2016) | Patch reassembly |
| [45] | DIV2K | Part of DF2K dataset |
| [27] | Flickr2K | Part of DF2K dataset |
| [58] | CAFHQ | Part of DF8K-Inpainting dataset |

---

## 9. Dependencies

From official `pyproject.toml`:
```
torch, torchvision, numpy, einops, kornia
```
Plus unlisted: `mobileone` (Apple ml-mobileone)

Our V3 will additionally need:
```
lpips, tensorboard, tqdm, pyyaml, scikit-image, opencv-python-headless, timm
```
