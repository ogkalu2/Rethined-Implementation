"""RETHINED V6: refinement fixes on top of the V3 codebase."""

import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np

from mobileone import MobileOne, mobileone, reparameterize_model


class NativeGaussianBlur2d(nn.Module):
    """Drop-in replacement for kornia.filters.GaussianBlur2d.

    Uses precomputed separable Gaussian kernel stored as buffer.
    No .detach().cpu() hook — fully compatible with torch.compile fullgraph.
    Separable: 2 * k multiplies per pixel instead of k*k (3.5x fewer for k=7).
    """

    def __init__(self, kernel_size: tuple = (7, 7), sigma: tuple = (2.01, 2.01), **kwargs):
        super().__init__()
        ks = kernel_size[0]
        sig = sigma[0]
        x = torch.arange(ks, dtype=torch.float32) - ks // 2
        gauss_1d = torch.exp(-0.5 * (x / sig) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        self.register_buffer('kernel_h', gauss_1d.view(1, 1, 1, -1))
        self.register_buffer('kernel_v', gauss_1d.view(1, 1, -1, 1))
        self.padding = ks // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        kh = self.kernel_h.to(dtype=x.dtype).expand(C, -1, -1, -1)
        kv = self.kernel_v.to(dtype=x.dtype).expand(C, -1, -1, -1)
        x = F.pad(x, [self.padding] * 4, mode='reflect')
        x = F.conv2d(x, kh, groups=C)
        x = F.conv2d(x, kv, groups=C)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        d_v,
        n_head,
        split,
        dropout,
        d_qk,
        compute_v,
        use_argmax=False,
        add_residual=True,
        topk_patches=None,
    ):
        super().__init__()
        self.d_v = d_v
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.w_qs = nn.Linear(embed_dim, n_head * d_qk, bias=False)
        self.w_ks = nn.Linear(embed_dim, n_head * d_qk, bias=False)
        self.w_vs = nn.Linear(d_v, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_v, bias=False)
        self.attention = None
        self.d_k = d_qk
        self.use_argmax = use_argmax
        self.add_residual = add_residual
        self.topk_patches = topk_patches

    def forward(self, q, k, v, qpos, kpos, qk_mask=None, k_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = v

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = torch.matmul(q / self.d_k**0.5, k.transpose(2, 3))

        if qk_mask is not None:
            attn = attn + qk_mask

        if k_mask is not None:
            # Broadcast from (B, 1, N, 1) to attention logits (B, n_head, N, N).
            attn = attn + k_mask.squeeze(-1).unsqueeze(-2)

        if self.topk_patches is not None and self.topk_patches < attn.size(-1):
            topk_vals, topk_idx = torch.topk(attn, k=self.topk_patches, dim=-1)
            sparse_attn = torch.full_like(attn, -1e4)
            attn = sparse_attn.scatter(-1, topk_idx, topk_vals)

        # Force FP32 for softmax to prevent FP16 overflow from large mask values
        attn = F.softmax(attn.float(), dim=-1).to(v.dtype)

        if self.use_argmax:
            idx = torch.argmax(attn, dim=1, keepdims=True)
            attn = torch.zeros_like(attn).scatter_(1, idx, 1.)

        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        if self.add_residual:
            output = output + residual

        return output, attn


class PatchInpainting(nn.Module):
    def __init__(
        self,
        *,
        kernel_size: int,
        nheads: int,
        stem_out_stride: int = 1,
        stem_out_channels: int = 3,
        cross_attention: bool = False,
        mask_query_with_segmentation_mask: bool = False,
        merge_mode: str = 'sum',
        use_kpos: bool = True,
        image_size: int = 512,
        embed_dim: int = 512,
        use_qpos: bool = True,
        dropout: float = 0.1,
        attention_type: str = 'MultiHeadAttention',
        compute_v: float = 0.1,
        feature_i: int = 3,
        feature_dim: int = 128,
        concat_features: bool = True,
        attention_masking: bool = True,
        final_conv: bool = False,
        mask_inpainting: bool = True,
        use_argmax: bool = False,
        residual_refinement: bool = True,
        refinement_gate_init: float = 1.0,
        refinement_backend: str = "patch_attention",
        refinement_hidden_dim: int = 128,
        patch_match_mode: str = "hf",
        attn_topk: int = 8,
        attn_add_residual: bool = False,
        token_use_mask_ratio: bool = True,
        soft_key_mask_scale: float = 0.0,
        model,
    ):
        self.cross_attention = cross_attention
        self.kernel_size = kernel_size
        self.mask_query_with_segmentation_mask = mask_query_with_segmentation_mask
        self.nheads = nheads
        self.use_kpos = use_kpos
        self.use_qpos = use_qpos
        self.feature_i = feature_i
        self.feature_dim = feature_dim
        self.concat_features = concat_features
        self.attention_masking = attention_masking
        self.window_size = image_size // kernel_size
        self.final_conv = final_conv
        self.mask_inpainting = mask_inpainting
        self.use_argmax = use_argmax
        self.use_residual_refinement = residual_refinement
        self.refinement_backend = refinement_backend
        self.refinement_hidden_dim = refinement_hidden_dim
        self.patch_match_mode = patch_match_mode
        self.attn_topk = attn_topk
        self.attn_add_residual = attn_add_residual
        self.token_use_mask_ratio = token_use_mask_ratio
        self.soft_key_mask_scale = soft_key_mask_scale
        super().__init__()
        if self.refinement_backend not in ("patch_attention", "patch", None):
            raise ValueError(
                f"Unsupported refinement_backend={self.refinement_backend!r}. "
                "Only the patch-based backend is supported."
            )
        if self.patch_match_mode not in ("hf", "full", "hybrid"):
            raise ValueError(
                f"Unsupported patch_match_mode={self.patch_match_mode!r}. "
                "Expected one of: 'hf', 'full', 'hybrid'."
            )
        # V3-A: Native Gaussian blur (replaces Kornia — no CUDA graph break)
        self.final_gaussian_blur = NativeGaussianBlur2d((7, 7), sigma=(2.01, 2.01))
        self.pooling_layer = nn.MaxPool2d(kernel_size, stride=kernel_size)
        self.patch_value_dim = stem_out_channels * kernel_size * kernel_size
        if self.patch_match_mode == "hybrid":
            self.patch_match_dim = self.patch_value_dim * 2
        else:
            self.patch_match_dim = self.patch_value_dim
        self.patch_token_dim = self.patch_match_dim + self.feature_dim if self.concat_features else self.patch_match_dim
        if self.token_use_mask_ratio:
            self.patch_token_dim += 1
        self.multihead_attention = MultiHeadAttention(
            embed_dim=self.patch_token_dim,
            d_v=self.patch_value_dim,
            n_head=self.nheads,
            split=True,
            dropout=dropout,
            d_qk=embed_dim,
            compute_v=compute_v,
            use_argmax=self.use_argmax,
            add_residual=self.attn_add_residual,
            topk_patches=self.attn_topk,
        )
        self.stem_out_channels = stem_out_channels
        self.stem_out_stride = stem_out_stride
        self.register_buffer('qk_mask', 1e4 * torch.eye(
            int((image_size / stem_out_stride / self.kernel_size) ** 2)
        ).unsqueeze(0).unsqueeze(0))
        if not mask_query_with_segmentation_mask:
            self.mask_query = torch.nn.Parameter(torch.zeros(
                1, int((image_size / stem_out_stride / self.kernel_size) ** 2), 1, 1).float())

        self.encoder_decoder = model
        self.image_size = image_size
        self.positionalencoding = torch.nn.Parameter(torch.zeros(
            1, self.patch_token_dim,
            int((image_size / stem_out_stride / self.kernel_size) ** 2)
        )) if use_kpos or use_qpos else None
        self.patch_token_norm = nn.LayerNorm(self.patch_token_dim)
        self.patch_delta_mixer = torch.nn.Sequential(
            nn.Conv2d(self.patch_value_dim, self.patch_value_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(self.patch_value_dim, self.patch_value_dim, kernel_size=1, stride=1),
        ) if self.final_conv else None
        self.patch_gate_mixer = torch.nn.Sequential(
            nn.Conv2d(self.patch_value_dim * 2, self.patch_value_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(self.patch_value_dim, self.patch_value_dim, kernel_size=1, stride=1),
        ) if self.final_conv else None
        self.refinement_gate = nn.Parameter(torch.tensor(refinement_gate_init, dtype=torch.float32))
        self.pixel_shuffle = nn.PixelShuffle(self.kernel_size)
        if merge_mode == 'all':
            self.merge_func = self.merge_all_patches_sum

        # V2: Keep identity kernel buffers for V1 checkpoint compatibility during loading,
        # but they are NOT used in forward pass anymore.
        self.register_buffer(
            name="unfolding_weights",
            tensor=self._compute_unfolding_weights(self.kernel_size, self.stem_out_channels),
            persistent=False,
        )
        self.register_buffer(
            name="unfolding_weights_image",
            tensor=self._compute_unfolding_weights(self.kernel_size, 3),
            persistent=False,
        )
        self.register_buffer(
            name="unfolding_weights_mask",
            tensor=self._compute_unfolding_weights(self.kernel_size, 1),
            persistent=False,
        )

    def _compute_unfolding_weights(self, kernel_size, channels) -> torch.Tensor:
        """Kept for V1 checkpoint compatibility. Not used in V2 forward pass."""
        weights = torch.eye(kernel_size * kernel_size, dtype=torch.float)
        weights = weights.reshape((kernel_size * kernel_size, 1, kernel_size, kernel_size))
        weights = weights.repeat(channels, 1, 1, 1)
        return weights

    def unfold_native(self, feature_map: torch.Tensor, kernel_size: int):
        """CUDA-optimized unfolding using native F.unfold.

        Replaces unfolding_coreml() which used F.conv2d with identity kernels.
        Produces identical output shape: (B, C*k*k, n_h, n_w).
        """
        B, C, H, W = feature_map.shape
        # F.unfold returns (B, C*k*k, L) where L = n_h * n_w
        patches = F.unfold(feature_map, kernel_size=kernel_size, stride=kernel_size)
        n_h, n_w = H // kernel_size, W // kernel_size
        patches = patches.view(B, C * kernel_size * kernel_size, n_h, n_w)
        return patches, (H, W)

    def fold_native(self, patches: torch.Tensor, output_size, kernel_size: int, use_final_conv: bool) -> torch.Tensor:
        """Folding patches back to image.

        V3-B: Replaced einops.rearrange with native view/permute to enable
        torch.compile fullgraph mode (einops causes graph breaks).
        """
        n_h = output_size[0] // kernel_size
        n_w = output_size[1] // kernel_size
        B = patches.shape[0]

        # patches: (B, n_h*n_w, C*k*k)
        C = patches.shape[2] // (kernel_size * kernel_size)
        # -> (B, n_h, n_w, C, k, k) -> (B, C, n_h, k, n_w, k) -> (B, C, H, W)
        patches = patches.view(B, n_h, n_w, C, kernel_size, kernel_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        final_image = patches.view(B, C, n_h * kernel_size, n_w * kernel_size)
        return final_image

    def mix_patch_tokens(self, patches: torch.Tensor, output_size) -> torch.Tensor:
        """Apply a lightweight learned mixer in patch-token space."""
        if self.patch_delta_mixer is None:
            return patches

        n_h = output_size[0] // self.kernel_size
        n_w = output_size[1] // self.kernel_size
        B = patches.shape[0]

        patches_2d = patches.transpose(1, 2).view(B, -1, n_h, n_w)
        patches_2d = self.patch_delta_mixer(patches_2d)
        return patches_2d.view(B, -1, n_h * n_w).transpose(1, 2)

    def compute_patch_gate(self, base_patches: torch.Tensor, delta_patches: torch.Tensor, output_size) -> torch.Tensor:
        """Predict a per-token gate for residual updates."""
        if self.patch_gate_mixer is None:
            return delta_patches.new_ones(delta_patches.shape)

        n_h = output_size[0] // self.kernel_size
        n_w = output_size[1] // self.kernel_size
        B = delta_patches.shape[0]

        base_2d = base_patches.transpose(1, 2).view(B, -1, n_h, n_w)
        delta_2d = delta_patches.transpose(1, 2).view(B, -1, n_h, n_w)
        gate_logits = self.patch_gate_mixer(torch.cat([base_2d, delta_2d], dim=1))
        return gate_logits.view(B, -1, n_h * n_w).transpose(1, 2)

    def forward(self, image, mask):
        masked_input = image
        image_coarse_inpainting, features = self.encoder_decoder(masked_input)
        if self.mask_inpainting:
            coarse_composite = image_coarse_inpainting * mask + masked_input * (1 - mask)
        else:
            coarse_composite = image_coarse_inpainting
        image_to_return = image_coarse_inpainting

        # V2: Use native unfold instead of CoreML conv2d hack
        image_as_patches_full, sizes = self.unfold_native(coarse_composite, self.kernel_size)
        image_blurred = self.final_gaussian_blur(coarse_composite)
        image_as_patches_blurred, _ = self.unfold_native(image_blurred, self.kernel_size)
        image_as_patches_hf = image_as_patches_full - image_as_patches_blurred

        pos = self.positionalencoding.repeat(
            image_as_patches_full.size(0), 1, 1).unsqueeze(2) if self.use_qpos else None

        # V2: Use native unfold for mask
        mask_as_patches, _ = self.unfold_native(mask, self.kernel_size)
        # A patch is corrupted if any pixel inside it is corrupted.
        mask_same_res_as_features_pooled = mask_as_patches.amax(dim=1, keepdim=True)
        mask_same_res_as_features_pooled = mask_same_res_as_features_pooled.flatten(
            start_dim=2).unsqueeze(-1)
        mask_ratio = mask_as_patches.mean(dim=1, keepdim=True)
        pixel_mask_flat = mask_as_patches.flatten(start_dim=2).transpose(1, 2)
        pixel_mask_flat = pixel_mask_flat.repeat(1, 1, self.stem_out_channels)

        if self.patch_match_mode == "hf":
            match_patches = image_as_patches_hf
        elif self.patch_match_mode == "full":
            match_patches = image_as_patches_full
        else:
            match_patches = torch.cat([image_as_patches_hf, image_as_patches_full], dim=1)
        if self.concat_features:
            features_to_concat = features[self.feature_i]
            features_to_concat = F.interpolate(features_to_concat, size=image_as_patches_full.shape[-2:], mode='bilinear', align_corners=False)
            token_map = torch.cat([match_patches, features_to_concat], dim=1)
        else:
            token_map = match_patches

        if self.token_use_mask_ratio:
            token_map = torch.cat([token_map, mask_ratio], dim=1)

        input_attn = token_map.flatten(start_dim=2).transpose(1, 2)
        input_attn = self.patch_token_norm(input_attn)

        full_patches_flat = image_as_patches_full.flatten(start_dim=2).transpose(1, 2)

        qk_mask = -1e4 * self.qk_mask.repeat(full_patches_flat.size(0), 1, 1, 1) if self.attention_masking else None
        if self.attention_masking:
            if self.soft_key_mask_scale > 0:
                key_mask_ratio = mask_ratio.flatten(start_dim=2).unsqueeze(-1)
                k_mask = -self.soft_key_mask_scale * key_mask_ratio
            else:
                k_mask = -1e4 * mask_same_res_as_features_pooled
        else:
            k_mask = None
        out, atten_weights = self.multihead_attention(input_attn, input_attn,
            full_patches_flat, qpos=pos, kpos=pos, qk_mask=qk_mask, k_mask=k_mask
        )

        patch_mask = mask_same_res_as_features_pooled.squeeze(1).squeeze(-1).unsqueeze(-1)
        if self.use_residual_refinement:
            delta_full = out - full_patches_flat
            delta_full = self.mix_patch_tokens(delta_full, sizes)
            gate_logits = self.compute_patch_gate(full_patches_flat, delta_full, sizes)
            refinement_gate = torch.sigmoid(self.refinement_gate) * torch.sigmoid(gate_logits)
            out = full_patches_flat + refinement_gate * pixel_mask_flat * delta_full
        else:
            out = out * patch_mask + full_patches_flat * (1 - patch_mask)

        # V2: Use native fold
        out = self.fold_native(out, sizes, self.kernel_size, use_final_conv=False)

        return out, atten_weights, image_to_return

    def merge_all_patches_sum(self, patch_scores, sequence_of_patches):
        return torch.einsum('bkhq,bchk->bchq', patch_scores, sequence_of_patches.unsqueeze(2)).squeeze(2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, d_model, max_len)
        pe[0, 0::2, :] = torch.sin(position * div_term).T
        pe[0, 1::2, :] = torch.cos(position * div_term).T
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe[:, :, :x.size(-1)]
        return self.dropout(x)


class MobileOneCoarse(nn.Module):
    def __init__(self, variant='s4', **kwargs):
        super().__init__()
        self.model = mobileone(variant=variant, **kwargs)

        # Decoder - channels match standard MobileOne S4:
        # stage0=64, stage1=192, stage2=448, stage3=896, stage4=2048
        self.d4 = nn.ConvTranspose2d(2048, 896, kernel_size=4, stride=2, padding=1)
        self.d3 = nn.ConvTranspose2d(896 + 896, 448, kernel_size=4, stride=2, padding=1)
        self.d2 = nn.ConvTranspose2d(448 + 448, 192, kernel_size=4, stride=2, padding=1)
        self.d1 = nn.ConvTranspose2d(192 + 192, 64, kernel_size=4, stride=2, padding=1)
        self.d0 = nn.ConvTranspose2d(64 + 64, 3, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = []
        x0 = self.model.stage0(x)
        features.append(x0)
        x1 = self.model.stage1(x0)
        features.append(x1)
        x2 = self.model.stage2(x1)
        features.append(x2)
        x3 = self.model.stage3(x2)
        features.append(x3)
        x4 = self.model.stage4(x3)
        features.append(x4)

        out = self.relu(self.d4(x4))
        out = torch.cat([out, x3], dim=1)
        out = self.relu(self.d3(out))
        out = torch.cat([out, x2], dim=1)
        out = self.relu(self.d2(out))
        out = torch.cat([out, x1], dim=1)
        out = self.relu(self.d1(out))
        out = torch.cat([out, x0], dim=1)
        out = self.sigmoid(self.d0(out))

        return out, features


class AttentionUpscaling(nn.Module):
    def __init__(self, patch_inpainting_module: "PatchInpainting"):
        super().__init__()
        self.patch_inpainting = patch_inpainting_module

    def forward(self, x_hr, x_lr_inpainted, attn_map, x_lr_blurred=None):
        hr_h, hr_w = x_hr.shape[-2:]
        lr_h, lr_w = x_lr_inpainted.shape[-2:]

        # V3-C: Bilinear instead of bicubic (2-4x faster, negligible quality loss)
        x_hr_base = F.interpolate(x_lr_inpainted, size=(hr_h, hr_w), mode='bilinear', align_corners=False)

        hr_patch_size = self.patch_inpainting.kernel_size * (hr_h // lr_h)

        hr_patches, _ = self.patch_inpainting.unfold_native(x_hr, hr_patch_size)

        # V3-D: Blur at LR domain then upsample (instead of blurring at HR)
        if x_lr_blurred is not None:
            hr_blurred = F.interpolate(x_lr_blurred, size=(hr_h, hr_w), mode='bilinear', align_corners=False)
        else:
            # Fallback: blur at HR (for base resolution where LR blur isn't provided)
            hr_blurred = self.patch_inpainting.final_gaussian_blur(x_hr)
        hr_patches_blurred, _ = self.patch_inpainting.unfold_native(hr_blurred, hr_patch_size)

        # High-frequency = original - blurred
        hr_hf_patches = hr_patches - hr_patches_blurred
        hr_hf_patches = hr_hf_patches.flatten(start_dim=2).transpose(1, 2)

        # Attention-weighted sum of HR high-frequency patches
        reconstructed_hr_hf_patches = torch.matmul(attn_map.squeeze(1), hr_hf_patches)

        # Fold back to image
        reconstructed_hr_hf_image = self.patch_inpainting.fold_native(
            reconstructed_hr_hf_patches, (hr_h, hr_w), kernel_size=hr_patch_size, use_final_conv=False)

        final_hr_image = x_hr_base + reconstructed_hr_hf_image

        return final_hr_image


class InpaintingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.coarse_model = MobileOneCoarse(**config['coarse_model']['parameters'])
        self.generator = PatchInpainting(**config['generator']['params'], model=self.coarse_model)

    def forward(self, image, mask):
        return self.generator(image, mask)

    def reparameterize(self):
        """Fuse MobileOne multi-branch+BN into single conv for inference.

        This is a one-way operation — the model can no longer be trained after this.
        Call after loading checkpoint, before inference.
        """
        self.coarse_model.model = reparameterize_model(self.coarse_model.model)
        return self
