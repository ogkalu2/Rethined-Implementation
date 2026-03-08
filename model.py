"""RETHINED paper-path model."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mobileone import MobileOne, PARAMS, reparameterize_model


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
        softmax_temperature=1.0,
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
        self.softmax_temperature = softmax_temperature

    def forward(
        self,
        q,
        k,
        v,
        qpos,
        kpos,
        qk_mask=None,
        k_mask=None,
        post_softmax_mask=None,
        renorm_post_mask=False,
        direct_patch_mixing=False,
        return_head_outputs=False,
    ):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = v

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        if direct_patch_mixing:
            if n_head != 1:
                raise ValueError("direct_patch_mixing only supports n_head=1.")
            v_mixed = v.unsqueeze(1)
        else:
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
            v_mixed = v.transpose(1, 2)

        q, k = q.transpose(1, 2), k.transpose(1, 2)

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

        # Sharpen or soften the patch mixture without changing the top-k set.
        attn = attn / max(self.softmax_temperature, 1e-6)
        # Force FP32 for softmax to prevent FP16 overflow from large mask values
        attn = F.softmax(attn.float(), dim=-1).to(v.dtype)
        if post_softmax_mask is not None:
            attn = attn * post_softmax_mask
            if renorm_post_mask:
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.use_argmax:
            # Pick source patches, not attention heads. Use straight-through hard attention
            # so training still gets soft-attention gradients.
            idx = torch.argmax(attn, dim=-1, keepdim=True)
            attn_hard = torch.zeros_like(attn).scatter_(-1, idx, 1.0)
            attn = attn_hard - attn.detach() + attn

        attn = self.dropout(attn)
        head_output = torch.matmul(attn, v_mixed)
        output = head_output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        if direct_patch_mixing:
            output = output
        else:
            output = self.dropout(self.fc(output))
        if self.add_residual and not direct_patch_mixing:
            output = output + residual

        if return_head_outputs:
            head_output = head_output.transpose(1, 2).contiguous()
            return output, attn, head_output
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
        attn_topk: int = 8,
        attn_add_residual: bool = False,
        attn_temperature: float = 1.0,
        token_use_mask_ratio: bool = False,
        paper_mask_renormalize: bool = False,
        mask_patch_ratio_threshold: float = 0.05,
        min_valid_patches: int = 8,
        use_entropy_confidence: bool = False,
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
        self.attn_topk = attn_topk
        self.attn_add_residual = attn_add_residual
        self.attn_temperature = attn_temperature
        self.token_use_mask_ratio = token_use_mask_ratio
        self.paper_mask_renormalize = paper_mask_renormalize
        self.mask_patch_ratio_threshold = float(mask_patch_ratio_threshold)
        self.min_valid_patches = max(1, int(min_valid_patches))
        self.use_entropy_confidence = use_entropy_confidence
        super().__init__()
        # V3-A: Native Gaussian blur (replaces Kornia — no CUDA graph break)
        self.final_gaussian_blur = NativeGaussianBlur2d((7, 7), sigma=(2.01, 2.01))
        self.pooling_layer = nn.MaxPool2d(kernel_size, stride=kernel_size)
        self.patch_value_dim = stem_out_channels * kernel_size * kernel_size
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
            softmax_temperature=self.attn_temperature,
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
        self.refinement_gate = nn.Parameter(torch.tensor([1.0]))
        self.refinement_runtime_scale = 1.0
        self.paper_coherence_layer = nn.Conv2d(
            self.patch_value_dim, self.patch_value_dim,
            kernel_size=3, stride=1, padding=1, padding_mode='reflect',
        ) if self.final_conv else None
        # Zero-init so the residual coherence layer starts as identity
        if self.paper_coherence_layer is not None:
            nn.init.zeros_(self.paper_coherence_layer.weight)
            nn.init.zeros_(self.paper_coherence_layer.bias)
        self.last_base_patches_flat = None
        self.last_pixel_mask_flat = None
        self.last_output_patches_flat = None
        self.last_refinement_confidence = None
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
        """Native accelerator-friendly unfolding using F.unfold.

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

    def build_paper_attention_mask(self, pooled_patch_mask: torch.Tensor) -> torch.Tensor:
        """Pre-softmax additive mask for attention logits.

        Valid queries attend only to themselves (identity copy).
        Masked queries attend only to valid keys.
        Applied before softmax so it naturally normalizes over the correct set
        of keys — no post-softmax renormalization needed.
        """
        patch_mask_flat = pooled_patch_mask.squeeze(1).squeeze(-1)  # (B, N)
        B, N = patch_mask_flat.shape
        is_masked_q = patch_mask_flat.unsqueeze(-1)           # (B, N, 1)
        is_valid_k = (1.0 - patch_mask_flat).unsqueeze(1)    # (B, 1, N)
        eye = torch.eye(N, device=patch_mask_flat.device, dtype=patch_mask_flat.dtype).unsqueeze(0)

        # valid query → self only; masked query → valid keys
        allowed = (1.0 - is_masked_q) * eye + is_masked_q * is_valid_k  # (B, N, N)

        # 0 where allowed, -1e4 where blocked
        return ((1.0 - allowed) * -1e4).unsqueeze(1)  # (B, 1, N, N)

    def apply_paper_coherence(self, patches: torch.Tensor, output_size) -> torch.Tensor:
        """Apply the lightweight patch-grid coherence layer described in the paper."""
        if self.paper_coherence_layer is None:
            return patches

        n_h = output_size[0] // self.kernel_size
        n_w = output_size[1] // self.kernel_size
        B = patches.shape[0]
        patches_2d = patches.transpose(1, 2).view(B, -1, n_h, n_w)
        patches_2d = patches_2d + self.paper_coherence_layer(patches_2d)
        return patches_2d.view(B, -1, n_h * n_w).transpose(1, 2)

    def set_refinement_runtime_scale(self, scale: float) -> None:
        self.refinement_runtime_scale = float(scale)

    def forward(self, image, mask):
        masked_input = image
        image_coarse_inpainting, features = self.encoder_decoder(masked_input)
        if self.mask_inpainting:
            coarse_composite = image_coarse_inpainting * mask + masked_input * (1 - mask)
        else:
            coarse_composite = image_coarse_inpainting
        image_to_return = image_coarse_inpainting

        # Match the reference path: build attention tokens from the coarse
        # composite image decomposed into low/high frequencies.
        composite_blurred = self.final_gaussian_blur(coarse_composite)
        composite_patches_full, sizes = self.unfold_native(coarse_composite, self.kernel_size)
        blurred_patches_full, _ = self.unfold_native(composite_blurred, self.kernel_size)
        hf_patches = composite_patches_full - blurred_patches_full

        # V2: Use native unfold for mask
        mask_as_patches, _ = self.unfold_native(mask, self.kernel_size)
        mask_ratio = mask_as_patches.mean(dim=1, keepdim=True)
        mask_ratio_flat = mask_ratio.flatten(start_dim=2).squeeze(1)
        masked_patch_flat = (mask_ratio_flat > self.mask_patch_ratio_threshold).to(mask_ratio.dtype)
        valid_counts = (1.0 - masked_patch_flat).sum(dim=1).to(torch.int64)
        if torch.any(valid_counts < self.min_valid_patches):
            ranked = torch.argsort(mask_ratio_flat, dim=1)
            needed = (self.min_valid_patches - valid_counts).clamp_min(0)
            for batch_idx in torch.where(needed > 0)[0].tolist():
                promote_count = int(min(needed[batch_idx].item(), masked_patch_flat.shape[1]))
                if promote_count <= 0:
                    continue
                promote_idx = ranked[batch_idx, :promote_count]
                masked_patch_flat[batch_idx, promote_idx] = 0.0
        mask_same_res_as_features_pooled = masked_patch_flat.unsqueeze(1).unsqueeze(-1)
        pixel_mask_flat = mask_as_patches.flatten(start_dim=2).transpose(1, 2)
        pixel_mask_flat = pixel_mask_flat.repeat(1, 1, self.stem_out_channels)

        match_patches = hf_patches
        preserve_patches_full = composite_patches_full
        features_to_concat = None
        if self.concat_features:
            features_to_concat = features[self.feature_i]
            features_to_concat = F.interpolate(features_to_concat, size=hf_patches.shape[-2:], mode='bilinear', align_corners=False)
            token_map = torch.cat([match_patches, features_to_concat], dim=1)
        else:
            token_map = match_patches

        if self.token_use_mask_ratio:
            token_map = torch.cat([token_map, mask_ratio], dim=1)

        input_attn = token_map.flatten(start_dim=2).transpose(1, 2)

        # Add learnable positional encoding to token embeddings so Q/K
        # projections gain spatial awareness (standard transformer PE).
        if self.positionalencoding is not None:
            input_attn = input_attn + self.positionalencoding.transpose(1, 2)

        hf_patches_flat = hf_patches.flatten(start_dim=2).transpose(1, 2)
        preserve_patches_flat = preserve_patches_full.flatten(start_dim=2).transpose(1, 2)
        blurred_patches_flat = blurred_patches_full.flatten(start_dim=2).transpose(1, 2)
        base_hf_flat = preserve_patches_flat - blurred_patches_flat
        self.last_base_patches_flat = preserve_patches_flat

        pre_softmax_mask = self.build_paper_attention_mask(mask_same_res_as_features_pooled) if self.attention_masking else None
        out, atten_weights = self.multihead_attention(
            input_attn,
            input_attn,
            hf_patches_flat,
            qpos=None,
            kpos=None,
            qk_mask=pre_softmax_mask,
            k_mask=None,
            post_softmax_mask=None,
            renorm_post_mask=False,
            direct_patch_mixing=True,
        )

        patch_mask = mask_same_res_as_features_pooled.squeeze(1).squeeze(-1).unsqueeze(-1)

        if self.use_entropy_confidence:
            attn_probs = atten_weights.squeeze(1)
            attn_entropy = -(attn_probs.clamp_min(1e-8) * attn_probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
            valid_key_count = (1.0 - patch_mask.squeeze(-1)).sum(dim=1, keepdim=True)
            max_entropy = valid_key_count.clamp_min(2.0).log().unsqueeze(-1)
            refinement_confidence = (1.0 - attn_entropy / max_entropy.clamp_min(1e-6)).clamp(0.0, 1.0)
            refinement_confidence = refinement_confidence * patch_mask
        else:
            refinement_confidence = patch_mask

        refinement_scale = torch.tanh(self.refinement_gate) * self.refinement_runtime_scale
        
        # Smoothly interpolate between the hallucinated coarse HF and the retrieved valid HF
        # Gated by both the learned global scale and the per-patch entropy confidence
        delta = refinement_scale * refinement_confidence * (out - base_hf_flat) * patch_mask
        delta = self.apply_paper_coherence(delta, sizes)
        out = preserve_patches_flat + delta
        out = out * patch_mask + preserve_patches_flat * (1 - patch_mask)
        self.last_output_patches_flat = out
        self.last_pixel_mask_flat = pixel_mask_flat
        self.last_refinement_confidence = refinement_confidence

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


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.proj = None
        if use_residual and (stride != 1 or in_channels != out_channels):
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.depthwise(x)
        out = self.depthwise_bn(out)
        out = self.activation(out)
        out = self.pointwise(out)
        out = self.pointwise_bn(out)
        if self.use_residual:
            if self.proj is not None:
                residual = self.proj(residual)
            out = out + residual
        return self.activation(out)


class PaperEncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = DepthwiseSeparableBlock(in_channels, out_channels, stride=2, use_residual=True)
        self.refine = DepthwiseSeparableBlock(out_channels, out_channels, stride=1, use_residual=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.refine(x)


class PaperUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block1 = DepthwiseSeparableBlock(
            in_channels + skip_channels,
            out_channels,
            stride=1,
            use_residual=True,
        )
        self.block2 = DepthwiseSeparableBlock(out_channels, out_channels, stride=1, use_residual=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x)
        return self.block2(x)


class PaperCoarse(nn.Module):
    def __init__(self, channels=None, head_channels: int = 32, **kwargs):
        super().__init__()
        if kwargs:
            raise ValueError(f"Unsupported PaperCoarse arguments: {sorted(kwargs)}")

        if channels is None:
            channels = [64, 128, 256, 384, 512]
        if len(channels) != 5:
            raise ValueError(f"PaperCoarse expects 5 channel values, got {channels}")

        c0, c1, c2, c3, c4 = [int(c) for c in channels]
        self.feature_channels = [c0, c1, c2, c3, c4]

        self.stage0 = PaperEncoderStage(3, c0)
        self.stage1 = PaperEncoderStage(c0, c1)
        self.stage2 = PaperEncoderStage(c1, c2)
        self.stage3 = PaperEncoderStage(c2, c3)
        self.stage4 = PaperEncoderStage(c3, c4)

        self.up4 = PaperUpBlock(c4, c3, c3)
        self.up3 = PaperUpBlock(c3, c2, c2)
        self.up2 = PaperUpBlock(c2, c1, c1)
        self.up1 = PaperUpBlock(c1, c0, c0)
        self.head_channels = int(head_channels)
        self.head_block = DepthwiseSeparableBlock(c0 + 3, self.head_channels, stride=1, use_residual=True)
        self.out_conv = nn.Conv2d(self.head_channels, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        features = []
        x0 = self.stage0(x)
        features.append(x0)
        x1 = self.stage1(x0)
        features.append(x1)
        x2 = self.stage2(x1)
        features.append(x2)
        x3 = self.stage3(x2)
        features.append(x3)
        x4 = self.stage4(x3)
        features.append(x4)

        out = self.up4(x4, x3)
        out = self.up3(out, x2)
        out = self.up2(out, x1)
        out = self.up1(out, x0)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.cat([out, x], dim=1)
        out = self.head_block(out)
        out = self.sigmoid(self.out_conv(out))
        return out, features

    def reparameterize(self):
        return self


class MobileOneCoarse(nn.Module):
    def __init__(self, variant='s4', **kwargs):
        super().__init__()
        if variant not in PARAMS:
            raise ValueError(f"Unsupported MobileOne variant: {variant}")

        variant_params = dict(PARAMS[variant])
        variant_params.update(kwargs)
        width_multipliers = variant_params["width_multipliers"]
        stage0_channels = min(64, int(64 * width_multipliers[0]))
        stage1_channels = int(64 * width_multipliers[0])
        stage2_channels = int(128 * width_multipliers[1])
        stage3_channels = int(256 * width_multipliers[2])
        stage4_channels = int(512 * width_multipliers[3])
        self.feature_channels = [
            stage0_channels,
            stage1_channels,
            stage2_channels,
            stage3_channels,
            stage4_channels,
        ]

        self.model = MobileOne(**variant_params)
        self.d4 = nn.ConvTranspose2d(stage4_channels, stage3_channels, kernel_size=4, stride=2, padding=1)
        self.d3 = nn.ConvTranspose2d(stage3_channels + stage3_channels, stage2_channels, kernel_size=4, stride=2, padding=1)
        self.d2 = nn.ConvTranspose2d(stage2_channels + stage2_channels, stage1_channels, kernel_size=4, stride=2, padding=1)
        self.d1 = nn.ConvTranspose2d(stage1_channels + stage1_channels, stage0_channels, kernel_size=4, stride=2, padding=1)
        self.d0 = nn.ConvTranspose2d(stage0_channels + stage0_channels, 3, kernel_size=4, stride=2, padding=1)
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


COARSE_MODEL_REGISTRY = {
    "MobileOneCoarse": MobileOneCoarse,
    "PaperCoarse": PaperCoarse,
}


class AttentionUpscaling(nn.Module):
    def __init__(self, patch_inpainting_module: "PatchInpainting"):
        super().__init__()
        self.patch_inpainting = patch_inpainting_module

    def forward(self, x_hr, x_lr_inpainted, attn_map):
        hr_h, hr_w = x_hr.shape[-2:]
        lr_h, lr_w = x_lr_inpainted.shape[-2:]

        scale_h = hr_h // lr_h
        scale_w = hr_w // lr_w
        if hr_h % lr_h != 0 or hr_w % lr_w != 0 or scale_h != scale_w:
            raise ValueError(
                "AttentionUpscaling requires an integer isotropic HR/LR scale factor "
                f"(got LR {lr_h}x{lr_w}, HR {hr_h}x{hr_w})"
            )
        if attn_map.dim() != 4 or attn_map.size(1) != 1:
            raise ValueError(
                "AttentionUpscaling expects a single-head attention map with shape "
                f"(B, 1, N, N); got {tuple(attn_map.shape)}"
            )

        # Section 3.4: bicubic upsample the LR inpainted image, then add HR HF details.
        x_hr_base = F.interpolate(x_lr_inpainted, size=(hr_h, hr_w), mode='bicubic', align_corners=False)

        hr_patch_size = self.patch_inpainting.kernel_size * scale_h

        hr_patches, _ = self.patch_inpainting.unfold_native(x_hr, hr_patch_size)

        # Section 3.4 mixes only HR high frequencies extracted from the HR masked image.
        hr_blurred = self.patch_inpainting.final_gaussian_blur(x_hr)
        hr_patches_blurred, _ = self.patch_inpainting.unfold_native(hr_blurred, hr_patch_size)

        hr_hf_patches = hr_patches - hr_patches_blurred
        hr_hf_patches = hr_hf_patches.flatten(start_dim=2).transpose(1, 2)

        # Section 3.4 transfers HR high frequencies using the LR attention map
        # without applying an extra learned LR refinement gate on the HR branch.
        reconstructed_hr_hf_patches = torch.matmul(attn_map.squeeze(1), hr_hf_patches)

        # Reassemble non-overlapping HR patches without an HR coherence layer.
        reconstructed_hr_hf_image = self.patch_inpainting.fold_native(
            reconstructed_hr_hf_patches, (hr_h, hr_w), kernel_size=hr_patch_size, use_final_conv=False)

        final_hr_image = x_hr_base + reconstructed_hr_hf_image

        return final_hr_image


class InpaintingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        coarse_class_name = config['coarse_model'].get('class', 'MobileOneCoarse')
        coarse_class = COARSE_MODEL_REGISTRY.get(coarse_class_name)
        if coarse_class is None:
            raise ValueError(f"Unsupported coarse model class: {coarse_class_name}")

        self.coarse_model = coarse_class(**config['coarse_model']['parameters'])
        generator_params = dict(config['generator']['params'])

        if generator_params.get('concat_features', True):
            feature_i = generator_params.get('feature_i', 3)
            feature_channels = self.coarse_model.feature_channels
            if feature_i < 0 or feature_i >= len(feature_channels):
                raise ValueError(
                    f"model.generator.feature_i={feature_i} is out of range for coarse feature maps"
                )

            inferred_feature_dim = feature_channels[feature_i]
            configured_feature_dim = generator_params.get('feature_dim')
            if configured_feature_dim is None:
                generator_params['feature_dim'] = inferred_feature_dim
            elif configured_feature_dim != inferred_feature_dim:
                raise ValueError(
                    "model.generator.feature_dim does not match the selected coarse feature map: "
                    f"got {configured_feature_dim}, expected {inferred_feature_dim} for feature_i={feature_i}"
                )

        self.generator = PatchInpainting(**generator_params, model=self.coarse_model)

    def forward(self, image, mask):
        return self.generator(image, mask)

    def reparameterize(self):
        """Fuse MobileOne multi-branch+BN into single conv for inference.

        This is a one-way operation — the model can no longer be trained after this.
        Call after loading checkpoint, before inference.
        """
        if hasattr(self.coarse_model, 'reparameterize'):
            self.coarse_model.reparameterize()
        elif hasattr(self.coarse_model, 'model'):
            self.coarse_model.model = reparameterize_model(self.coarse_model.model)
        return self
