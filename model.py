"""RETHINED paper-path model."""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


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


def _fuse_conv_bn_pair(conv: nn.Module, bn: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Fuse a Conv2d+BatchNorm2d pair for inference."""
    if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
        return conv, bn
    return fuse_conv_bn_eval(conv.eval(), bn.eval()), nn.Identity()


def _fuse_conv_bn_to_weight_bias(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fused kernel and bias tensors for a Conv2d+BN branch."""
    fused = fuse_conv_bn_eval(conv.eval(), bn.eval())
    return fused.weight.detach().clone(), fused.bias.detach().clone()


def _pad_kernel_to_size(kernel: torch.Tensor, target_size: int) -> torch.Tensor:
    """Center-pad a smaller spatial kernel to a target square size."""
    current_size = kernel.shape[-1]
    if current_size == target_size:
        return kernel
    if current_size > target_size or (target_size - current_size) % 2 != 0:
        raise ValueError(f"Cannot pad kernel {current_size} to target {target_size}")
    pad = (target_size - current_size) // 2
    return F.pad(kernel, [pad, pad, pad, pad])


def _fuse_identity_bn_to_weight_bias(
    num_channels: int,
    bn: nn.BatchNorm2d,
    kernel_size: int,
    groups: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fused kernel and bias tensors for an identity+BN branch."""
    if num_channels % groups != 0:
        raise ValueError(f"Identity fusion expects num_channels divisible by groups (got {num_channels}, {groups})")

    input_dim = num_channels // groups
    kernel = torch.zeros(
        (num_channels, input_dim, kernel_size, kernel_size),
        dtype=bn.weight.dtype,
        device=bn.weight.device,
    )
    center = kernel_size // 2
    channel_idx = torch.arange(num_channels, device=kernel.device)
    kernel[channel_idx, channel_idx % input_dim, center, center] = 1.0

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    std = torch.sqrt(var + bn.eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    bias = beta - mean * gamma / std
    return kernel * scale, bias


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
        hr_candidate_topk: int | None = None,
        hr_feature_pool_size: int = 4,
        hr_rescore_dim: int = 64,
        hr_rescore_hidden_dim: int = 128,
        hr_query_chunk_size: int = 128,
        use_hr_residual_refiner: bool = True,
        hr_refiner_channels: int = 32,
        hr_refiner_blocks: int = 4,
        hr_refiner_template_size: int = 4,
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
        self.hr_candidate_topk = int(hr_candidate_topk) if hr_candidate_topk is not None else (
            int(attn_topk) if attn_topk is not None and int(attn_topk) > 0 else 16
        )
        self.hr_feature_pool_size = max(1, int(hr_feature_pool_size))
        self.hr_rescore_dim = max(8, int(hr_rescore_dim))
        self.hr_rescore_hidden_dim = max(8, int(hr_rescore_hidden_dim))
        self.hr_query_chunk_size = max(1, int(hr_query_chunk_size))
        self.use_hr_residual_refiner = bool(use_hr_residual_refiner)
        self.hr_refiner_channels = max(8, int(hr_refiner_channels))
        self.hr_refiner_blocks = max(1, int(hr_refiner_blocks))
        self.hr_refiner_template_size = max(1, int(hr_refiner_template_size))
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
        if self.paper_coherence_layer is not None:
            nn.init.zeros_(self.paper_coherence_layer.weight)
            nn.init.zeros_(self.paper_coherence_layer.bias)
        self.hr_patch_descriptor_dim = self.stem_out_channels * self.hr_feature_pool_size * self.hr_feature_pool_size
        self.hr_query_encoder = nn.Sequential(
            nn.LayerNorm(self.hr_patch_descriptor_dim),
            nn.Linear(self.hr_patch_descriptor_dim, self.hr_rescore_dim),
            nn.GELU(),
            nn.Linear(self.hr_rescore_dim, self.hr_rescore_dim),
        )
        self.hr_key_encoder = nn.Sequential(
            nn.LayerNorm(self.hr_patch_descriptor_dim),
            nn.Linear(self.hr_patch_descriptor_dim, self.hr_rescore_dim),
            nn.GELU(),
            nn.Linear(self.hr_rescore_dim, self.hr_rescore_dim),
        )
        self.hr_pair_rescorer = nn.Sequential(
            nn.Linear(self.hr_rescore_dim * 4 + 1, self.hr_rescore_hidden_dim),
            nn.GELU(),
            nn.Linear(self.hr_rescore_hidden_dim, 1),
        )
        nn.init.zeros_(self.hr_pair_rescorer[-1].weight)
        nn.init.zeros_(self.hr_pair_rescorer[-1].bias)
        self.hr_residual_refiner = (
            HRResidualRefiner(
                in_channels=13,
                hidden_channels=self.hr_refiner_channels,
                num_blocks=self.hr_refiner_blocks,
                template_size=self.hr_refiner_template_size,
            )
            if self.use_hr_residual_refiner
            else None
        )
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

    def reparameterize(self):
        if self.hr_residual_refiner is not None and hasattr(self.hr_residual_refiner, "reparameterize"):
            self.hr_residual_refiner.reparameterize()
        return self

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

        refinement_scale = self.refinement_runtime_scale

        delta = refinement_scale * refinement_confidence * (out - base_hf_flat) * patch_mask
        delta = self.apply_paper_coherence(delta, sizes)
        out = preserve_patches_flat + delta
        out = out * patch_mask + preserve_patches_flat * (1 - patch_mask)
        self.last_output_patches_flat = out
        self.last_pixel_mask_flat = pixel_mask_flat
        self.last_refinement_confidence = refinement_confidence

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

    def reparameterize(self):
        self.depthwise, self.depthwise_bn = _fuse_conv_bn_pair(self.depthwise, self.depthwise_bn)
        self.pointwise, self.pointwise_bn = _fuse_conv_bn_pair(self.pointwise, self.pointwise_bn)
        if isinstance(self.proj, nn.Sequential) and len(self.proj) == 2:
            fused_proj, proj_bn = _fuse_conv_bn_pair(self.proj[0], self.proj[1])
            if isinstance(proj_bn, nn.Identity):
                self.proj = fused_proj
        return self


class RepDepthwiseSeparableBlock(nn.Module):
    """Depthwise-separable block with a reparameterizable multi-branch depthwise stage."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_residual: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.use_residual = use_residual
        self.activation = nn.ReLU(inplace=True)

        self.depthwise = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(self.in_channels)

        self.depthwise_scale = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            stride=self.stride,
            padding=0,
            groups=self.in_channels,
            bias=False,
        )
        self.depthwise_scale_bn = nn.BatchNorm2d(self.in_channels)
        nn.init.zeros_(self.depthwise_scale_bn.weight)
        nn.init.zeros_(self.depthwise_scale_bn.bias)

        self.depthwise_identity_bn = None
        if self.stride == 1:
            self.depthwise_identity_bn = nn.BatchNorm2d(self.in_channels)
            nn.init.zeros_(self.depthwise_identity_bn.weight)
            nn.init.zeros_(self.depthwise_identity_bn.bias)

        self.pointwise = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(self.out_channels)

        self.proj = None
        if self.use_residual and (self.stride != 1 or self.in_channels != self.out_channels):
            self.proj = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.depthwise_bn(self.depthwise(x))
        if self.depthwise_scale is not None:
            out = out + self.depthwise_scale_bn(self.depthwise_scale(x))
        if self.depthwise_identity_bn is not None:
            out = out + self.depthwise_identity_bn(x)
        out = self.activation(out)

        out = self.pointwise_bn(self.pointwise(out))
        if self.use_residual:
            if self.proj is not None:
                residual = self.proj(residual)
            out = out + residual
        return self.activation(out)

    def reparameterize(self):
        dw_kernel, dw_bias = _fuse_conv_bn_to_weight_bias(self.depthwise, self.depthwise_bn)
        scale_kernel, scale_bias = _fuse_conv_bn_to_weight_bias(self.depthwise_scale, self.depthwise_scale_bn)
        dw_kernel = dw_kernel + _pad_kernel_to_size(scale_kernel, target_size=3)
        dw_bias = dw_bias + scale_bias
        if self.depthwise_identity_bn is not None:
            id_kernel, id_bias = _fuse_identity_bn_to_weight_bias(
                num_channels=self.in_channels,
                bn=self.depthwise_identity_bn,
                kernel_size=3,
                groups=self.in_channels,
            )
            dw_kernel = dw_kernel + id_kernel
            dw_bias = dw_bias + id_bias

        fused_depthwise = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.in_channels,
            bias=True,
        ).to(device=dw_kernel.device, dtype=dw_kernel.dtype)
        fused_depthwise.weight.data.copy_(dw_kernel)
        fused_depthwise.bias.data.copy_(dw_bias)
        self.depthwise = fused_depthwise
        self.depthwise_bn = nn.Identity()
        self.depthwise_scale = None
        self.depthwise_scale_bn = None
        self.depthwise_identity_bn = None

        self.pointwise, self.pointwise_bn = _fuse_conv_bn_pair(self.pointwise, self.pointwise_bn)
        if isinstance(self.proj, nn.Sequential) and len(self.proj) == 2:
            fused_proj, proj_bn = _fuse_conv_bn_pair(self.proj[0], self.proj[1])
            if isinstance(proj_bn, nn.Identity):
                self.proj = fused_proj
        return self


class HRResidualRefiner(nn.Module):
    """Patch-grid HR residual corrector with compute tied to the LR query grid."""

    def __init__(
        self,
        in_channels: int = 13,
        hidden_channels: int = 32,
        num_blocks: int = 4,
        template_size: int = 4,
    ):
        super().__init__()
        self.template_size = max(1, int(template_size))
        patch_hidden = max(8, hidden_channels // 2)
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(in_channels, patch_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(patch_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(patch_hidden, patch_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(patch_hidden),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                DepthwiseSeparableBlock(
                    hidden_channels,
                    hidden_channels,
                    stride=1,
                    use_residual=True,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.patch_to_grid = nn.Linear(
            patch_hidden * self.template_size * self.template_size,
            hidden_channels,
        )
        self.out_conv = nn.Conv2d(
            hidden_channels,
            3 * self.template_size * self.template_size,
            kernel_size=1,
        )
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _pool_flat_patches(
        self,
        patches_flat: torch.Tensor,
        channels: int,
        patch_size: int,
    ) -> torch.Tensor:
        """Reshape flat non-overlapping patches and adaptively pool to a fixed template."""
        batch_size, num_patches, _ = patches_flat.shape
        patches = patches_flat.reshape(batch_size * num_patches, channels, patch_size, patch_size)
        return F.adaptive_avg_pool2d(patches, output_size=(self.template_size, self.template_size))

    def forward(
        self,
        candidate_patches_flat: torch.Tensor,
        base_patches_flat: torch.Tensor,
        transferred_hf_patches_flat: torch.Tensor,
        source_patches_flat: torch.Tensor,
        mask_patches_flat: torch.Tensor,
        patch_size: int,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, num_patches, _ = candidate_patches_flat.shape
        grid_h, grid_w = grid_size
        if num_patches != grid_h * grid_w:
            raise ValueError(
                f"HRResidualRefiner expected {grid_h * grid_w} patches, got {num_patches}"
            )

        pooled_inputs = torch.cat(
            [
                self._pool_flat_patches(candidate_patches_flat, channels=3, patch_size=patch_size),
                self._pool_flat_patches(base_patches_flat, channels=3, patch_size=patch_size),
                self._pool_flat_patches(transferred_hf_patches_flat, channels=3, patch_size=patch_size),
                self._pool_flat_patches(source_patches_flat, channels=3, patch_size=patch_size),
                self._pool_flat_patches(mask_patches_flat, channels=1, patch_size=patch_size),
            ],
            dim=1,
        )
        encoded = self.patch_encoder(pooled_inputs)
        encoded = encoded.flatten(start_dim=1)
        encoded = self.patch_to_grid(encoded)
        encoded = (
            encoded.view(batch_size, grid_h, grid_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        encoded = self.blocks(encoded)

        residual_template = self.out_conv(encoded)
        residual_template = (
            residual_template.permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size * num_patches, 3, self.template_size, self.template_size)
        )
        residual = F.interpolate(
            residual_template,
            size=(patch_size, patch_size),
            mode='bilinear',
            align_corners=False,
        )
        mask = mask_patches_flat.reshape(batch_size * num_patches, 1, patch_size, patch_size)
        residual = residual * mask
        return residual.reshape(batch_size, num_patches, 3 * patch_size * patch_size)

    def reparameterize(self):
        if isinstance(self.patch_encoder, nn.Sequential) and len(self.patch_encoder) >= 5:
            self.patch_encoder[0], self.patch_encoder[1] = _fuse_conv_bn_pair(
                self.patch_encoder[0], self.patch_encoder[1]
            )
            self.patch_encoder[3], self.patch_encoder[4] = _fuse_conv_bn_pair(
                self.patch_encoder[3], self.patch_encoder[4]
            )
        for block in self.blocks:
            if hasattr(block, "reparameterize"):
                block.reparameterize()
        return self


class PaperEncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block_cls: type[nn.Module] = DepthwiseSeparableBlock):
        super().__init__()
        self.down = block_cls(in_channels, out_channels, stride=2, use_residual=True)
        self.refine = block_cls(out_channels, out_channels, stride=1, use_residual=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.refine(x)

    def reparameterize(self):
        self.down.reparameterize()
        self.refine.reparameterize()
        return self


class PaperUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        block_cls: type[nn.Module] = DepthwiseSeparableBlock,
    ):
        super().__init__()
        self.block1 = block_cls(
            in_channels + skip_channels,
            out_channels,
            stride=1,
            use_residual=True,
        )
        self.block2 = block_cls(out_channels, out_channels, stride=1, use_residual=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x)
        return self.block2(x)

    def reparameterize(self):
        self.block1.reparameterize()
        self.block2.reparameterize()
        return self


class PaperCoarse(nn.Module):
    def __init__(self, channels=None, head_channels: int = 32, use_rep_blocks: bool = False, **kwargs):
        super().__init__()
        if kwargs:
            raise ValueError(f"Unsupported PaperCoarse arguments: {sorted(kwargs)}")

        if channels is None:
            channels = [64, 128, 256, 384, 512]
        if len(channels) != 5:
            raise ValueError(f"PaperCoarse expects 5 channel values, got {channels}")

        c0, c1, c2, c3, c4 = [int(c) for c in channels]
        self.feature_channels = [c0, c1, c2, c3, c4]
        self.use_rep_blocks = bool(use_rep_blocks)
        block_cls = RepDepthwiseSeparableBlock if self.use_rep_blocks else DepthwiseSeparableBlock

        self.stage0 = PaperEncoderStage(3, c0, block_cls=block_cls)
        self.stage1 = PaperEncoderStage(c0, c1, block_cls=block_cls)
        self.stage2 = PaperEncoderStage(c1, c2, block_cls=block_cls)
        self.stage3 = PaperEncoderStage(c2, c3, block_cls=block_cls)
        self.stage4 = PaperEncoderStage(c3, c4, block_cls=block_cls)

        self.up4 = PaperUpBlock(c4, c3, c3, block_cls=block_cls)
        self.up3 = PaperUpBlock(c3, c2, c2, block_cls=block_cls)
        self.up2 = PaperUpBlock(c2, c1, c1, block_cls=block_cls)
        self.up1 = PaperUpBlock(c1, c0, c0, block_cls=block_cls)
        self.head_channels = int(head_channels)
        self.head_block = block_cls(c0 + 3, self.head_channels, stride=1, use_residual=True)
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
        self.stage0.reparameterize()
        self.stage1.reparameterize()
        self.stage2.reparameterize()
        self.stage3.reparameterize()
        self.stage4.reparameterize()
        self.up4.reparameterize()
        self.up3.reparameterize()
        self.up2.reparameterize()
        self.up1.reparameterize()
        self.head_block.reparameterize()
        return self


COARSE_MODEL_REGISTRY = {
    "PaperCoarse": PaperCoarse,
}


class AttentionUpscaling(nn.Module):
    def __init__(self, patch_inpainting_module: "PatchInpainting"):
        super().__init__()
        self.patch_inpainting = patch_inpainting_module

    def _reshape_patch_tokens(self, patches: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Convert unfolded patches into per-patch 2D tensors."""
        batch_size, channels_times_area, n_h, n_w = patches.shape
        channels = channels_times_area // (patch_size * patch_size)
        return (
            patches.view(batch_size, channels, patch_size, patch_size, n_h, n_w)
            .permute(0, 4, 5, 1, 2, 3)
            .contiguous()
            .view(batch_size, n_h * n_w, channels, patch_size, patch_size)
        )

    def _encode_patch_descriptors(self, patch_tokens: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """Adaptive-pool arbitrary HR patch sizes to a fixed descriptor size."""
        batch_size, num_patches, channels, patch_h, patch_w = patch_tokens.shape
        pooled = F.adaptive_avg_pool2d(
            patch_tokens.reshape(batch_size * num_patches, channels, patch_h, patch_w),
            output_size=(
                self.patch_inpainting.hr_feature_pool_size,
                self.patch_inpainting.hr_feature_pool_size,
            ),
        )
        pooled = pooled.flatten(start_dim=1).view(batch_size, num_patches, -1)
        return encoder(pooled)

    def _rescore_topk_attention(
        self,
        hr_attn: torch.Tensor,
        hr_hf_patches: torch.Tensor,
        base_hf_patches: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Re-rank LR-proposed candidates using a small HR-specific scorer."""
        batch_size, num_queries, _ = hr_attn.shape
        topk_prior, topk_idx = torch.topk(hr_attn, k=topk, dim=-1)

        hr_hf_tokens = self._reshape_patch_tokens(
            hr_hf_patches,
            patch_size=int(math.sqrt(hr_hf_patches.shape[1] // self.patch_inpainting.stem_out_channels)),
        )
        base_hf_tokens = self._reshape_patch_tokens(
            base_hf_patches,
            patch_size=int(math.sqrt(base_hf_patches.shape[1] // self.patch_inpainting.stem_out_channels)),
        )
        key_embed_all = self._encode_patch_descriptors(hr_hf_tokens, self.patch_inpainting.hr_key_encoder)
        query_embed_all = self._encode_patch_descriptors(base_hf_tokens, self.patch_inpainting.hr_query_encoder)
        hr_hf_flat = hr_hf_patches.flatten(start_dim=2).transpose(1, 2)

        batch_idx = torch.arange(batch_size, device=hr_attn.device)
        rescored_chunks = []
        chunk_size = self.patch_inpainting.hr_query_chunk_size
        for start in range(0, num_queries, chunk_size):
            end = min(start + chunk_size, num_queries)
            chunk_len = end - start
            idx_chunk = topk_idx[:, start:end, :]
            prior_chunk = topk_prior[:, start:end, :]
            gather_batch = batch_idx.view(batch_size, 1, 1).expand(-1, chunk_len, topk)

            query_chunk = query_embed_all[:, start:end, :].unsqueeze(2).expand(-1, -1, topk, -1)
            key_chunk = key_embed_all[gather_batch, idx_chunk]
            pair_features = torch.cat(
                [
                    query_chunk,
                    key_chunk,
                    query_chunk - key_chunk,
                    query_chunk * key_chunk,
                    prior_chunk.unsqueeze(-1),
                ],
                dim=-1,
            )
            residual_scores = self.patch_inpainting.hr_pair_rescorer(pair_features).squeeze(-1)
            logits = prior_chunk.clamp_min(1e-8).log().float() + residual_scores.float()
            weights = F.softmax(logits, dim=-1).to(hr_hf_flat.dtype)
            key_hf_chunk = hr_hf_flat[gather_batch, idx_chunk]
            rescored_chunk = (weights.unsqueeze(-1) * key_hf_chunk).sum(dim=2)
            rescored_chunks.append(rescored_chunk)

        return torch.cat(rescored_chunks, dim=1)

    def forward(self, x_hr, x_lr_inpainted, attn_map, mask_hr=None):
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
        hr_base_blurred = self.patch_inpainting.final_gaussian_blur(x_hr_base)
        hr_base_patches, _ = self.patch_inpainting.unfold_native(x_hr_base, hr_patch_size)
        hr_base_patches_blurred, _ = self.patch_inpainting.unfold_native(hr_base_blurred, hr_patch_size)

        # Section 3.4 mixes only HR high frequencies extracted from the HR masked image.
        hr_blurred = self.patch_inpainting.final_gaussian_blur(x_hr)
        hr_patches_blurred, _ = self.patch_inpainting.unfold_native(hr_blurred, hr_patch_size)

        hr_hf_patches_full = hr_patches - hr_patches_blurred
        hr_base_hf_patches_full = hr_base_patches - hr_base_patches_blurred
        hr_hf_patches = hr_hf_patches_full.flatten(start_dim=2).transpose(1, 2)
        hr_base_patches_flat = hr_base_patches.flatten(start_dim=2).transpose(1, 2)
        hr_source_patches_flat = hr_patches.flatten(start_dim=2).transpose(1, 2)
        grid_size = (hr_h // hr_patch_size, hr_w // hr_patch_size)
        hr_mask_patches_flat = None

        hr_attn = attn_map.squeeze(1)
        if mask_hr is not None:
            hr_mask_patches, _ = self.patch_inpainting.unfold_native(mask_hr, hr_patch_size)
            hr_mask_patches_flat = hr_mask_patches.flatten(start_dim=2).transpose(1, 2)
            hr_mask_ratio = hr_mask_patches.mean(dim=1, keepdim=True).flatten(start_dim=2).squeeze(1)
            valid_hr_keys = (hr_mask_ratio <= 1e-6).to(hr_attn.dtype)
            masked_hr_attn = hr_attn * valid_hr_keys.unsqueeze(1)
            masked_hr_attn_sum = masked_hr_attn.sum(dim=-1, keepdim=True)
            normalized_hr_attn = masked_hr_attn / masked_hr_attn_sum.clamp_min(1e-8)
            hr_attn = torch.where(masked_hr_attn_sum > 1e-8, normalized_hr_attn, hr_attn)
        else:
            hr_mask_patches_flat = hr_hf_patches.new_zeros(
                hr_hf_patches.shape[0],
                hr_hf_patches.shape[1],
                hr_patch_size * hr_patch_size,
            )

        topk = min(self.patch_inpainting.hr_candidate_topk, hr_attn.size(-1))
        rescored_hr_hf_patches = self._rescore_topk_attention(
            hr_attn,
            hr_hf_patches_full,
            hr_base_hf_patches_full,
            topk=topk,
        )

        refinement_scale = self.patch_inpainting.refinement_runtime_scale

        # Reuse the LR branch confidence so HR transfer is selective rather than
        # copying every attended patch equally.
        confidence = self.patch_inpainting.last_refinement_confidence
        if confidence is not None:
            reconstructed_hr_hf_patches = refinement_scale * confidence * rescored_hr_hf_patches
        else:
            reconstructed_hr_hf_patches = refinement_scale * rescored_hr_hf_patches

        final_hr_patches = hr_base_patches_flat + reconstructed_hr_hf_patches
        if self.patch_inpainting.hr_residual_refiner is not None:
            hr_residual_patches = self.patch_inpainting.hr_residual_refiner(
                candidate_patches_flat=final_hr_patches,
                base_patches_flat=hr_base_patches_flat,
                transferred_hf_patches_flat=reconstructed_hr_hf_patches,
                source_patches_flat=hr_source_patches_flat,
                mask_patches_flat=hr_mask_patches_flat,
                patch_size=hr_patch_size,
                grid_size=grid_size,
            )
            final_hr_patches = final_hr_patches + hr_residual_patches

        final_hr_image = self.patch_inpainting.fold_native(
            final_hr_patches,
            (hr_h, hr_w),
            kernel_size=hr_patch_size,
            use_final_conv=False,
        )

        return final_hr_image


class InpaintingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        coarse_class_name = config['coarse_model'].get('class', 'PaperCoarse')
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
        """Apply any model-specific inference-time reparameterization."""
        if hasattr(self.coarse_model, 'reparameterize'):
            self.coarse_model.reparameterize()
        if hasattr(self.generator, 'reparameterize'):
            self.generator.reparameterize()
        return self
