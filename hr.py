"""High-resolution refinement and upscaling modules."""

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from blocks import DepthwiseSeparableBlock, fuse_conv_bn_pair

if TYPE_CHECKING:
    from patchmatch import PatchInpainting


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
            mode="bilinear",
            align_corners=False,
        )
        mask = mask_patches_flat.reshape(batch_size * num_patches, 1, patch_size, patch_size)
        residual = residual * mask
        return residual.reshape(batch_size, num_patches, 3 * patch_size * patch_size)

    def reparameterize(self):
        if isinstance(self.patch_encoder, nn.Sequential) and len(self.patch_encoder) >= 5:
            self.patch_encoder[0], self.patch_encoder[1] = fuse_conv_bn_pair(
                self.patch_encoder[0], self.patch_encoder[1]
            )
            self.patch_encoder[3], self.patch_encoder[4] = fuse_conv_bn_pair(
                self.patch_encoder[3], self.patch_encoder[4]
            )
        for block in self.blocks:
            if hasattr(block, "reparameterize"):
                block.reparameterize()
        return self


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

        x_hr_base = F.interpolate(x_lr_inpainted, size=(hr_h, hr_w), mode="bicubic", align_corners=False)

        hr_patch_size = self.patch_inpainting.kernel_size * scale_h

        hr_patches, _ = self.patch_inpainting.unfold_native(x_hr, hr_patch_size)
        hr_base_blurred = self.patch_inpainting.final_gaussian_blur(x_hr_base)
        hr_base_patches, _ = self.patch_inpainting.unfold_native(x_hr_base, hr_patch_size)
        hr_base_patches_blurred, _ = self.patch_inpainting.unfold_native(hr_base_blurred, hr_patch_size)

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
