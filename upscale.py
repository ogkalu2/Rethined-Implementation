from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AttentionUpscaling(nn.Module):
    def __init__(self, patch_inpainting_module):
        super().__init__()
        self.patch_inpainting = patch_inpainting_module

    def _remask_attention_for_hr(
        self,
        attn_weights: torch.Tensor,
        mask_hr: torch.Tensor | None,
        *,
        hr_stride: int,
        hr_patch_size: int,
        hr_padding: int,
    ) -> torch.Tensor:
        if mask_hr is None:
            return attn_weights

        query_mask_patch_map, _ = self.patch_inpainting.extract_patches(
            mask_hr,
            hr_stride,
            stride=hr_stride,
            padding=0,
            pad_mode="constant",
        )
        key_mask_patch_map, _ = self.patch_inpainting.extract_patches(
            mask_hr,
            hr_patch_size,
            stride=hr_stride,
            padding=hr_padding,
            pad_mode="constant",
        )
        query_mask_flat = (query_mask_patch_map.amax(dim=1) > 0.5).flatten(start_dim=1)
        key_valid_flat = (key_mask_patch_map.amax(dim=1) == 0).flatten(start_dim=1)
        expected_shape = (query_mask_flat.shape[1], key_valid_flat.shape[1])
        if attn_weights.shape[-2:] != expected_shape:
            raise ValueError(
                "AttentionUpscaling HR remasking expected attention weights with shape "
                f"{expected_shape}, got {tuple(attn_weights.shape[-2:])}."
            )

        filtered_weights = attn_weights * key_valid_flat.unsqueeze(1).to(dtype=attn_weights.dtype)
        filtered_sums = filtered_weights.sum(dim=-1, keepdim=True)
        masked_queries = query_mask_flat.unsqueeze(-1)
        normalized_filtered = filtered_weights / filtered_sums.clamp_min(1e-8)
        return torch.where(masked_queries, normalized_filtered, attn_weights)

    def forward(
        self,
        x_hr: torch.Tensor,
        x_lr_inpainted: torch.Tensor,
        attn_map: torch.Tensor,
        mask_hr: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
                "AttentionUpscaling expects a single masked attention map with shape "
                f"(B, 1, N, N); got {tuple(attn_map.shape)}"
            )

        hr_stride = self.patch_inpainting.kernel_size * scale_h
        hr_patch_size = self.patch_inpainting.value_patch_size * scale_h
        hr_padding = self.patch_inpainting.value_patch_padding * scale_h
        hr_base = F.interpolate(x_lr_inpainted, size=(hr_h, hr_w), mode="bicubic", align_corners=False)
        if mask_hr is None:
            source_known = x_hr
            source_context = x_hr
        else:
            mask_hr = mask_hr.to(dtype=x_hr.dtype)
            source_known = x_hr * (1 - mask_hr)
            # Fill the hole with the LR prediction before computing HR high-frequency residuals.
            # Blurring a zero-filled hole injects artificial edge energy that gets copied into the
            # missing region as tiled artifacts.
            source_context = source_known + hr_base.to(dtype=x_hr.dtype) * mask_hr
        source_blurred = self.patch_inpainting.final_gaussian_blur(source_context)

        source_patches, _ = self.patch_inpainting.extract_patches(
            source_context,
            hr_patch_size,
            stride=hr_stride,
            padding=hr_padding,
        )
        source_blurred_patches, _ = self.patch_inpainting.extract_patches(
            source_blurred,
            hr_patch_size,
            stride=hr_stride,
            padding=hr_padding,
        )
        source_hf_flat = (source_patches - source_blurred_patches).flatten(start_dim=2).transpose(1, 2)
        attn_weights = attn_map.squeeze(1)
        attn_weights = self._remask_attention_for_hr(
            attn_weights,
            mask_hr,
            hr_stride=hr_stride,
            hr_patch_size=hr_patch_size,
            hr_padding=hr_padding,
        )
        compute_dtype = torch.promote_types(attn_weights.dtype, source_hf_flat.dtype)
        if attn_weights.dtype != compute_dtype:
            attn_weights = attn_weights.to(dtype=compute_dtype)
        if source_hf_flat.dtype != compute_dtype:
            source_hf_flat = source_hf_flat.to(dtype=compute_dtype)
        reconstructed_hf_flat = torch.matmul(attn_weights, source_hf_flat)
        reconstructed_hf = self.patch_inpainting.fold_native(
            reconstructed_hf_flat,
            (hr_h, hr_w),
            kernel_size=hr_patch_size,
            stride=hr_stride,
            padding=hr_padding,
            use_window=hr_padding > 0,
        )
        if hr_base.dtype != reconstructed_hf.dtype:
            hr_base = hr_base.to(dtype=reconstructed_hf.dtype)
        output = hr_base + reconstructed_hf
        if mask_hr is None:
            return output
        mask_hr = mask_hr.to(dtype=output.dtype)
        known = source_known.to(dtype=output.dtype)
        return output * mask_hr + known
