from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AttentionUpscaling(nn.Module):
    def __init__(self, patch_inpainting_module):
        super().__init__()
        self.patch_inpainting = patch_inpainting_module

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
        source = x_hr if mask_hr is None else x_hr * (1 - mask_hr)
        source_blurred = self.patch_inpainting.final_gaussian_blur(source)

        source_patches, _ = self.patch_inpainting.extract_patches(
            source,
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
        known = x_hr.to(dtype=output.dtype) * (1 - mask_hr)
        return output * mask_hr + known
