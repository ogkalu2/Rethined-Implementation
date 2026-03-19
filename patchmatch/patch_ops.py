from __future__ import annotations

import torch
from torch.nn import functional as F


class PatchOpsMixin:
    def _pool_to_token_grid(self, feature_map: torch.Tensor, token_hw: tuple[int, int]) -> torch.Tensor:
        height, width = feature_map.shape[-2:]
        out_h, out_w = int(token_hw[0]), int(token_hw[1])
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"Invalid token grid size: {token_hw}.")
        if height % out_h == 0 and width % out_w == 0:
            kernel_h = height // out_h
            kernel_w = width // out_w
            return F.avg_pool2d(feature_map, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        return F.adaptive_avg_pool2d(feature_map, token_hw)

    def _compute_unfolding_weights(self, kernel_size: int, channels: int) -> torch.Tensor:
        weights = torch.eye(kernel_size * kernel_size, dtype=torch.float32)
        weights = weights.view(kernel_size * kernel_size, 1, kernel_size, kernel_size)
        return weights.repeat(channels, 1, 1, 1)

    def _get_unfolding_weights(
        self,
        kernel_size: int,
        channels: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if kernel_size == self.kernel_size:
            if channels == self.stem_out_channels:
                weights = self.unfolding_weights
            elif channels == 3:
                weights = self.unfolding_weights_image
            elif channels == 1:
                weights = self.unfolding_weights_mask
            else:
                weights = self._compute_unfolding_weights(kernel_size, channels)
        else:
            weights = self._compute_unfolding_weights(kernel_size, channels)
        return weights.to(device=device, dtype=dtype)

    def unfold_native(self, feature_map: torch.Tensor, kernel_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
        return self.extract_patches(feature_map, kernel_size, stride=kernel_size, padding=0)

    def extract_patches(
        self,
        feature_map: torch.Tensor,
        patch_size: int,
        *,
        stride: int,
        padding: int = 0,
        pad_mode: str = "reflect",
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        batch_size, channels, height, width = feature_map.shape
        patch_size = int(patch_size)
        stride = int(stride)
        padding = int(padding)
        if height % stride != 0 or width % stride != 0:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by stride {stride} for patch extraction."
            )

        n_h = height // stride
        n_w = width // stride

        if patch_size == stride and padding == 0:
            if self.use_conv_unfold:
                weights = self._get_unfolding_weights(
                    patch_size, channels, dtype=feature_map.dtype, device=feature_map.device
                )
                patches = F.conv2d(feature_map, weights, stride=patch_size, groups=channels)
            else:
                x = feature_map.view(batch_size, channels, n_h, patch_size, n_w, patch_size)
                x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
                patches = x.view(batch_size, channels * patch_size * patch_size, n_h, n_w)
            return patches, (height, width)

        if padding > 0:
            feature_map = F.pad(feature_map, (padding, padding, padding, padding), mode=pad_mode)

        patches = F.unfold(feature_map, kernel_size=patch_size, stride=stride)
        patches = patches.view(batch_size, channels * patch_size * patch_size, n_h, n_w)

        return patches, (height, width)

    def fold_native(
        self,
        patches: torch.Tensor,
        output_size: tuple[int, int],
        *,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        use_window: bool = False,
    ) -> torch.Tensor:
        kernel_size = int(kernel_size)
        stride = kernel_size if stride is None else int(stride)
        padding = int(padding)
        if patches.dim() == 4:
            batch_size, patch_dim, n_h, n_w = patches.shape
            cols = patches.view(batch_size, patch_dim, n_h * n_w)
        elif patches.dim() == 3:
            batch_size, num_patches, patch_dim = patches.shape
            cols = patches.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported patch tensor shape: {tuple(patches.shape)}")

        channels = patch_dim // (kernel_size * kernel_size)
        if channels * kernel_size * kernel_size != patch_dim:
            raise ValueError(
                f"Patch dimension {patch_dim} is not divisible by kernel footprint {kernel_size * kernel_size}."
            )

        # Fast path for non-overlapping patch reconstruction.
        if stride == kernel_size and padding == 0 and not use_window and not torch.onnx.is_in_onnx_export():
            out_h, out_w = int(output_size[0]), int(output_size[1])
            if out_h % kernel_size != 0 or out_w % kernel_size != 0:
                raise ValueError(
                    f"Output size {output_size} must be divisible by kernel_size {kernel_size} "
                    "for no-overlap PixelShuffle reconstruction."
                )
            expected_n_h = out_h // kernel_size
            expected_n_w = out_w // kernel_size

            if patches.dim() == 4:
                if n_h != expected_n_h or n_w != expected_n_w:
                    raise ValueError(
                        "Patch grid shape does not match output size for no-overlap reconstruction: "
                        f"grid {(n_h, n_w)} vs expected {(expected_n_h, expected_n_w)}."
                    )
                patch_map = patches
            else:
                if num_patches != expected_n_h * expected_n_w:
                    raise ValueError(
                        "Number of patches does not match output size for no-overlap reconstruction: "
                        f"got {num_patches}, expected {expected_n_h * expected_n_w}."
                    )
                patch_map = cols.view(batch_size, patch_dim, expected_n_h, expected_n_w)

            return F.pixel_shuffle(patch_map, kernel_size)

        padded_output = (output_size[0] + 2 * padding, output_size[1] + 2 * padding)

        if use_window and (stride != kernel_size or padding > 0):
            window = torch.hann_window(kernel_size, periodic=False, device=cols.device, dtype=cols.dtype)
            window_2d = torch.outer(window, window).clamp_min(1e-3)
            window_2d = window_2d / window_2d.max().clamp_min(1e-8)
            weight_cols = window_2d.reshape(1, 1, -1).repeat(1, channels, 1).reshape(1, -1, 1)
            weighted_cols = cols * weight_cols
            output = F.fold(weighted_cols, output_size=padded_output, kernel_size=kernel_size, stride=stride)
            norm = F.fold(
                cols.new_ones((cols.shape[0], 1, cols.shape[-1])) * weight_cols,
                output_size=padded_output,
                kernel_size=kernel_size,
                stride=stride,
            )
            output = output / norm.clamp_min(1e-6)
        else:
            output = F.fold(cols, output_size=padded_output, kernel_size=kernel_size, stride=stride)

        if padding > 0:
            output = output[..., padding:-padding, padding:-padding]
        return output

    def _get_normalized_token_coords(
        self,
        token_hw: tuple[int, int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        height, width = token_hw
        ys = torch.linspace(-1.0, 1.0, steps=height, dtype=dtype, device=device)
        xs = torch.linspace(-1.0, 1.0, steps=width, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)

    def _flatten_patch_map(self, patch_map: torch.Tensor) -> torch.Tensor:
        return patch_map.flatten(start_dim=2).transpose(1, 2)

    def flatten_query_mask(self, mask: torch.Tensor) -> torch.Tensor:
        query_mask_patch_map, _ = self.unfold_native(mask, self.kernel_size)
        return (query_mask_patch_map.amax(dim=1) > 0).to(dtype=mask.dtype).flatten(start_dim=1)
