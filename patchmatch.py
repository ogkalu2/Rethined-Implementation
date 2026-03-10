"""Paper-aligned NeuralPatchMatch refinement modules."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from blocks import NativeGaussianBlur2d


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        d_v: int,
        n_head: int,
        dropout: float,
        d_qk: int,
    ):
        super().__init__()
        self.d_v = int(d_v)
        self.n_head = int(n_head)
        self.d_k = int(d_qk)
        self.dropout = nn.Dropout(float(dropout))
        self.w_qs = nn.Linear(embed_dim, self.n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(embed_dim, self.n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(self.d_v, self.n_head * self.d_v, bias=False)
        self.fc = nn.Linear(self.n_head * self.d_v, self.d_v, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        post_softmax_mask: torch.Tensor | None = None,
        direct_patch_mixing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        q_proj = self.w_qs(q).view(batch_size, len_q, self.n_head, self.d_k).transpose(1, 2)
        k_proj = self.w_ks(k).view(batch_size, len_k, self.n_head, self.d_k).transpose(1, 2)

        if direct_patch_mixing:
            if self.n_head != 1:
                raise ValueError("Paper-style direct patch mixing expects a single attention head.")
            v_proj = v.unsqueeze(1)
        else:
            v_proj = self.w_vs(v).view(batch_size, len_v, self.n_head, self.d_v).transpose(1, 2)

        attn = torch.matmul(q_proj / (self.d_k ** 0.5), k_proj.transpose(2, 3))
        attn = F.softmax(attn.float(), dim=-1).to(v.dtype)
        if post_softmax_mask is not None:
            attn = attn * post_softmax_mask.to(dtype=attn.dtype)
        attn = self.dropout(attn)

        mixed = torch.matmul(attn, v_proj)
        output = mixed.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        if not direct_patch_mixing:
            output = self.fc(output)
        return output, attn


class PatchInpainting(nn.Module):
    def __init__(
        self,
        *,
        kernel_size: int,
        nheads: int,
        stem_out_stride: int = 1,
        stem_out_channels: int = 3,
        use_positional_encoding: bool = True,
        image_size: int = 512,
        embed_dim: int = 2048,
        dropout: float = 0.1,
        feature_i: int = 2,
        feature_dim: int = 128,
        concat_features: bool = True,
        attention_masking: bool = True,
        final_conv: bool = False,
        positional_grid_size: int = 32,
        model,
    ):
        super().__init__()

        self.kernel_size = int(kernel_size)
        self.nheads = int(nheads)
        self.stem_out_stride = int(stem_out_stride)
        self.stem_out_channels = int(stem_out_channels)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.feature_i = int(feature_i)
        self.feature_dim = int(feature_dim)
        self.concat_features = bool(concat_features)
        self.attention_masking = bool(attention_masking)
        self.final_conv = bool(final_conv)
        self.image_size = int(image_size)
        self.token_grid_size = self.image_size // self.stem_out_stride // self.kernel_size
        self.patch_value_dim = self.stem_out_channels * self.kernel_size * self.kernel_size
        self.patch_token_dim = self.patch_value_dim + (self.feature_dim if self.concat_features else 0)
        self.positional_grid_size = max(1, min(int(positional_grid_size), self.token_grid_size))

        self.encoder_decoder = model
        self.final_gaussian_blur = NativeGaussianBlur2d((7, 7), sigma=(2.01, 2.01))
        self.multihead_attention = MultiHeadAttention(
            embed_dim=self.patch_token_dim,
            d_v=self.patch_value_dim,
            n_head=self.nheads,
            dropout=float(dropout),
            d_qk=int(embed_dim),
        )
        self.positionalencoding = (
            nn.Parameter(
                torch.zeros(
                    1,
                    self.patch_token_dim,
                    self.positional_grid_size,
                    self.positional_grid_size,
                )
            )
            if self.use_positional_encoding
            else None
        )
        self.paper_coherence_layer = (
            nn.Conv2d(
                self.patch_value_dim,
                self.patch_value_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
            if self.final_conv
            else None
        )
        if self.paper_coherence_layer is not None:
            nn.init.zeros_(self.paper_coherence_layer.weight)
            nn.init.zeros_(self.paper_coherence_layer.bias)

        self.register_buffer(
            "unfolding_weights",
            self._compute_unfolding_weights(self.kernel_size, self.stem_out_channels),
            persistent=False,
        )
        self.register_buffer(
            "unfolding_weights_image",
            self._compute_unfolding_weights(self.kernel_size, 3),
            persistent=False,
        )
        self.register_buffer(
            "unfolding_weights_mask",
            self._compute_unfolding_weights(self.kernel_size, 1),
            persistent=False,
        )

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
        batch_size, channels, height, width = feature_map.shape
        weights = self._get_unfolding_weights(
            kernel_size,
            channels,
            dtype=feature_map.dtype,
            device=feature_map.device,
        )
        patches = F.conv2d(feature_map, weights, stride=kernel_size, groups=channels)
        n_h = height // kernel_size
        n_w = width // kernel_size
        patches = patches.view(batch_size, channels * kernel_size * kernel_size, n_h, n_w)
        return patches, (height, width)

    def fold_native(
        self,
        patches: torch.Tensor,
        output_size: tuple[int, int],
        *,
        kernel_size: int,
    ) -> torch.Tensor:
        n_h = output_size[0] // kernel_size
        n_w = output_size[1] // kernel_size
        if patches.dim() == 3:
            patches = patches.transpose(1, 2).contiguous().view(patches.shape[0], -1, n_h, n_w)
        return F.pixel_shuffle(patches, upscale_factor=kernel_size)

    def build_paper_attention_mask(self, patch_mask_flat: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches = patch_mask_flat.shape
        is_masked_q = patch_mask_flat.unsqueeze(-1)
        is_valid_k = (1.0 - patch_mask_flat).unsqueeze(1)
        eye = torch.eye(num_patches, device=patch_mask_flat.device, dtype=patch_mask_flat.dtype).unsqueeze(0)
        allowed = (1.0 - is_masked_q) * eye + is_masked_q * is_valid_k
        return allowed.unsqueeze(1)

    def get_positional_encoding(self) -> torch.Tensor | None:
        if self.positionalencoding is None:
            return None
        if self.positionalencoding.shape[-2:] != (self.token_grid_size, self.token_grid_size):
            pos = F.interpolate(
                self.positionalencoding,
                size=(self.token_grid_size, self.token_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            pos = self.positionalencoding
        return pos.flatten(start_dim=2).transpose(1, 2)

    def reparameterize(self):
        return self

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        coarse_raw, features = self.encoder_decoder(image)

        patch_map, output_size = self.unfold_native(coarse_raw, self.kernel_size)
        source_patch_map, _ = self.unfold_native(image, self.kernel_size)
        mask_patch_map, _ = self.unfold_native(mask, self.kernel_size)
        patch_mask_flat = (mask_patch_map.amax(dim=1) > 0).to(dtype=patch_map.dtype).flatten(start_dim=1)

        token_map = patch_map
        if self.concat_features:
            coarse_features = F.interpolate(
                features[self.feature_i],
                size=patch_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            token_map = torch.cat([token_map, coarse_features], dim=1)

        input_tokens = token_map.flatten(start_dim=2).transpose(1, 2)
        positional_encoding = self.get_positional_encoding()
        if positional_encoding is not None:
            input_tokens = input_tokens + positional_encoding

        # The paper mixes source LR patches with attention learned from coarse tokens.
        patch_values = source_patch_map.flatten(start_dim=2).transpose(1, 2)
        attention_mask = (
            self.build_paper_attention_mask(patch_mask_flat) if self.attention_masking else None
        )
        mixed_patches_flat, masked_attention = self.multihead_attention(
            input_tokens,
            input_tokens,
            patch_values,
            post_softmax_mask=attention_mask,
            direct_patch_mixing=True,
        )

        mixed_patch_map = mixed_patches_flat.transpose(1, 2).contiguous().view_as(patch_map)
        if self.paper_coherence_layer is not None:
            mixed_patch_map = mixed_patch_map + self.paper_coherence_layer(mixed_patch_map)
        refined = self.fold_native(
            mixed_patch_map,
            output_size,
            kernel_size=self.kernel_size,
        )
        refined = refined * mask + image * (1 - mask)

        return refined, masked_attention, coarse_raw
