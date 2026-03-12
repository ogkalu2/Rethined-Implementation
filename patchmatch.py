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
        attention_temperature: float = 1.0,
        attention_top_k: int | None = None,
        attention_selection: str = "softmax",
        attention_gumbel_tau: float = 1.0,
    ):
        super().__init__()
        self.d_v = int(d_v)
        self.n_head = int(n_head)
        self.d_k = int(d_qk)
        self.dropout = nn.Dropout(float(dropout))
        self.attention_temperature = float(attention_temperature)
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive.")
        self.attention_top_k = None if attention_top_k is None else int(attention_top_k)
        if self.attention_top_k is not None and self.attention_top_k <= 0:
            self.attention_top_k = None
        self.attention_selection = str(attention_selection).lower()
        if self.attention_selection not in {"softmax", "gumbel", "argmax"}:
            raise ValueError(
                "attention_selection must be one of {'softmax', 'gumbel', 'argmax'}."
            )
        self.attention_gumbel_tau = float(attention_gumbel_tau)
        if self.attention_gumbel_tau <= 0:
            raise ValueError("attention_gumbel_tau must be positive.")
        self.w_qs = nn.Linear(embed_dim, self.n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(embed_dim, self.n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(self.d_v, self.n_head * self.d_v, bias=False)
        self.fc = nn.Linear(self.n_head * self.d_v, self.d_v, bias=False)

    def _build_masked_query_selector(
        self,
        attn_logits: torch.Tensor,
        query_mask_flat: torch.Tensor | None,
    ) -> torch.Tensor:
        if query_mask_flat is None:
            return torch.ones(
                (attn_logits.shape[0], 1, attn_logits.shape[2], 1),
                dtype=torch.bool,
                device=attn_logits.device,
            )
        return (query_mask_flat > 0.5).unsqueeze(1).unsqueeze(-1)

    def _restrict_attention_logits(
        self,
        attn_logits: torch.Tensor,
        query_mask_flat: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.attention_top_k is None or self.attention_top_k >= attn_logits.shape[-1]:
            return attn_logits

        top_k = min(self.attention_top_k, attn_logits.shape[-1])
        _, topk_indices = torch.topk(attn_logits, k=top_k, dim=-1)
        keep_mask = torch.zeros_like(attn_logits, dtype=torch.bool)
        keep_mask.scatter_(-1, topk_indices, True)

        masked_queries = self._build_masked_query_selector(attn_logits, query_mask_flat)
        keep_mask = torch.where(masked_queries, keep_mask, torch.ones_like(keep_mask))
        return attn_logits.masked_fill(~keep_mask, torch.finfo(attn_logits.dtype).min)

    def _hard_attention_from_logits(self, attn_logits: torch.Tensor) -> torch.Tensor:
        attn = torch.zeros_like(attn_logits)
        top_indices = attn_logits.argmax(dim=-1, keepdim=True)
        attn.scatter_(-1, top_indices, 1.0)
        return attn

    def _normalize_attention_logits(self, attn_logits: torch.Tensor) -> torch.Tensor:
        if self.attention_selection == "softmax":
            return F.softmax(attn_logits, dim=-1)
        if self.attention_selection == "gumbel":
            if self.training:
                return F.gumbel_softmax(
                    attn_logits,
                    tau=self.attention_gumbel_tau,
                    hard=True,
                    dim=-1,
                )
            return self._hard_attention_from_logits(attn_logits)
        return self._hard_attention_from_logits(attn_logits)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        post_softmax_mask: torch.Tensor | None = None,
        direct_patch_mixing: bool = False,
        query_mask_flat: torch.Tensor | None = None,
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

        attn = torch.matmul(q_proj / (self.d_k ** 0.5), k_proj.transpose(2, 3)).float()
        if self.attention_temperature != 1.0:
            attn = attn / self.attention_temperature

        if post_softmax_mask is not None:
            attn = attn.masked_fill(post_softmax_mask == 0, torch.finfo(attn.dtype).min)

        attn = self._restrict_attention_logits(attn, query_mask_flat)
        attn = self._normalize_attention_logits(attn).to(v.dtype)
        # Direct patch mixing behaves like weighted image reconstruction, so dropping
        # sparse attention weights introduces random dark artifacts instead of useful regularization.
        if not direct_patch_mixing:
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
        value_patch_size: int | None = None,
        attention_temperature: float = 1.0,
        attention_top_k: int | None = None,
        attention_selection: str = "softmax",
        attention_gumbel_tau: float = 1.0,
        matching_descriptor_dim: int | None = None,
        match_coarse_rgb: bool = True,
        detach_coarse_rgb: bool = False,
        value_source: str = "rgb",
        fusion_mode: str = "replace",
        fusion_hidden_channels: int = 32,
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
        use_conv_unfold: bool = False,
        model,
    ):
        super().__init__()
        self.use_conv_unfold = use_conv_unfold
        self.kernel_size = int(kernel_size)
        self.value_patch_size = self.kernel_size if value_patch_size is None else int(value_patch_size)
        self.nheads = int(nheads)
        self.stem_out_stride = int(stem_out_stride)
        self.stem_out_channels = int(stem_out_channels)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.feature_i = int(feature_i)
        self.feature_dim = int(feature_dim)
        self.matching_descriptor_dim = (
            None if matching_descriptor_dim is None else int(matching_descriptor_dim)
        )
        self.match_coarse_rgb = bool(match_coarse_rgb)
        self.detach_coarse_rgb = bool(detach_coarse_rgb)
        self.value_source = str(value_source).lower()
        if self.value_source not in {"rgb", "high_freq_residual"}:
            raise ValueError("value_source must be either 'rgb' or 'high_freq_residual'.")
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in {"replace", "add", "gate"}:
            raise ValueError("fusion_mode must be one of {'replace', 'add', 'gate'}.")
        self.fusion_hidden_channels = int(fusion_hidden_channels)
        if self.fusion_hidden_channels <= 0:
            raise ValueError("fusion_hidden_channels must be positive.")
        self.concat_features = bool(concat_features)
        self.attention_masking = bool(attention_masking)
        self.final_conv = bool(final_conv)
        self.image_size = int(image_size)
        if self.value_patch_size < self.kernel_size:
            raise ValueError("value_patch_size must be greater than or equal to kernel_size.")
        if (self.value_patch_size - self.kernel_size) % 2 != 0:
            raise ValueError("value_patch_size - kernel_size must be even for centered overlap-add padding.")
        self.value_patch_padding = (self.value_patch_size - self.kernel_size) // 2
        self.token_grid_size = self.image_size // self.stem_out_stride // self.kernel_size
        self.query_patch_dim = self.stem_out_channels * self.kernel_size * self.kernel_size
        self.value_patch_dim = self.stem_out_channels * self.value_patch_size * self.value_patch_size
        self.matching_input_dim = 0
        if self.match_coarse_rgb:
            self.matching_input_dim += self.query_patch_dim
        if self.concat_features:
            self.matching_input_dim += self.feature_dim
        if self.matching_input_dim == 0:
            raise ValueError("Matching must use coarse RGB patches, coarse features, or both.")
        self.patch_token_dim = (
            self.matching_input_dim
            if self.matching_descriptor_dim is None
            else self.matching_descriptor_dim
        )
        self.positional_grid_size = max(1, min(int(positional_grid_size), self.token_grid_size))

        self.encoder_decoder = model
        self.final_gaussian_blur = NativeGaussianBlur2d((7, 7), sigma=(2.01, 2.01))
        self.matching_descriptor_head = None
        if self.matching_descriptor_dim is not None:
            hidden_dim = max(self.matching_input_dim, self.matching_descriptor_dim)
            self.matching_descriptor_head = nn.Sequential(
                nn.Conv2d(
                    self.matching_input_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="reflect",
                    bias=False,
                ),
                nn.GELU(),
                nn.Conv2d(hidden_dim, self.matching_descriptor_dim, kernel_size=1, stride=1, bias=False),
            )
        self.multihead_attention = MultiHeadAttention(
            embed_dim=self.patch_token_dim,
            d_v=self.value_patch_dim,
            n_head=self.nheads,
            dropout=float(dropout),
            d_qk=int(embed_dim),
            attention_temperature=float(attention_temperature),
            attention_top_k=attention_top_k,
            attention_selection=attention_selection,
            attention_gumbel_tau=float(attention_gumbel_tau),
        )
        self.pre_attention_norm = nn.LayerNorm(self.patch_token_dim)
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
        self.fusion_gate = None
        if self.fusion_mode == "gate":
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(
                    10,
                    self.fusion_hidden_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="reflect",
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.fusion_hidden_channels,
                    1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="reflect",
                ),
            )
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.constant_(self.fusion_gate[-1].bias, -2.0)
        self.coherence_layer = (
            nn.Conv2d(
                3,
                3,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
            if self.final_conv
            else None
        )
        if self.coherence_layer is not None:
            nn.init.zeros_(self.coherence_layer.weight)
            nn.init.zeros_(self.coherence_layer.bias)

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
                patches = patches.view(batch_size, channels * patch_size * patch_size, n_h, n_w)
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

    def direct_patch_mix_masked_queries(
        self,
        input_tokens: torch.Tensor,
        patch_values: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        default_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_patches, _ = input_tokens.shape
        mixed = patch_values.clone() if default_tokens is None else default_tokens.clone()
        eye = torch.eye(num_patches, device=patch_values.device, dtype=patch_values.dtype)
        dense_attn = eye.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        masked_queries = query_mask_flat > 0.5
        valid_keys = key_valid_flat > 0.5

        for batch_idx in range(batch_size):
            query_indices = masked_queries[batch_idx].nonzero(as_tuple=False).flatten()
            if query_indices.numel() == 0:
                continue

            key_indices = valid_keys[batch_idx].nonzero(as_tuple=False).flatten()
            replacement_rows = patch_values.new_zeros((query_indices.numel(), num_patches))
            if key_indices.numel() > 0:
                query_tokens = input_tokens[batch_idx : batch_idx + 1, query_indices]
                key_tokens = input_tokens[batch_idx : batch_idx + 1, key_indices]
                value_tokens = patch_values[batch_idx : batch_idx + 1, key_indices]
                mixed_queries, masked_attention = self.multihead_attention(
                    query_tokens,
                    key_tokens,
                    value_tokens,
                    direct_patch_mixing=True,
                )
                mixed_queries = mixed_queries.squeeze(0).to(dtype=mixed.dtype)
                masked_attention = masked_attention.squeeze(0).squeeze(0).to(dtype=replacement_rows.dtype)
                mixed[batch_idx].index_copy_(0, query_indices, mixed_queries)
                replacement_rows[:, key_indices] = masked_attention

            dense_attn[batch_idx, 0].index_copy_(0, query_indices, replacement_rows)

        return mixed, dense_attn

    def build_paper_attention_mask(
        self,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, num_patches = query_mask_flat.shape
        is_masked_q = query_mask_flat.unsqueeze(-1)
        if key_valid_flat is None:
            key_valid_flat = 1.0 - query_mask_flat
        is_valid_k = key_valid_flat.unsqueeze(1)
        eye = torch.eye(num_patches, device=query_mask_flat.device, dtype=query_mask_flat.dtype).unsqueeze(0)
        allowed = (1.0 - is_masked_q) * eye + is_masked_q * is_valid_k
        return allowed.unsqueeze(1)

    def flatten_query_mask(self, mask: torch.Tensor) -> torch.Tensor:
        query_mask_patch_map, _ = self.unfold_native(mask, self.kernel_size)
        return (query_mask_patch_map.amax(dim=1) > 0).to(dtype=mask.dtype).flatten(start_dim=1)

    def summarize_attention(
        self,
        attn_map: torch.Tensor,
        query_mask_flat: torch.Tensor,
    ) -> dict[str, float]:
        probs = attn_map.detach()
        if probs.dim() == 4:
            probs = probs.mean(dim=1)
        if probs.dim() != 3:
            raise ValueError(f"Expected attention map with 3 or 4 dims, got {tuple(attn_map.shape)}")

        masked_queries = query_mask_flat > 0.5
        masked_query_ratio = masked_queries.float().mean().item()
        if not masked_queries.any():
            return {
                "attention_top1": 1.0,
                "attention_top4": 1.0,
                "attention_entropy": 0.0,
                "attention_masked_ratio": masked_query_ratio,
            }

        masked_probs = probs[masked_queries]
        top1 = masked_probs.max(dim=-1).values.mean().item()
        top4_k = min(4, masked_probs.shape[-1])
        top4 = masked_probs.topk(k=top4_k, dim=-1).values.sum(dim=-1).mean().item()
        entropy = (
            -(masked_probs.clamp_min(1e-8) * masked_probs.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
        )
        return {
            "attention_top1": top1,
            "attention_top4": top4,
            "attention_entropy": entropy,
            "attention_masked_ratio": masked_query_ratio,
        }

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

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        value_image: torch.Tensor | None = None,
    ):
        coarse_raw, features = self.encoder_decoder(image)
        known_image = image if value_image is None else value_image
        coarse_composite = coarse_raw * mask + known_image * (1 - mask)

        patch_map, output_size = self.unfold_native(coarse_raw, self.kernel_size)
        default_patch_values = None
        if self.value_source == "high_freq_residual":
            value_base = known_image - self.final_gaussian_blur(known_image)
        else:
            value_base = known_image

        source_patch_map, _ = self.extract_patches(
            value_base,
            self.value_patch_size,
            stride=self.kernel_size,
            padding=self.value_patch_padding,
        )
        query_mask_flat = self.flatten_query_mask(mask)
        key_mask_patch_map, _ = self.extract_patches(
            mask,
            self.value_patch_size,
            stride=self.kernel_size,
            padding=self.value_patch_padding,
        )
        key_valid_flat = (key_mask_patch_map.amax(dim=1) == 0).to(dtype=patch_map.dtype).flatten(start_dim=1)

        token_inputs = []
        if self.match_coarse_rgb:
            coarse_match_map = patch_map.detach() if self.detach_coarse_rgb else patch_map
            token_inputs.append(coarse_match_map)
        if self.concat_features:
            coarse_features = F.interpolate(
                features[self.feature_i],
                size=patch_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            token_inputs.append(coarse_features)

        token_map = token_inputs[0] if len(token_inputs) == 1 else torch.cat(token_inputs, dim=1)
        if self.matching_descriptor_head is not None:
            token_map = self.matching_descriptor_head(token_map)

        input_tokens = token_map.flatten(start_dim=2).transpose(1, 2)
        positional_encoding = self.get_positional_encoding()
        if positional_encoding is not None:
            input_tokens = input_tokens + positional_encoding
        input_tokens = self.pre_attention_norm(input_tokens)

        # The paper mixes source LR patches with attention learned from coarse tokens.
        patch_values = source_patch_map.flatten(start_dim=2).transpose(1, 2)
        if self.value_source == "high_freq_residual":
            default_patch_values = torch.zeros_like(patch_values)
        if self.attention_masking:
            mixed_patches_flat, masked_attention = self.direct_patch_mix_masked_queries(
                input_tokens,
                patch_values,
                query_mask_flat,
                key_valid_flat,
                default_tokens=default_patch_values,
            )
        else:
            mixed_patches_flat, masked_attention = self.multihead_attention(
                input_tokens,
                input_tokens,
                patch_values,
                direct_patch_mixing=True,
                query_mask_flat=query_mask_flat,
            )

        mixed_image = self.fold_native(
            mixed_patches_flat,
            output_size,
            kernel_size=self.value_patch_size,
            stride=self.kernel_size,
            padding=self.value_patch_padding,
            use_window=self.value_patch_padding > 0,
        )
        if self.fusion_mode == "add" or self.value_source == "high_freq_residual":
            refined = coarse_composite + mixed_image
        elif self.fusion_mode == "gate":
            gate_input = torch.cat(
                [coarse_composite, mixed_image, (coarse_composite - mixed_image).abs(), mask],
                dim=1,
            )
            gate = torch.sigmoid(self.fusion_gate(gate_input))
            refined = gate * mixed_image + (1.0 - gate) * coarse_composite
        else:
            refined = mixed_image

        if self.coherence_layer is not None:
            refined = refined + self.coherence_layer(refined)

        refined = refined * mask + known_image * (1 - mask)

        return refined, masked_attention, coarse_raw
