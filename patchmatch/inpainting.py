from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from blocks import NativeGaussianBlur2d

from .attention import MultiHeadAttention
from .helpers import PatchmatchHelpersMixin
from .patch_ops import PatchOpsMixin


class TopKReranker(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, query_chunk_size: int = 256):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.query_chunk_size = max(0, int(query_chunk_size))
        self.query_proj = nn.Linear(int(token_dim), hidden_dim, bias=False)
        self.key_proj = nn.Linear(int(token_dim), hidden_dim, bias=False)
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _forward_chunk(
        self,
        query_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        stage1_logits: torch.Tensor,
        relative_coords: torch.Tensor,
    ) -> torch.Tensor:
        query_features = self.query_proj(query_tokens).unsqueeze(1)
        candidate_features = self.key_proj(candidate_tokens)
        features = torch.cat(
            [
                query_features * candidate_features,
                (query_features - candidate_features).abs(),
                stage1_logits.unsqueeze(-1),
                relative_coords,
            ],
            dim=-1,
        )
        return self.net(features).squeeze(-1)

    def forward(
        self,
        query_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        stage1_logits: torch.Tensor,
        relative_coords: torch.Tensor,
    ) -> torch.Tensor:
        if self.query_chunk_size <= 0 or query_tokens.shape[0] <= self.query_chunk_size:
            return self._forward_chunk(query_tokens, candidate_tokens, stage1_logits, relative_coords)

        outputs = []
        for start in range(0, query_tokens.shape[0], self.query_chunk_size):
            end = start + self.query_chunk_size
            outputs.append(
                self._forward_chunk(
                    query_tokens[start:end],
                    candidate_tokens[start:end],
                    stage1_logits[start:end],
                    relative_coords[start:end],
                )
            )
        return torch.cat(outputs, dim=0)


class PatchInpainting(PatchmatchHelpersMixin, PatchOpsMixin, nn.Module):
    def __init__(
        self,
        *,
        kernel_size: int,
        value_patch_size: int | None = None,
        attention_temperature: float = 1.0,
        attention_top_k: int | None = None,
        attention_selection: str = "softmax",
        attention_gumbel_tau: float = 1.0,
        attention_gumbel_hard: bool = True,
        attention_warmup_selection: str | None = None,
        attention_warmup_steps: int = 0,
        attention_warmup_top_k: int | None = None,
        attention_gumbel_hard_start_step: int = 0,
        matching_descriptor_dim: int | None = None,
        matching_hidden_dim: int | None = None,
        reranker_hidden_dim: int = 0,
        reranker_top_k: int | None = None,
        reranker_query_chunk_size: int = 256,
        match_coarse_rgb: bool = True,
        detach_coarse_rgb: bool = False,
        coarse_rgb_branch_dropout: float = 0.0,
        query_image_context_matching: bool = False,
        separate_query_key_matching: bool = False,
        shared_query_key_descriptor: bool = False,
        query_context_channels: int = 32,
        key_context_channels: int = 32,
        query_context_residual_init: float = 0.0,
        key_coarse_rgb_residual_init: float = 0.0,
        key_feature_residual_init: float = 0.0,
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
        supervision_band_radius: int = 1,
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
        self.matching_hidden_dim = None if matching_hidden_dim is None else int(matching_hidden_dim)
        if self.matching_hidden_dim is not None and self.matching_hidden_dim <= 0:
            self.matching_hidden_dim = None
        self.reranker_hidden_dim = max(0, int(reranker_hidden_dim))
        self.reranker_top_k = None if reranker_top_k is None else int(reranker_top_k)
        if self.reranker_top_k is not None and self.reranker_top_k <= 0:
            self.reranker_top_k = None
        self.reranker_query_chunk_size = max(0, int(reranker_query_chunk_size))
        self.match_coarse_rgb = bool(match_coarse_rgb)
        self.detach_coarse_rgb = bool(detach_coarse_rgb)
        self.coarse_rgb_branch_dropout = float(coarse_rgb_branch_dropout)
        self.query_image_context_matching = bool(query_image_context_matching)
        self.separate_query_key_matching = bool(separate_query_key_matching)
        self.shared_query_key_descriptor = bool(shared_query_key_descriptor)
        self.query_context_channels = int(query_context_channels)
        self.key_context_channels = int(key_context_channels)
        self.query_context_residual_init = float(query_context_residual_init)
        self.key_coarse_rgb_residual_init = float(key_coarse_rgb_residual_init)
        self.key_feature_residual_init = float(key_feature_residual_init)
        if not 0.0 <= self.coarse_rgb_branch_dropout < 1.0:
            raise ValueError("coarse_rgb_branch_dropout must be in [0, 1).")
        if self.query_image_context_matching and self.separate_query_key_matching:
            raise ValueError(
                "query_image_context_matching and separate_query_key_matching cannot both be enabled."
            )
        if self.separate_query_key_matching:
            if self.query_context_channels <= 0:
                raise ValueError("query_context_channels must be positive when separate_query_key_matching=True.")
            if self.key_context_channels <= 0:
                raise ValueError("key_context_channels must be positive when separate_query_key_matching=True.")
            if self.shared_query_key_descriptor and self.query_context_channels != self.key_context_channels:
                raise ValueError(
                    "query_context_channels and key_context_channels must match when shared_query_key_descriptor=True."
                )
        elif self.query_image_context_matching and self.query_context_channels <= 0:
            raise ValueError(
                "query_context_channels must be positive when query_image_context_matching=True."
            )
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
        if not self.separate_query_key_matching and self.matching_input_dim == 0:
            raise ValueError("Matching must use coarse RGB patches, coarse features, or both.")
        self.query_matching_input_dim = self.matching_input_dim
        self.key_matching_input_dim = self.matching_input_dim
        if self.separate_query_key_matching:
            self.query_matching_input_dim = self.query_patch_dim + self.query_context_channels
            if self.match_coarse_rgb:
                self.query_matching_input_dim += self.query_patch_dim
            if self.concat_features:
                self.query_matching_input_dim += self.feature_dim
            self.key_matching_input_dim = self.query_patch_dim + self.key_context_channels
            if self.match_coarse_rgb:
                self.key_matching_input_dim += self.query_patch_dim
            if self.concat_features:
                self.key_matching_input_dim += self.feature_dim
        self.patch_token_dim = (
            max(self.query_matching_input_dim, self.key_matching_input_dim)
            if self.matching_descriptor_dim is None
            else self.matching_descriptor_dim
        )
        self.positional_grid_size = max(1, min(int(positional_grid_size), self.token_grid_size))
        self.base_attention_selection = str(attention_selection).lower()
        self.attention_warmup_selection = (
            None if attention_warmup_selection is None else str(attention_warmup_selection).lower()
        )
        self.base_attention_top_k = None if attention_top_k is None else int(attention_top_k)
        if self.base_attention_top_k is not None and self.base_attention_top_k <= 0:
            self.base_attention_top_k = None
        self.attention_warmup_top_k = None if attention_warmup_top_k is None else int(attention_warmup_top_k)
        if self.attention_warmup_top_k is not None and self.attention_warmup_top_k <= 0:
            self.attention_warmup_top_k = None
        if self.attention_warmup_selection not in {None, "softmax", "gumbel", "argmax"}:
            raise ValueError(
                "attention_warmup_selection must be one of {None, 'softmax', 'gumbel', 'argmax'}."
            )
        self.attention_warmup_steps = max(0, int(attention_warmup_steps))
        self.attention_gumbel_hard_start_step = max(0, int(attention_gumbel_hard_start_step))
        self.supervision_band_radius = max(0, int(supervision_band_radius))
        self.current_training_step = 0

        self.encoder_decoder = model
        self.final_gaussian_blur = NativeGaussianBlur2d((7, 7), sigma=(2.01, 2.01))
        self.query_context_encoder = None
        self.key_context_encoder = None
        self.query_context_descriptor_head = None
        self.query_context_scale = None
        self.key_coarse_rgb_scale = None
        self.key_feature_scale = None
        self.shared_query_key_descriptor_head = None
        self.query_descriptor_head = None
        self.key_descriptor_head = None
        self.matching_descriptor_head = None
        self.patch_reranker = None
        if self.separate_query_key_matching:
            self.query_context_encoder = self._build_context_encoder(4, self.query_context_channels)
            self.key_context_encoder = self._build_context_encoder(3, self.key_context_channels)
            if self.match_coarse_rgb:
                self.key_coarse_rgb_scale = nn.Parameter(torch.tensor(self.key_coarse_rgb_residual_init))
            if self.concat_features:
                self.key_feature_scale = nn.Parameter(torch.tensor(self.key_feature_residual_init))
            if self.shared_query_key_descriptor:
                self.shared_query_key_descriptor_head = self._build_matching_descriptor_head(
                    self.query_matching_input_dim,
                    self.patch_token_dim,
                    hidden_dim=self.matching_hidden_dim,
                )
            else:
                self.query_descriptor_head = self._build_matching_descriptor_head(
                    self.query_matching_input_dim,
                    self.patch_token_dim,
                    hidden_dim=self.matching_hidden_dim,
                )
                self.key_descriptor_head = self._build_matching_descriptor_head(
                    self.key_matching_input_dim,
                    self.patch_token_dim,
                    hidden_dim=self.matching_hidden_dim,
                )
        elif self.matching_descriptor_dim is not None:
            self.matching_descriptor_head = self._build_matching_descriptor_head(
                self.matching_input_dim,
                self.matching_descriptor_dim,
                hidden_dim=self.matching_hidden_dim,
            )
        if self.query_image_context_matching:
            self.query_context_encoder = self._build_context_encoder(4, self.query_context_channels)
            self.query_context_descriptor_head = self._build_projection_head(
                self.query_context_channels + self.query_patch_dim,
                self.patch_token_dim,
            )
            self.query_context_scale = nn.Parameter(torch.tensor(self.query_context_residual_init))
        self.multihead_attention = MultiHeadAttention(
            embed_dim=self.patch_token_dim,
            d_v=self.value_patch_dim,
            n_head=self.nheads,
            dropout=float(dropout),
            d_qk=int(embed_dim),
            attention_temperature=float(attention_temperature),
            attention_top_k=self.base_attention_top_k,
            attention_selection=attention_selection,
            attention_gumbel_tau=float(attention_gumbel_tau),
            attention_gumbel_hard=attention_gumbel_hard,
        )
        if self.reranker_hidden_dim > 0:
            self.patch_reranker = TopKReranker(
                self.patch_token_dim,
                self.reranker_hidden_dim,
                query_chunk_size=self.reranker_query_chunk_size,
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
        self._apply_attention_schedule()

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_attention_schedule()
        return self

    def set_training_step(self, step: int):
        self.current_training_step = max(0, int(step))
        self._apply_attention_schedule()

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        value_image: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        coarse_raw, features = self.encoder_decoder(image)
        known_image = image if value_image is None else value_image
        coarse_composite = coarse_raw * mask + known_image * (1 - mask)

        patch_map, output_size = self.unfold_native(coarse_raw, self.kernel_size)
        default_patch_values = None
        if self.value_source == "high_freq_residual":
            value_base = known_image - self.final_gaussian_blur(known_image)
            visible_value_base = image - self.final_gaussian_blur(image)
        else:
            value_base = known_image
            visible_value_base = image

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

        query_matching_tokens = None
        key_matching_tokens = None
        if self.separate_query_key_matching:
            visible_patch_map, _ = self.unfold_native(image, self.kernel_size)
            query_context_map = self._pool_to_token_grid(
                self.query_context_encoder(torch.cat([image, mask], dim=1)),
                patch_map.shape[-2:],
            )
            key_context_map = self._pool_to_token_grid(
                self.key_context_encoder(image),
                patch_map.shape[-2:],
            )

            query_token_inputs = [query_context_map, visible_patch_map]
            if self.match_coarse_rgb:
                coarse_match_map = patch_map.detach() if self.detach_coarse_rgb else patch_map
                coarse_match_map = self._prepare_matching_branch(
                    coarse_match_map,
                    drop_prob=self.coarse_rgb_branch_dropout,
                )
                query_token_inputs.append(coarse_match_map)
            if self.concat_features:
                coarse_features = F.interpolate(
                    features[self.feature_i],
                    size=patch_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                coarse_features = self._prepare_matching_branch(
                    coarse_features,
                    drop_prob=0.0,
                )
                query_token_inputs.append(coarse_features)

            key_token_inputs = [key_context_map, visible_patch_map]
            if self.match_coarse_rgb:
                key_token_inputs.append(self.key_coarse_rgb_scale * coarse_match_map)
            if self.concat_features:
                key_token_inputs.append(self.key_feature_scale * coarse_features)
            query_token_map = torch.cat(query_token_inputs, dim=1)
            key_token_map = torch.cat(key_token_inputs, dim=1)
            if self.shared_query_key_descriptor:
                query_token_map = self.shared_query_key_descriptor_head(query_token_map)
                key_token_map = self.shared_query_key_descriptor_head(key_token_map)
            else:
                query_token_map = self.query_descriptor_head(query_token_map)
                key_token_map = self.key_descriptor_head(key_token_map)
            query_matching_tokens = query_token_map.flatten(start_dim=2).transpose(1, 2)
            key_matching_tokens = key_token_map.flatten(start_dim=2).transpose(1, 2)
            query_tokens = query_matching_tokens
            key_tokens = key_matching_tokens
        else:
            token_inputs = []
            if self.match_coarse_rgb:
                coarse_match_map = patch_map.detach() if self.detach_coarse_rgb else patch_map
                coarse_match_map = self._prepare_matching_branch(
                    coarse_match_map,
                    drop_prob=self.coarse_rgb_branch_dropout,
                )
                token_inputs.append(coarse_match_map)
            if self.concat_features:
                coarse_features = F.interpolate(
                    features[self.feature_i],
                    size=patch_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                coarse_features = self._prepare_matching_branch(
                    coarse_features,
                    drop_prob=0.0,
                )
                token_inputs.append(coarse_features)

            token_map = token_inputs[0] if len(token_inputs) == 1 else torch.cat(token_inputs, dim=1)
            if self.matching_descriptor_head is not None:
                token_map = self.matching_descriptor_head(token_map)

            query_matching_tokens = token_map.flatten(start_dim=2).transpose(1, 2)
            key_matching_tokens = query_matching_tokens
            if self.query_image_context_matching:
                visible_patch_map, _ = self.unfold_native(image, self.kernel_size)
                query_context_map = self._pool_to_token_grid(
                    self.query_context_encoder(torch.cat([image, mask], dim=1)),
                    patch_map.shape[-2:],
                )
                query_context_map = torch.cat([query_context_map, visible_patch_map], dim=1)
                query_context_map = self.query_context_descriptor_head(query_context_map)
                query_context_tokens = query_context_map.flatten(start_dim=2).transpose(1, 2)
                query_matching_tokens = query_matching_tokens + self.query_context_scale * query_context_tokens

            query_tokens = query_matching_tokens
            key_tokens = key_matching_tokens

        positional_encoding = self.get_positional_encoding()
        if positional_encoding is not None:
            query_tokens = query_tokens + positional_encoding
            key_tokens = key_tokens + positional_encoding
        query_tokens = self.pre_attention_norm(query_tokens)
        key_tokens = self.pre_attention_norm(key_tokens)

        token_hw = patch_map.shape[-2:]
        supervision_band_mask_flat = None
        attention_supervision_entries = None
        if return_aux:
            supervision_band_mask_flat, attention_supervision_entries = self.build_attention_supervision_entries(
                query_tokens,
                key_tokens,
                query_mask_flat,
                key_valid_flat,
                token_hw,
            )

        patch_values = source_patch_map.flatten(start_dim=2).transpose(1, 2)
        if self.value_source == "high_freq_residual":
            default_patch_values = torch.zeros_like(patch_values)
        if self.attention_masking:
            mixed_patches_flat, masked_attention, copy_aux = self.direct_patch_mix_masked_queries(
                query_tokens,
                key_tokens,
                patch_values,
                query_mask_flat,
                key_valid_flat,
                default_tokens=default_patch_values,
                token_hw=token_hw,
            )
        else:
            mixed_patches_flat, masked_attention = self.multihead_attention(
                query_tokens,
                key_tokens,
                patch_values,
                direct_patch_mixing=True,
                query_mask_flat=query_mask_flat,
            )
            copy_aux = None

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

        if return_aux:
            aux = {
                "query_mask_flat": query_mask_flat,
                "key_valid_flat": key_valid_flat,
                "kernel_size": self.kernel_size,
                "value_patch_size": self.value_patch_size,
                "value_patch_padding": self.value_patch_padding,
                "token_hw": token_hw,
                "supervision_band_mask_flat": supervision_band_mask_flat,
                "attention_supervision_entries": attention_supervision_entries,
                "matching_tokens": query_matching_tokens,
                "query_matching_tokens": query_matching_tokens,
                "key_matching_tokens": key_matching_tokens,
                "copy_aux": copy_aux,
            }
            return refined, masked_attention, coarse_raw, aux
        return refined, masked_attention, coarse_raw
