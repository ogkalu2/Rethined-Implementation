from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from blocks import NativeGaussianBlur2d

from .attention import MultiHeadAttention
from .helpers import PatchmatchHelpersMixin
from .patch_ops import PatchOpsMixin


class PatchInpainting(PatchmatchHelpersMixin, PatchOpsMixin, nn.Module):
    def __init__(
        self,
        *,
        kernel_size: int,
        value_patch_size: int | None = None,
        attention_temperature: float = 1.0,
        attention_top_k: int | None = None,
        attention_selection: str = "softmax",
        attention_eval_selection: str | None = None,
        attention_gumbel_tau: float = 1.0,
        attention_gumbel_hard: bool = True,
        use_transport: bool = False,
        transport_hidden_channels: int = 64,
        transport_refine_steps: int = 2,
        transport_offset_scale: float = 1.0,
        transport_refine_scale: float = 0.25,
        transport_self_supervision_ratio: float = 0.0,
        transport_fallback_validity_threshold: float = 0.1,
        transport_snap_to_valid_eval: bool = True,
        matching_descriptor_dim: int | None = None,
        matching_hidden_dim: int | None = None,
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
        self.matching_hidden_dim = None if matching_hidden_dim is None else int(matching_hidden_dim)
        if self.matching_hidden_dim is not None and self.matching_hidden_dim <= 0:
            self.matching_hidden_dim = None
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
        self.attention_selection = str(attention_selection).lower()
        self.attention_eval_selection = (
            self.attention_selection
            if attention_eval_selection is None
            else str(attention_eval_selection).lower()
        )
        self.use_transport = bool(use_transport)
        self.transport_hidden_channels = int(transport_hidden_channels)
        self.transport_refine_steps = max(0, int(transport_refine_steps))
        self.transport_offset_scale = float(transport_offset_scale)
        self.transport_refine_scale = float(transport_refine_scale)
        self.transport_self_supervision_ratio = float(transport_self_supervision_ratio)
        self.transport_fallback_validity_threshold = float(transport_fallback_validity_threshold)
        self.transport_snap_to_valid_eval = bool(transport_snap_to_valid_eval)
        self.attention_top_k = None if attention_top_k is None else int(attention_top_k)
        if self.attention_top_k is not None and self.attention_top_k <= 0:
            self.attention_top_k = None
        if self.transport_hidden_channels <= 0:
            raise ValueError("transport_hidden_channels must be positive.")
        if self.transport_offset_scale < 0:
            raise ValueError("transport_offset_scale must be non-negative.")
        if self.transport_refine_scale < 0:
            raise ValueError("transport_refine_scale must be non-negative.")
        if not 0.0 <= self.transport_self_supervision_ratio <= 1.0:
            raise ValueError("transport_self_supervision_ratio must be in [0, 1].")
        if not 0.0 <= self.transport_fallback_validity_threshold <= 1.0:
            raise ValueError("transport_fallback_validity_threshold must be in [0, 1].")

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
        self.transport_init_head = None
        self.transport_refine_head = None
        self.transport_confidence_head = None
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
        if self.use_transport:
            transport_init_in = self.patch_token_dim + 1
            transport_refine_in = (self.patch_token_dim * 2) + 4
            self.transport_init_head = self._build_transport_head(transport_init_in, 2)
            self.transport_refine_head = self._build_transport_head(transport_refine_in, 2)
            self.transport_confidence_head = self._build_transport_head(transport_refine_in, 1)
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
            attention_top_k=self.attention_top_k,
            attention_selection=self.attention_selection,
            attention_gumbel_tau=float(attention_gumbel_tau),
            attention_gumbel_hard=attention_gumbel_hard,
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
        self._apply_attention_mode()

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_attention_mode()
        return self

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
        # `coarse_raw` is only supervised inside the hole. For matching, use the coarse
        # prediction inside masked regions and keep known pixels as the source of truth
        # elsewhere so visible-key descriptors are not polluted by arbitrary coarse output.
        coarse_token_source = coarse_composite

        patch_map, output_size = self.unfold_native(coarse_token_source, self.kernel_size)
        source_patch_map, _ = self.extract_patches(
            known_image,
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
        use_coarse_rgb_matching = self.match_coarse_rgb
        coarse_match_map = None
        if use_coarse_rgb_matching:
            coarse_match_map = patch_map.detach() if self.detach_coarse_rgb else patch_map
            coarse_match_map = self._prepare_matching_branch(
                coarse_match_map,
                drop_prob=self.coarse_rgb_branch_dropout,
            )
        coarse_features = None
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
            if use_coarse_rgb_matching:
                query_token_inputs.append(coarse_match_map)
            if self.concat_features:
                query_token_inputs.append(coarse_features)

            key_token_inputs = [key_context_map, visible_patch_map]
            if use_coarse_rgb_matching:
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
            if use_coarse_rgb_matching:
                token_inputs.append(coarse_match_map)
            if self.concat_features:
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
        attention_supervision_entries = None
        copy_aux = None
        if return_aux and not self.use_transport:
            attention_supervision_entries = self.build_attention_supervision_entries(
                query_tokens,
                key_tokens,
                query_mask_flat,
                key_valid_flat,
                token_hw,
            )

        patch_values = source_patch_map.flatten(start_dim=2).transpose(1, 2)
        if self.use_transport:
            transport_result = self.transport_patch_mix_masked_queries(
                query_tokens,
                key_tokens,
                source_patch_map,
                query_mask_flat,
                key_valid_flat,
                token_hw=token_hw,
                default_tokens=None,
                return_aux_entries=return_aux,
            )
            if return_aux:
                mixed_patches_flat, masked_attention, copy_aux = transport_result
            else:
                mixed_patches_flat, masked_attention = transport_result
        elif self.attention_masking:
            mixed_patches_flat, masked_attention = self.direct_patch_mix_masked_queries(
                query_tokens,
                key_tokens,
                patch_values,
                query_mask_flat,
                key_valid_flat,
            )
        else:
            mixed_patches_flat, masked_attention = self.multihead_attention(
                query_tokens,
                key_tokens,
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
                "copy_mode": "transport" if self.use_transport else "attention",
                "attention_supervision_entries": attention_supervision_entries,
                "query_matching_tokens": query_matching_tokens,
                "key_matching_tokens": key_matching_tokens,
            }
            if copy_aux is not None:
                aux.update(copy_aux)
            if self.use_transport and self.training:
                transport_self_mask = self._sample_transport_self_mask(key_valid_flat)
                if transport_self_mask is not None:
                    transport_self_keys = key_valid_flat * (1.0 - transport_self_mask)
                    if (transport_self_keys > 0.5).any():
                        self_transport_state = self._predict_transport_field(
                            query_tokens,
                            key_tokens,
                            transport_self_mask,
                            transport_self_keys,
                            token_hw,
                            compute_confidence=False,
                        )
                        _, transport_self_aux = self._build_transport_aux(
                            query_tokens,
                            key_tokens,
                            source_patch_map,
                            transport_self_mask,
                            transport_self_keys,
                            token_hw,
                            self_transport_state,
                            default_tokens=None,
                            return_diagnostics=False,
                        )
                        aux["transport_self_aux"] = transport_self_aux
            return refined, masked_attention, coarse_raw, aux
        return refined, masked_attention, coarse_raw
