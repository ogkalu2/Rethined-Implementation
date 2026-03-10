"""Patch-match style refinement modules."""

import torch
from torch import nn
from torch.nn import functional as F

from blocks import NativeGaussianBlur2d
from hr import HRResidualRefiner


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
        d_v, n_head = self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = v

        q = self.w_qs(q).view(sz_b, len_q, n_head, self.d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, self.d_k)
        if direct_patch_mixing:
            if n_head != 1:
                raise ValueError("direct_patch_mixing only supports n_head=1.")
            v_mixed = v.unsqueeze(1)
        else:
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
            v_mixed = v.transpose(1, 2)

        q, k = q.transpose(1, 2), k.transpose(1, 2)

        attn = torch.matmul(q / self.d_k ** 0.5, k.transpose(2, 3))

        if qk_mask is not None:
            attn = attn + qk_mask

        if k_mask is not None:
            attn = attn + k_mask.squeeze(-1).unsqueeze(-2)

        if self.topk_patches is not None and self.topk_patches < attn.size(-1):
            topk_vals, topk_idx = torch.topk(attn, k=self.topk_patches, dim=-1)
            sparse_attn = torch.full_like(attn, -1e4)
            attn = sparse_attn.scatter(-1, topk_idx, topk_vals)

        attn = attn / max(self.softmax_temperature, 1e-6)
        attn = F.softmax(attn.float(), dim=-1).to(v.dtype)
        if post_softmax_mask is not None:
            attn = attn * post_softmax_mask
            if renorm_post_mask:
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.use_argmax:
            idx = torch.argmax(attn, dim=-1, keepdim=True)
            attn_hard = torch.zeros_like(attn).scatter_(-1, idx, 1.0)
            attn = attn_hard - attn.detach() + attn

        attn = self.dropout(attn)
        head_output = torch.matmul(attn, v_mixed)
        output = head_output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        if not direct_patch_mixing:
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
        merge_mode: str = "sum",
        use_kpos: bool = True,
        image_size: int = 512,
        embed_dim: int = 512,
        use_qpos: bool = True,
        dropout: float = 0.1,
        attention_type: str = "MultiHeadAttention",
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
        positional_grid_size: int = 32,
        model,
    ):
        super().__init__()
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
        self.token_grid_size = int(image_size / stem_out_stride / self.kernel_size)
        self.positional_grid_size = max(1, min(int(positional_grid_size), self.token_grid_size))
        self.register_buffer(
            "qk_mask",
            1e4 * torch.eye(self.token_grid_size ** 2).unsqueeze(0).unsqueeze(0),
        )
        if not mask_query_with_segmentation_mask:
            self.mask_query = torch.nn.Parameter(
                torch.zeros(1, self.token_grid_size ** 2, 1, 1).float()
            )

        self.encoder_decoder = model
        self.image_size = image_size
        self.positionalencoding = (
            torch.nn.Parameter(
                torch.zeros(
                    1,
                    self.patch_token_dim,
                    self.positional_grid_size,
                    self.positional_grid_size,
                )
            )
            if use_kpos or use_qpos
            else None
        )
        self.refinement_gate = nn.Parameter(torch.tensor([1.0]))
        self.refinement_runtime_scale = 1.0
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
        self.hr_transfer_gate = nn.Sequential(
            nn.LayerNorm(self.hr_rescore_dim + 3),
            nn.Linear(self.hr_rescore_dim + 3, self.hr_rescore_hidden_dim),
            nn.GELU(),
            nn.Linear(self.hr_rescore_hidden_dim, 1),
        )
        nn.init.zeros_(self.hr_transfer_gate[-1].weight)
        nn.init.constant_(self.hr_transfer_gate[-1].bias, -3.0)
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
        if merge_mode == "all":
            self.merge_func = self.merge_all_patches_sum

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
        weights = torch.eye(kernel_size * kernel_size, dtype=torch.float)
        weights = weights.reshape((kernel_size * kernel_size, 1, kernel_size, kernel_size))
        weights = weights.repeat(channels, 1, 1, 1)
        return weights

    def unfold_native(self, feature_map: torch.Tensor, kernel_size: int):
        batch_size, channels, height, width = feature_map.shape
        patches = F.unfold(feature_map, kernel_size=kernel_size, stride=kernel_size)
        n_h, n_w = height // kernel_size, width // kernel_size
        patches = patches.view(batch_size, channels * kernel_size * kernel_size, n_h, n_w)
        return patches, (height, width)

    def fold_native(self, patches: torch.Tensor, output_size, kernel_size: int, use_final_conv: bool) -> torch.Tensor:
        n_h = output_size[0] // kernel_size
        n_w = output_size[1] // kernel_size
        batch_size = patches.shape[0]
        channels = patches.shape[2] // (kernel_size * kernel_size)
        patches = patches.view(batch_size, n_h, n_w, channels, kernel_size, kernel_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(batch_size, channels, n_h * kernel_size, n_w * kernel_size)

    def build_paper_attention_mask(self, pooled_patch_mask: torch.Tensor) -> torch.Tensor:
        patch_mask_flat = pooled_patch_mask.squeeze(1).squeeze(-1)
        _, num_patches = patch_mask_flat.shape
        is_masked_q = patch_mask_flat.unsqueeze(-1)
        is_valid_k = (1.0 - patch_mask_flat).unsqueeze(1)
        eye = torch.eye(num_patches, device=patch_mask_flat.device, dtype=patch_mask_flat.dtype).unsqueeze(0)
        allowed = (1.0 - is_masked_q) * eye + is_masked_q * is_valid_k
        return ((1.0 - allowed) * -1e4).unsqueeze(1)

    def apply_paper_coherence(self, patches: torch.Tensor, output_size) -> torch.Tensor:
        if self.paper_coherence_layer is None:
            return patches

        n_h = output_size[0] // self.kernel_size
        n_w = output_size[1] // self.kernel_size
        batch_size = patches.shape[0]
        patches_2d = patches.transpose(1, 2).view(batch_size, -1, n_h, n_w)
        patches_2d = patches_2d + self.paper_coherence_layer(patches_2d)
        return patches_2d.view(batch_size, -1, n_h * n_w).transpose(1, 2)

    def set_refinement_runtime_scale(self, scale: float) -> None:
        self.refinement_runtime_scale = float(scale)

    def reparameterize(self):
        if self.hr_residual_refiner is not None and hasattr(self.hr_residual_refiner, "reparameterize"):
            self.hr_residual_refiner.reparameterize()
        return self

    def get_positional_encoding(self) -> torch.Tensor | None:
        """Upsample the learnable positional grid to the current token grid."""
        if self.positionalencoding is None:
            return None
        if self.positionalencoding.shape[-2:] == (self.token_grid_size, self.token_grid_size):
            pe = self.positionalencoding
        else:
            pe = F.interpolate(
                self.positionalencoding,
                size=(self.token_grid_size, self.token_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        return pe.flatten(start_dim=2).transpose(1, 2)

    def forward(self, image, mask):
        masked_input = image
        image_coarse_inpainting, features = self.encoder_decoder(masked_input)
        if self.mask_inpainting:
            coarse_composite = image_coarse_inpainting * mask + masked_input * (1 - mask)
        else:
            coarse_composite = image_coarse_inpainting
        image_to_return = image_coarse_inpainting

        composite_blurred = self.final_gaussian_blur(coarse_composite)
        composite_patches_full, sizes = self.unfold_native(coarse_composite, self.kernel_size)
        blurred_patches_full, _ = self.unfold_native(composite_blurred, self.kernel_size)
        hf_patches = composite_patches_full - blurred_patches_full

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

        # Use full coarse-composite patches for affinity estimation so attention
        # can follow semantics/structure, but keep LR values in the HF domain to
        # preserve the coarse branch as the low-frequency base.
        match_patches = composite_patches_full
        preserve_patches_full = composite_patches_full
        if self.concat_features:
            features_to_concat = features[self.feature_i]
            features_to_concat = F.interpolate(
                features_to_concat,
                size=match_patches.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            token_map = torch.cat([match_patches, features_to_concat], dim=1)
        else:
            token_map = match_patches

        if self.token_use_mask_ratio:
            token_map = torch.cat([token_map, mask_ratio], dim=1)

        input_attn = token_map.flatten(start_dim=2).transpose(1, 2)

        positional_encoding = self.get_positional_encoding()
        if positional_encoding is not None:
            input_attn = input_attn + positional_encoding

        match_patches_flat = match_patches.flatten(start_dim=2).transpose(1, 2)
        preserve_patches_flat = preserve_patches_full.flatten(start_dim=2).transpose(1, 2)
        blurred_patches_flat = blurred_patches_full.flatten(start_dim=2).transpose(1, 2)
        hf_patches_flat = hf_patches.flatten(start_dim=2).transpose(1, 2)
        base_hf_flat = preserve_patches_flat - blurred_patches_flat
        self.last_base_patches_flat = preserve_patches_flat

        pre_softmax_mask = (
            self.build_paper_attention_mask(mask_same_res_as_features_pooled) if self.attention_masking else None
        )
        out, atten_weights = self.multihead_attention(
            input_attn,
            input_attn,
            hf_patches_flat,
            None,
            None,
            qk_mask=pre_softmax_mask,
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
        return torch.einsum("bkhq,bchk->bchq", patch_scores, sequence_of_patches.unsqueeze(2)).squeeze(2)
