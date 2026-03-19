from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LightweightContextBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            padding_mode="reflect",
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.pointwise(x)
        x = self.activation(x)
        return residual + x


class LightweightContextEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden_channels = max(out_channels, 32)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            LightweightContextBlock(hidden_channels, dilation=1),
            LightweightContextBlock(hidden_channels, dilation=2),
            LightweightContextBlock(hidden_channels, dilation=4),
        )
        self.proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.proj(x)


class PatchmatchHelpersMixin:
    def _apply_attention_mode(self):
        self.multihead_attention.attention_selection = (
            self.attention_selection if self.training else self.attention_eval_selection
        )
        self.multihead_attention.attention_top_k = self.attention_top_k

    def _apply_branch_dropout(self, branch: torch.Tensor, drop_prob: float) -> torch.Tensor:
        if (not self.training) or drop_prob <= 0:
            return branch
        keep_prob = 1.0 - drop_prob
        mask = branch.new_empty((branch.shape[0], 1, 1, 1)).bernoulli_(keep_prob)
        mask = mask / max(keep_prob, 1e-8)
        return branch * mask

    def _prepare_matching_branch(
        self,
        branch: torch.Tensor,
        *,
        drop_prob: float,
    ) -> torch.Tensor:
        return self._apply_branch_dropout(branch, drop_prob)

    def _build_context_encoder(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return LightweightContextEncoder(in_channels, out_channels)

    def _build_projection_head(self, input_dim: int, output_dim: int) -> nn.Sequential:
        hidden_dim = output_dim
        return nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, stride=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1, stride=1, bias=False),
        )

    def _build_matching_descriptor_head(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
    ) -> nn.Sequential:
        hidden_dim = max(input_dim, output_dim) if hidden_dim is None else int(hidden_dim)
        return nn.Sequential(
            nn.Conv2d(
                input_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.GELU(),
            LightweightContextBlock(hidden_dim, dilation=2),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1, stride=1, bias=False),
        )

    def _build_transport_head(self, input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_dim, self.transport_hidden_channels, kernel_size=1, stride=1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                self.transport_hidden_channels,
                self.transport_hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                groups=self.transport_hidden_channels,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(self.transport_hidden_channels, output_dim, kernel_size=1, stride=1),
        )

    def build_attention_supervision_entries(
        self,
        query_tokens: torch.Tensor,
        key_tokens: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> list[dict[str, torch.Tensor | tuple[int, int]]]:
        supervision_entries: list[dict[str, torch.Tensor | tuple[int, int]]] = []
        valid_key_mask = key_valid_flat > 0.5
        masked_query_mask = query_mask_flat > 0.5

        for batch_idx in range(query_tokens.shape[0]):
            query_indices = masked_query_mask[batch_idx].nonzero(as_tuple=False).flatten()
            key_indices = valid_key_mask[batch_idx].nonzero(as_tuple=False).flatten()
            entry: dict[str, torch.Tensor | tuple[int, int]] = {
                "query_indices": query_indices,
                "key_indices": key_indices,
                "token_hw": token_hw,
            }
            if query_indices.numel() == 0 or key_indices.numel() == 0:
                entry["raw_logits"] = query_tokens.new_empty((0, 0), dtype=torch.float32)
                supervision_entries.append(entry)
                continue

            raw_logits, _ = self.multihead_attention.compute_attention_logits(
                query_tokens[batch_idx : batch_idx + 1, query_indices],
                key_tokens[batch_idx : batch_idx + 1, key_indices],
            )
            entry["raw_logits"] = raw_logits.mean(dim=1).squeeze(0)
            supervision_entries.append(entry)

        return supervision_entries

    def _sample_transport_map(
        self,
        feature_map: torch.Tensor,
        coords: torch.Tensor,
        *,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        grid = coords.permute(0, 2, 3, 1).contiguous()
        return F.grid_sample(
            feature_map,
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=True,
        )

    def _sample_transport_source_values(
        self,
        source_patch_map: torch.Tensor,
        coords: torch.Tensor,
        sampled_validity: torch.Tensor,
    ) -> torch.Tensor:
        sampled_values = self._sample_transport_map(source_patch_map, coords, mode="bilinear")
        return sampled_values * sampled_validity.to(dtype=sampled_values.dtype)

    def _snap_transport_to_valid_patches(
        self,
        source_patch_map: torch.Tensor,
        coords_flat: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        token_hw: tuple[int, int],
        *,
        snap_mask_flat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_patches, _ = coords_flat.shape
        source_patch_values = self._flatten_patch_map(source_patch_map)
        snapped_values = source_patch_values.clone()
        selected_indices = torch.full(
            (batch_size, num_patches),
            -1,
            dtype=torch.long,
            device=coords_flat.device,
        )
        base_coords_flat = self._get_normalized_token_coords(
            token_hw,
            dtype=coords_flat.dtype,
            device=coords_flat.device,
        ).flatten(start_dim=2).transpose(1, 2).expand(batch_size, -1, -1)

        for batch_idx in range(batch_size):
            query_indices = (query_mask_flat[batch_idx] > 0.5).nonzero(as_tuple=False).flatten()
            if snap_mask_flat is not None:
                allowed_queries = snap_mask_flat[batch_idx, query_indices] > 0.5
                query_indices = query_indices[allowed_queries]
            valid_indices = (key_valid_flat[batch_idx] > 0.5).nonzero(as_tuple=False).flatten()
            if query_indices.numel() == 0 or valid_indices.numel() == 0:
                continue

            query_coords = coords_flat[batch_idx, query_indices].float()
            valid_coords = base_coords_flat[batch_idx, valid_indices].float()
            nearest_valid = valid_indices[torch.cdist(query_coords, valid_coords).argmin(dim=1)]
            snapped_values[batch_idx].index_copy_(0, query_indices, source_patch_values[batch_idx, nearest_valid])
            selected_indices[batch_idx].index_copy_(0, query_indices, nearest_valid)

        return snapped_values, selected_indices

    def _predict_transport_field(
        self,
        query_tokens_full: torch.Tensor,
        key_tokens_full: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor | None]:
        batch_size, _, token_dim = query_tokens_full.shape
        height, width = token_hw
        query_token_map = query_tokens_full.transpose(1, 2).contiguous().view(batch_size, token_dim, height, width)
        key_token_map = key_tokens_full.transpose(1, 2).contiguous().view(batch_size, token_dim, height, width)
        query_mask_map = query_mask_flat.view(batch_size, 1, height, width).to(dtype=query_token_map.dtype)
        valid_key_map = key_valid_flat.view(batch_size, 1, height, width).to(dtype=query_token_map.dtype)
        base_coords = self._get_normalized_token_coords(
            token_hw,
            dtype=query_token_map.dtype,
            device=query_token_map.device,
        ).expand(batch_size, -1, -1, -1)

        init_input = torch.cat([query_token_map, query_mask_map], dim=1)
        init_offsets = torch.tanh(self.transport_init_head(init_input)) * self.transport_offset_scale
        coords = torch.clamp(base_coords + init_offsets, -1.0, 1.0)

        sampled_key_tokens = self._sample_transport_map(key_token_map, coords)
        sampled_validity = self._sample_transport_map(valid_key_map, coords)
        for _ in range(self.transport_refine_steps):
            refine_input = torch.cat(
                [query_token_map, sampled_key_tokens, query_mask_map, sampled_validity, coords],
                dim=1,
            )
            delta = torch.tanh(self.transport_refine_head(refine_input)) * self.transport_refine_scale
            coords = torch.clamp(coords + delta, -1.0, 1.0)
            sampled_key_tokens = self._sample_transport_map(key_token_map, coords)
            sampled_validity = self._sample_transport_map(valid_key_map, coords)

        return {
            "base_coords": base_coords,
            "coords": coords,
            "sampled_key_tokens": sampled_key_tokens,
            "sampled_validity": sampled_validity,
        }

    def _build_transport_aux(
        self,
        source_patch_map: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        token_hw: tuple[int, int],
        transport_state: dict[str, torch.Tensor | None],
        *,
        default_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coords = transport_state["coords"]
        base_coords = transport_state["base_coords"]
        sampled_validity = transport_state["sampled_validity"]

        sampled_validity = sampled_validity.clamp(0.0, 1.0)
        hard_validity = self._sample_transport_map(
            key_valid_flat.view(coords.shape[0], 1, token_hw[0], token_hw[1]).to(dtype=coords.dtype),
            coords,
            mode="nearest",
        ).clamp(0.0, 1.0)
        sampled_values = self._sample_transport_source_values(source_patch_map, coords, sampled_validity)
        sampled_values_flat = sampled_values.flatten(start_dim=2).transpose(1, 2)
        sampled_validity_flat = sampled_validity.flatten(start_dim=2).transpose(1, 2).squeeze(-1)
        hard_validity_flat = hard_validity.flatten(start_dim=2).transpose(1, 2).squeeze(-1)
        fallback_mask = hard_validity_flat < self.transport_fallback_validity_threshold
        if default_tokens is not None:
            valid_copy = (~fallback_mask).to(dtype=sampled_values_flat.dtype).unsqueeze(-1)
            sampled_values_flat = (valid_copy * sampled_values_flat) + ((1.0 - valid_copy) * default_tokens)

        coords_flat = coords.flatten(start_dim=2).transpose(1, 2)
        selected_indices = None
        if self.training and self.transport_train_selection == "straight_through_nearest_valid":
            snapped_values_flat, selected_indices = self._snap_transport_to_valid_patches(
                source_patch_map,
                coords_flat,
                query_mask_flat,
                key_valid_flat,
                token_hw,
                snap_mask_flat=query_mask_flat,
            )
            valid_rows = selected_indices >= 0
            if default_tokens is not None:
                snapped_values_flat = torch.where(
                    valid_rows.unsqueeze(-1),
                    snapped_values_flat,
                    default_tokens,
                )
            straight_through_values = sampled_values_flat + (snapped_values_flat - sampled_values_flat).detach()
            sampled_values_flat = torch.where(
                valid_rows.unsqueeze(-1),
                straight_through_values,
                sampled_values_flat,
            )
        elif not self.training and self.transport_eval_selection == "nearest_valid":
            snapped_values_flat, selected_indices = self._snap_transport_to_valid_patches(
                source_patch_map,
                coords_flat,
                query_mask_flat,
                key_valid_flat,
                token_hw,
                snap_mask_flat=query_mask_flat,
            )
            valid_rows = selected_indices >= 0
            if default_tokens is not None:
                snapped_values_flat = torch.where(
                    valid_rows.unsqueeze(-1),
                    snapped_values_flat,
                    default_tokens,
                )
            sampled_values_flat = torch.where(
                valid_rows.unsqueeze(-1),
                snapped_values_flat,
                sampled_values_flat,
            )
        elif (not self.training) and self.transport_snap_to_valid_eval:
            # During eval, hard-invalid transport rows can produce black patches because
            # the source sampler uses zero padding and the visible source bank contains
            # masked-out pixels. Snap only those rows back onto the nearest valid patch.
            snapped_values_flat, selected_indices = self._snap_transport_to_valid_patches(
                source_patch_map,
                coords_flat,
                query_mask_flat,
                key_valid_flat,
                token_hw,
                snap_mask_flat=fallback_mask,
            )
            valid_rows = selected_indices >= 0
            if default_tokens is not None:
                snapped_values_flat = torch.where(
                    valid_rows.unsqueeze(-1),
                    snapped_values_flat,
                    default_tokens,
                )
            sampled_values_flat = torch.where(
                valid_rows.unsqueeze(-1),
                snapped_values_flat,
                sampled_values_flat,
            )

        selection_coords_flat = coords_flat
        effective_validity_flat = sampled_validity_flat
        effective_fallback_mask = fallback_mask
        if selected_indices is not None:
            base_coords_flat = base_coords.flatten(start_dim=2).transpose(1, 2)
            safe_selected = selected_indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, base_coords_flat.shape[-1])
            snapped_coords_flat = torch.gather(base_coords_flat, 1, safe_selected)
            valid_rows = selected_indices >= 0
            straight_through_coords = selection_coords_flat + (snapped_coords_flat - selection_coords_flat).detach()
            selection_coords_flat = torch.where(
                valid_rows.unsqueeze(-1),
                straight_through_coords,
                selection_coords_flat,
            )
            snapped_validity_flat = torch.ones_like(sampled_validity_flat)
            straight_through_validity = (
                effective_validity_flat + (snapped_validity_flat - effective_validity_flat).detach()
            )
            effective_validity_flat = torch.where(
                valid_rows,
                straight_through_validity,
                effective_validity_flat,
            )
            effective_fallback_mask = torch.where(
                valid_rows,
                torch.zeros_like(fallback_mask),
                fallback_mask,
            )

        aux = {
            "copy_mode": "transport",
            "query_mask_flat": query_mask_flat,
            "key_valid_flat": key_valid_flat,
            "transport_coords": coords.flatten(start_dim=2).transpose(1, 2),
            "transport_selection_coords": selection_coords_flat,
            "transport_base_coords": base_coords.flatten(start_dim=2).transpose(1, 2),
            "transport_copy_values": sampled_values.flatten(start_dim=2).transpose(1, 2),
            "transport_values": sampled_values_flat,
            "transport_validity": sampled_validity_flat,
            "transport_effective_validity": effective_validity_flat,
            "transport_hard_validity": hard_validity_flat,
            "transport_fallback_mask": fallback_mask,
            "transport_effective_fallback_mask": effective_fallback_mask,
        }
        if selected_indices is not None:
            aux["transport_selected_indices"] = selected_indices
        return sampled_values_flat, aux

    def _build_transport_attention(
        self,
        coords_flat: torch.Tensor,
        query_mask_flat: torch.Tensor,
        token_hw: tuple[int, int],
        *,
        value_dtype: torch.dtype,
        selected_indices: torch.Tensor | None = None,
        validity_flat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_patches, _ = coords_flat.shape
        height, width = token_hw
        dense_attn = torch.eye(num_patches, device=coords_flat.device, dtype=value_dtype)
        dense_attn = dense_attn.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        masked_queries = query_mask_flat > 0.5
        if selected_indices is not None:
            for batch_idx in range(batch_size):
                query_indices = masked_queries[batch_idx].nonzero(as_tuple=False).flatten()
                if query_indices.numel() == 0:
                    continue

                replacement_rows = dense_attn.new_zeros((query_indices.numel(), num_patches))
                chosen = selected_indices[batch_idx, query_indices]
                valid_rows = chosen >= 0
                if valid_rows.any():
                    replacement_rows[valid_rows, chosen[valid_rows]] = 1.0
                dense_attn[batch_idx, 0].index_copy_(0, query_indices, replacement_rows)
            return dense_attn

        width_scale = max(width - 1, 1)
        height_scale = max(height - 1, 1)

        for batch_idx in range(batch_size):
            query_indices = masked_queries[batch_idx].nonzero(as_tuple=False).flatten()
            if query_indices.numel() == 0:
                continue

            coords = coords_flat[batch_idx, query_indices]
            replacement_rows = dense_attn.new_zeros((query_indices.numel(), num_patches))
            if validity_flat is None:
                strength = replacement_rows.new_ones((query_indices.numel(), 1))
            else:
                strength = validity_flat[batch_idx, query_indices].to(dtype=replacement_rows.dtype).unsqueeze(-1)

            x_pos = (coords[..., 0] + 1.0) * 0.5 * width_scale
            y_pos = (coords[..., 1] + 1.0) * 0.5 * height_scale
            x0 = x_pos.floor().long().clamp_(0, width - 1)
            y0 = y_pos.floor().long().clamp_(0, height - 1)
            x1 = (x0 + 1).clamp_(0, width - 1)
            y1 = (y0 + 1).clamp_(0, height - 1)
            wx1 = (x_pos - x0.to(dtype=x_pos.dtype)).clamp_(0.0, 1.0)
            wy1 = (y_pos - y0.to(dtype=y_pos.dtype)).clamp_(0.0, 1.0)
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1

            candidate_indices = torch.stack(
                [
                    (y0 * width) + x0,
                    (y0 * width) + x1,
                    (y1 * width) + x0,
                    (y1 * width) + x1,
                ],
                dim=-1,
            )
            bilinear_weights = torch.stack(
                [
                    wy0 * wx0,
                    wy0 * wx1,
                    wy1 * wx0,
                    wy1 * wx1,
                ],
                dim=-1,
            )
            weighted = (bilinear_weights * strength).reshape(query_indices.numel(), -1)
            replacement_rows.scatter_add_(
                1,
                candidate_indices.reshape(query_indices.numel(), -1),
                weighted,
            )
            dense_attn[batch_idx, 0].index_copy_(0, query_indices, replacement_rows)

        return dense_attn

    def direct_patch_mix_masked_queries(
        self,
        query_tokens_full: torch.Tensor,
        key_tokens_full: torch.Tensor,
        patch_values: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        default_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_patches, _ = query_tokens_full.shape
        mixed_default = patch_values if default_tokens is None else default_tokens
        mixed_default = mixed_default.clone()
        eye = torch.eye(num_patches, device=patch_values.device, dtype=patch_values.dtype)
        dense_attn = eye.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        masked_queries = query_mask_flat > 0.5
        valid_keys = key_valid_flat > 0.5
        has_valid_keys = valid_keys.any(dim=1, keepdim=True)
        safe_valid_keys = torch.where(has_valid_keys, valid_keys, torch.ones_like(valid_keys))
        key_mask = safe_valid_keys.to(dtype=query_tokens_full.dtype).unsqueeze(1).unsqueeze(1)

        mixed_all, probs_all = self.multihead_attention(
            query_tokens_full,
            key_tokens_full,
            patch_values,
            post_softmax_mask=key_mask,
            direct_patch_mixing=True,
            query_mask_flat=query_mask_flat,
        )
        mixed_all = mixed_all.to(dtype=mixed_default.dtype)
        probs_all = probs_all.to(dtype=dense_attn.dtype)

        active_queries = masked_queries & has_valid_keys
        mixed = torch.where(active_queries.unsqueeze(-1), mixed_all, mixed_default)

        dense_attn = torch.where(active_queries.unsqueeze(1).unsqueeze(-1), probs_all, dense_attn)
        empty_key_queries = masked_queries & (~has_valid_keys)
        dense_attn = torch.where(
            empty_key_queries.unsqueeze(1).unsqueeze(-1),
            torch.zeros_like(dense_attn),
            dense_attn,
        )
        return mixed, dense_attn

    def transport_patch_mix_masked_queries(
        self,
        query_tokens_full: torch.Tensor,
        key_tokens_full: torch.Tensor,
        source_patch_map: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        *,
        token_hw: tuple[int, int] | None = None,
        default_tokens: torch.Tensor | None = None,
        return_aux_entries: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if token_hw is None:
            raise ValueError("token_hw is required for transport patch mixing.")

        batch_size = query_tokens_full.shape[0]
        transport_state = self._predict_transport_field(
            query_tokens_full,
            key_tokens_full,
            query_mask_flat,
            key_valid_flat,
            token_hw,
        )
        _, aux = self._build_transport_aux(
            source_patch_map,
            query_mask_flat,
            key_valid_flat,
            token_hw,
            transport_state,
            default_tokens=default_tokens,
        )
        sampled_values_flat = aux["transport_values"]
        sampled_validity_flat = aux["transport_validity"]
        coords_flat = aux["transport_coords"]
        selected_indices = aux.get("transport_selected_indices")

        mixed = self._flatten_patch_map(source_patch_map)
        mixed = default_tokens.clone() if default_tokens is not None else mixed.clone()
        for batch_idx in range(batch_size):
            query_indices = (query_mask_flat[batch_idx] > 0.5).nonzero(as_tuple=False).flatten()
            if query_indices.numel() == 0:
                continue
            mixed[batch_idx].index_copy_(0, query_indices, sampled_values_flat[batch_idx, query_indices])

        hard_dense_attn = None
        if self.training and self.transport_train_selection == "straight_through_nearest_valid":
            hard_dense_attn = self._build_transport_attention(
                coords_flat,
                query_mask_flat,
                token_hw,
                value_dtype=source_patch_map.dtype,
                selected_indices=selected_indices,
                validity_flat=sampled_validity_flat,
            )
            soft_dense_attn = self._build_transport_attention(
                coords_flat,
                query_mask_flat,
                token_hw,
                value_dtype=source_patch_map.dtype,
                selected_indices=None,
                validity_flat=sampled_validity_flat,
            )
            dense_attn = soft_dense_attn + (hard_dense_attn - soft_dense_attn).detach()
        else:
            dense_attn = self._build_transport_attention(
                coords_flat,
                query_mask_flat,
                token_hw,
                value_dtype=source_patch_map.dtype,
                selected_indices=selected_indices,
                validity_flat=sampled_validity_flat,
            )
        if return_aux_entries:
            return mixed, dense_attn, aux
        return mixed, dense_attn

    def build_attention_mask(
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
