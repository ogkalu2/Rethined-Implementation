from __future__ import annotations

import torch
from torch.nn import functional as F


class PatchTransportMixin:
    def _sample_transport_map(
        self,
        feature_map: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        grid = coords.permute(0, 2, 3, 1).contiguous()
        return F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def _snap_transport_to_valid_patches(
        self,
        source_patch_map: torch.Tensor,
        query_tokens_full: torch.Tensor,
        key_tokens_full: torch.Tensor,
        coords_flat: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_patches, _ = coords_flat.shape
        source_patch_values = self._flatten_patch_map(source_patch_map)
        snapped_values = source_patch_values.clone()
        query_tokens = F.normalize(query_tokens_full.float(), dim=-1, eps=1e-6)
        key_tokens = F.normalize(key_tokens_full.float(), dim=-1, eps=1e-6)
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
            valid_indices = (key_valid_flat[batch_idx] > 0.5).nonzero(as_tuple=False).flatten()
            if query_indices.numel() == 0 or valid_indices.numel() == 0:
                continue

            query_coords = coords_flat[batch_idx, query_indices].float()
            valid_coords = base_coords_flat[batch_idx, valid_indices].float()
            query_descriptors = query_tokens[batch_idx, query_indices]
            valid_descriptors = key_tokens[batch_idx, valid_indices]
            similarity = torch.matmul(query_descriptors, valid_descriptors.transpose(0, 1))
            spatial_penalty = 0.25 * (
                (query_coords.unsqueeze(1) - valid_coords.unsqueeze(0)).pow(2).mean(dim=-1)
            )
            best_valid = valid_indices[(similarity - spatial_penalty).argmax(dim=1)]
            snapped_values[batch_idx].index_copy_(0, query_indices, source_patch_values[batch_idx, best_valid])
            selected_indices[batch_idx].index_copy_(0, query_indices, best_valid)

        return snapped_values, selected_indices

    def _sample_transport_self_mask(
        self,
        key_valid_flat: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.transport_self_supervision_ratio <= 0:
            return None

        mask = torch.zeros_like(key_valid_flat)
        for batch_idx in range(key_valid_flat.shape[0]):
            valid_indices = (key_valid_flat[batch_idx] > 0.5).nonzero(as_tuple=False).flatten()
            num_valid = int(valid_indices.numel())
            if num_valid <= 1:
                continue

            num_selected = int(round(self.transport_self_supervision_ratio * num_valid))
            num_selected = max(1, min(num_selected, num_valid - 1))
            if num_selected <= 0:
                continue

            choice = torch.randperm(num_valid, device=key_valid_flat.device)[:num_selected]
            selected_indices = valid_indices[choice]
            mask[batch_idx, selected_indices] = 1.0

        if (mask > 0.5).any():
            return mask
        return None

    def _predict_transport_field(
        self,
        query_tokens_full: torch.Tensor,
        key_tokens_full: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        token_hw: tuple[int, int],
        *,
        compute_confidence: bool = False,
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

        confidence = None
        if compute_confidence and self.transport_confidence_head is not None:
            confidence_input = torch.cat(
                [query_token_map, sampled_key_tokens, query_mask_map, sampled_validity, coords],
                dim=1,
            )
            confidence = torch.sigmoid(self.transport_confidence_head(confidence_input))

        return {
            "base_coords": base_coords,
            "coords": coords,
            "sampled_key_tokens": sampled_key_tokens,
            "sampled_validity": sampled_validity,
            "confidence": confidence,
        }

    def _build_transport_aux(
        self,
        query_tokens_full: torch.Tensor,
        key_tokens_full: torch.Tensor,
        source_patch_map: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        token_hw: tuple[int, int],
        transport_state: dict[str, torch.Tensor | None],
        *,
        return_diagnostics: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coords = transport_state["coords"]
        base_coords = transport_state["base_coords"]
        sampled_validity = transport_state["sampled_validity"]
        confidence = transport_state["confidence"]
        coords_flat = coords.flatten(start_dim=2).transpose(1, 2)
        sampled_values_flat, selected_indices = self._snap_transport_to_valid_patches(
            source_patch_map,
            query_tokens_full,
            key_tokens_full,
            coords_flat,
            query_mask_flat,
            key_valid_flat,
            token_hw,
        )
        aux = {
            "copy_mode": self.copy_mode,
            "query_mask_flat": query_mask_flat,
            "key_valid_flat": key_valid_flat,
            "transport_coords": coords_flat,
            "transport_base_coords": base_coords.flatten(start_dim=2).transpose(1, 2),
            "transport_values": sampled_values_flat,
            "transport_selected_indices": selected_indices,
            "transport_validity": sampled_validity.flatten(start_dim=2).transpose(1, 2).squeeze(-1),
        }
        if confidence is not None:
            aux["transport_confidence"] = confidence.flatten(start_dim=2).transpose(1, 2).squeeze(-1)
        if return_diagnostics:
            backward_state = self._predict_transport_field(
                key_tokens_full,
                query_tokens_full,
                key_valid_flat,
                torch.ones_like(query_mask_flat),
                token_hw,
                compute_confidence=False,
            )
            cycle_back_coords = self._sample_transport_map(backward_state["coords"], coords)
            aux["transport_cycle_back_coords"] = cycle_back_coords.flatten(start_dim=2).transpose(1, 2)
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

    def transport_patch_mix(
        self,
        query_tokens_full: torch.Tensor,
        key_tokens_full: torch.Tensor,
        source_patch_map: torch.Tensor,
        query_mask_flat: torch.Tensor,
        key_valid_flat: torch.Tensor,
        token_hw: tuple[int, int],
        default_tokens: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = query_tokens_full.shape[0]
        transport_state = self._predict_transport_field(
            query_tokens_full,
            key_tokens_full,
            query_mask_flat,
            key_valid_flat,
            token_hw,
            compute_confidence=return_diagnostics,
        )
        sampled_values, aux = self._build_transport_aux(
            query_tokens_full,
            key_tokens_full,
            source_patch_map,
            query_mask_flat,
            key_valid_flat,
            token_hw,
            transport_state,
            return_diagnostics=return_diagnostics,
        )
        sampled_values_flat = aux["transport_values"]
        sampled_validity_flat = aux["transport_validity"]
        coords_flat = aux["transport_coords"]
        selected_indices = aux["transport_selected_indices"]

        mixed = self._flatten_patch_map(source_patch_map)
        if default_tokens is not None:
            mixed = default_tokens.clone()
        else:
            mixed = mixed.clone()
        for batch_idx in range(batch_size):
            query_indices = (query_mask_flat[batch_idx] > 0.5).nonzero(as_tuple=False).flatten()
            if query_indices.numel() == 0:
                continue
            mixed[batch_idx].index_copy_(0, query_indices, sampled_values_flat[batch_idx, query_indices])

        dense_attn = self._build_transport_attention(
            coords_flat,
            query_mask_flat,
            token_hw,
            value_dtype=source_patch_map.dtype,
            selected_indices=selected_indices,
            validity_flat=sampled_validity_flat,
        )
        return mixed, dense_attn, aux
