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

    def _dilate_binary_mask(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        radius = int(radius)
        if radius <= 0:
            return (mask > 0.5).to(dtype=mask.dtype)
        kernel_size = 2 * radius + 1
        return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)

    def build_query_boundary_ring(self, mask: torch.Tensor) -> torch.Tensor:
        outer_ring = self._dilate_binary_mask(mask, self.query_boundary_outer_dilation)
        if self.query_boundary_inner_dilation > 0:
            inner_ring = self._dilate_binary_mask(mask, self.query_boundary_inner_dilation)
        else:
            inner_ring = (mask > 0.5).to(dtype=mask.dtype)
        return (outer_ring - inner_ring).clamp_(0.0, 1.0)

    def build_query_boundary_inputs(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boundary_ring = self.build_query_boundary_ring(mask)
        boundary_rgb = image * boundary_ring
        return torch.cat([boundary_rgb, boundary_ring, mask], dim=1), boundary_ring

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

    def _flatten_token_map(self, token_map: torch.Tensor) -> torch.Tensor:
        return token_map.flatten(start_dim=2).transpose(1, 2).contiguous()

    def _pool_transport_map(
        self,
        feature_map: torch.Tensor,
        token_hw: tuple[int, int],
        *,
        mode: str,
    ) -> torch.Tensor:
        if mode == "avg":
            return self._pool_to_token_grid(feature_map, token_hw)
        if mode == "max":
            return F.adaptive_max_pool2d(feature_map, token_hw)
        raise ValueError(f"Unsupported transport pooling mode: {mode}")

    def _score_transport_candidates(
        self,
        query_token_map: torch.Tensor,
        key_token_map: torch.Tensor,
        valid_key_map: torch.Tensor,
        candidate_coords: torch.Tensor,
        *,
        query_mask_map: torch.Tensor,
        anchor_coords: torch.Tensor,
        extra_logits: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_candidates, _, height, width = candidate_coords.shape
        query_expanded = query_token_map.unsqueeze(1).expand(-1, num_candidates, -1, -1, -1)
        query_mask_expanded = query_mask_map.unsqueeze(1).expand(-1, num_candidates, -1, -1, -1)
        anchor_expanded = anchor_coords.unsqueeze(1).expand(-1, num_candidates, -1, -1, -1)
        candidate_delta = candidate_coords - anchor_expanded
        flat_query = query_expanded.reshape(batch_size * num_candidates, query_token_map.shape[1], height, width)
        flat_mask = query_mask_expanded.reshape(batch_size * num_candidates, 1, height, width)
        flat_coords = candidate_coords.reshape(batch_size * num_candidates, 2, height, width)
        flat_delta = candidate_delta.reshape(batch_size * num_candidates, 2, height, width)
        repeated_key_map = key_token_map.repeat_interleave(num_candidates, dim=0)
        repeated_valid_map = valid_key_map.repeat_interleave(num_candidates, dim=0)
        flat_sampled_key = self._sample_transport_map(repeated_key_map, flat_coords)
        flat_sampled_validity = self._sample_transport_map(repeated_valid_map, flat_coords).clamp_min(1e-6)
        score_inputs = torch.cat(
            [flat_query, flat_sampled_key, flat_mask, flat_sampled_validity, flat_coords, flat_delta],
            dim=1,
        )
        selector_logits = self.transport_candidate_score_head(score_inputs)
        selector_logits = selector_logits.view(batch_size, num_candidates, 1, height, width)
        if extra_logits is not None:
            selector_logits = selector_logits + extra_logits
        selector_logits = selector_logits / max(float(temperature), 1e-6)
        selector_probs = F.softmax(selector_logits, dim=1)
        sampled_key = flat_sampled_key.view(batch_size, num_candidates, key_token_map.shape[1], height, width)
        sampled_validity = flat_sampled_validity.view(batch_size, num_candidates, 1, height, width)
        return selector_logits, selector_probs, sampled_key, sampled_validity

    def _propose_transport_candidates(
        self,
        proposal_head: nn.Module,
        proposal_input: torch.Tensor,
        anchor_coords: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        proposal_offsets = proposal_head(proposal_input)
        batch_size, _, height, width = proposal_offsets.shape
        proposal_offsets = proposal_offsets.view(batch_size, self.transport_candidate_count, 2, height, width)
        proposal_offsets = torch.tanh(proposal_offsets) * scale
        return torch.clamp(anchor_coords.unsqueeze(1) + proposal_offsets, -1.0, 1.0)

    def _select_transport_candidates(
        self,
        query_token_map: torch.Tensor,
        key_token_map: torch.Tensor,
        valid_key_map: torch.Tensor,
        query_mask_map: torch.Tensor,
        candidate_coords: torch.Tensor,
        *,
        anchor_coords: torch.Tensor,
        fallback_coords: torch.Tensor,
        extra_logits: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, selector_probs, _, _ = self._score_transport_candidates(
            query_token_map,
            key_token_map,
            valid_key_map,
            candidate_coords,
            query_mask_map=query_mask_map,
            anchor_coords=anchor_coords,
            extra_logits=extra_logits,
            temperature=temperature,
        )
        if self.training:
            hard_idx = selector_probs.argmax(dim=1, keepdim=True)
            hard_probs = torch.zeros_like(selector_probs).scatter_(1, hard_idx, 1.0)
            selector_weights = selector_probs + (hard_probs - selector_probs).detach()
        else:
            hard_idx = selector_probs.argmax(dim=1, keepdim=True)
            selector_weights = torch.zeros_like(selector_probs).scatter_(1, hard_idx, 1.0)
        selected_coords = (selector_weights * candidate_coords).sum(dim=1)
        active_queries = (query_mask_map > 0.5).expand_as(selected_coords)
        selected_coords = torch.where(active_queries, selected_coords, fallback_coords)
        selection_strength = selector_probs.max(dim=1).values
        selection_strength = torch.where(
            query_mask_map > 0.5,
            selection_strength,
            torch.ones_like(selection_strength),
        )
        return selected_coords, selection_strength

    def _capacity_constrained_topk_assignment(
        self,
        top_logits: torch.Tensor,
        top_indices: torch.Tensor,
        top_valid_mask: torch.Tensor,
        *,
        per_key_capacity: int,
    ) -> torch.Tensor:
        if top_logits.numel() == 0:
            return torch.empty((0,), device=top_logits.device, dtype=torch.long)
        if per_key_capacity <= 0:
            per_key_capacity = 1

        assignment = torch.full((top_logits.shape[0],), -1, device=top_logits.device, dtype=torch.long)
        valid_pair_mask = top_valid_mask.bool()
        if not valid_pair_mask.any():
            return assignment

        cpu_logits = top_logits.detach().cpu()
        cpu_indices = top_indices.detach().cpu()
        cpu_valid = valid_pair_mask.detach().cpu()
        num_candidates = top_logits.shape[1]
        flat_scores = cpu_logits.masked_fill(~cpu_valid, float("-inf")).view(-1)
        flat_order = flat_scores.argsort(descending=True)
        key_usage: dict[int, int] = {}

        for flat_idx in flat_order.tolist():
            query_idx = flat_idx // num_candidates
            slot_idx = flat_idx % num_candidates
            if not bool(cpu_valid[query_idx, slot_idx]):
                break
            if int(assignment[query_idx].item()) >= 0:
                continue
            key_idx = int(cpu_indices[query_idx, slot_idx].item())
            if key_idx < 0:
                continue
            used = key_usage.get(key_idx, 0)
            if used >= per_key_capacity:
                continue
            assignment[query_idx] = slot_idx
            key_usage[key_idx] = used + 1

        for query_idx in (assignment < 0).nonzero(as_tuple=False).flatten().detach().cpu().tolist():
            valid_slots = cpu_valid[query_idx].nonzero(as_tuple=False).flatten()
            if valid_slots.numel() == 0:
                continue
            assignment[query_idx] = int(valid_slots[0].item())
        return assignment

    def _compute_transport_score_init(
        self,
        query_token_map: torch.Tensor,
        key_token_map: torch.Tensor,
        query_mask_map: torch.Tensor,
        valid_key_map: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        batch_size = query_token_map.shape[0]
        base_coords = self._get_normalized_token_coords(
            token_hw,
            dtype=query_token_map.dtype,
            device=query_token_map.device,
        ).expand(batch_size, -1, -1, -1)
        base_coords_flat = base_coords.flatten(start_dim=2).transpose(1, 2)
        score_coords_flat = base_coords_flat.clone()
        score_strength_flat = torch.zeros(
            (batch_size, base_coords_flat.shape[1]),
            device=query_token_map.device,
            dtype=query_token_map.dtype,
        )
        query_tokens = self._flatten_token_map(query_token_map)
        key_tokens = self._flatten_token_map(key_token_map)
        query_mask_flat = query_mask_map.flatten(start_dim=1) > 0.5
        key_valid_flat = valid_key_map.flatten(start_dim=1) > 0.5
        top_k = self.transport_score_top_k
        if top_k is None:
            top_k = min(32, base_coords_flat.shape[1])
        top_k = max(1, min(int(top_k), base_coords_flat.shape[1]))
        invalid_logit = -1e4
        candidate_logits = score_coords_flat.new_full(
            (batch_size, base_coords_flat.shape[1], top_k),
            invalid_logit,
        )
        candidate_key_indices = torch.full(
            (batch_size, base_coords_flat.shape[1], top_k),
            -1,
            device=query_tokens.device,
            dtype=torch.long,
        )
        candidate_coords = base_coords_flat.new_zeros((batch_size, base_coords_flat.shape[1], top_k, 2))

        for batch_idx in range(batch_size):
            query_indices = query_mask_flat[batch_idx].nonzero(as_tuple=False).flatten()
            key_indices = key_valid_flat[batch_idx].nonzero(as_tuple=False).flatten()
            if query_indices.numel() == 0 or key_indices.numel() == 0:
                continue

            raw_logits, _ = self.multihead_attention.compute_attention_logits(
                query_tokens[batch_idx : batch_idx + 1, query_indices],
                key_tokens[batch_idx : batch_idx + 1, key_indices],
            )
            logits = raw_logits.mean(dim=1).squeeze(0).float()
            key_coords = base_coords_flat[batch_idx, key_indices].float()

            if logits.shape[-1] > top_k:
                top_logits, top_positions = torch.topk(logits, k=top_k, dim=-1)
                top_indices = key_indices[top_positions]
                top_valid_mask = torch.ones_like(top_indices, dtype=torch.bool)
            else:
                pad_width = top_k - logits.shape[-1]
                top_logits = F.pad(logits, (0, pad_width), value=invalid_logit)
                top_indices = F.pad(key_indices.unsqueeze(0).expand(logits.shape[0], -1), (0, pad_width), value=-1)
                top_valid_mask = top_indices >= 0

            batch_candidate_coords = base_coords_flat[batch_idx, top_indices.clamp_min(0)]
            batch_candidate_coords = torch.where(
                top_valid_mask.unsqueeze(-1),
                batch_candidate_coords,
                torch.zeros_like(batch_candidate_coords),
            )
            candidate_logits[batch_idx, query_indices] = top_logits
            candidate_key_indices[batch_idx, query_indices] = top_indices
            candidate_coords[batch_idx, query_indices] = batch_candidate_coords

            masked_top_logits = torch.where(
                top_valid_mask,
                top_logits,
                torch.full_like(top_logits, invalid_logit),
            )
            probs = F.softmax(masked_top_logits / self.transport_score_temperature, dim=-1)
            if self.transport_score_assignment == "capacity_greedy":
                assigned_slots = self._capacity_constrained_topk_assignment(
                    masked_top_logits,
                    top_indices,
                    top_valid_mask,
                    per_key_capacity=self.transport_score_assignment_capacity,
                )
                safe_slots = assigned_slots.clamp_min(0)
                row_indices = torch.arange(batch_candidate_coords.shape[0], device=batch_candidate_coords.device)
                matched_coords = batch_candidate_coords[row_indices, safe_slots]
                score_strength = probs[row_indices, safe_slots]
                score_strength = torch.where(
                    assigned_slots >= 0,
                    score_strength,
                    torch.zeros_like(score_strength),
                )
            else:
                if self.training:
                    hard_idx = probs.argmax(dim=-1, keepdim=True)
                    hard = torch.zeros_like(probs).scatter_(1, hard_idx, 1.0)
                    weights = probs + (hard - probs).detach()
                else:
                    hard_idx = probs.argmax(dim=-1, keepdim=True)
                    weights = torch.zeros_like(probs).scatter_(1, hard_idx, 1.0)
                matched_coords = (batch_candidate_coords * weights.unsqueeze(-1)).sum(dim=1)
                score_strength = probs.max(dim=-1).values

            current_coords = base_coords_flat[batch_idx, query_indices].float()
            blended_coords = current_coords + self.transport_score_init_scale * (matched_coords - current_coords)
            blended_coords = torch.clamp(blended_coords, -1.0, 1.0)
            score_coords_flat[batch_idx, query_indices] = blended_coords.to(dtype=score_coords_flat.dtype)
            score_strength_flat[batch_idx, query_indices] = score_strength.to(dtype=score_strength_flat.dtype)

        score_coords = score_coords_flat.transpose(1, 2).contiguous().view(batch_size, 2, token_hw[0], token_hw[1])
        score_strength = score_strength_flat.view(batch_size, 1, token_hw[0], token_hw[1])
        candidate_valid_mask = candidate_key_indices >= 0
        masked_logits = torch.where(
            candidate_valid_mask,
            candidate_logits,
            torch.full_like(candidate_logits, invalid_logit),
        )
        has_valid = candidate_valid_mask.any(dim=-1, keepdim=True)
        safe_candidate_logits = torch.where(has_valid, masked_logits, torch.zeros_like(candidate_logits))
        scaled_logits = safe_candidate_logits / max(float(self.transport_score_temperature), 1e-6)
        candidate_probs = F.softmax(scaled_logits, dim=-1)
        candidate_log_probs = F.log_softmax(scaled_logits, dim=-1)
        return {
            "coords": score_coords,
            "strength": score_strength,
            "candidate_logits": safe_candidate_logits.float(),
            "candidate_probs": candidate_probs.float(),
            "candidate_log_probs": candidate_log_probs.float(),
            "candidate_key_indices": candidate_key_indices,
            "candidate_valid_mask": candidate_valid_mask,
            "candidate_coords": candidate_coords.float(),
        }

    def _compute_transport_confidence(
        self,
        query_token_map: torch.Tensor,
        sampled_key_tokens: torch.Tensor,
        query_mask_map: torch.Tensor,
        sampled_validity: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.transport_use_confidence or self.transport_confidence_head is None:
            return None
        confidence_input = torch.cat(
            [query_token_map, sampled_key_tokens, query_mask_map, sampled_validity, coords],
            dim=1,
        )
        confidence = torch.sigmoid(self.transport_confidence_head(confidence_input))
        return (confidence * query_mask_map) + (1.0 - query_mask_map)

    def _run_transport_local_refinement(
        self,
        query_token_map: torch.Tensor,
        key_token_map: torch.Tensor,
        valid_key_map: torch.Tensor,
        query_mask_map: torch.Tensor,
        coords: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> torch.Tensor:
        if not self.transport_use_local_refinement:
            return coords

        sampled_key_tokens = self._sample_transport_map(key_token_map, coords)
        sampled_validity = self._sample_transport_map(valid_key_map, coords)
        proposal_input = torch.cat(
            [query_token_map, sampled_key_tokens, query_mask_map, sampled_validity, coords],
            dim=1,
        )
        proposed_coords = self._propose_transport_candidates(
            self.transport_candidate_refine_head,
            proposal_input,
            coords,
            scale=self.transport_refine_scale,
        )
        candidate_coords = torch.cat([coords.unsqueeze(1), proposed_coords], dim=1)
        refined_coords, _ = self._select_transport_candidates(
            query_token_map,
            key_token_map,
            valid_key_map,
            query_mask_map,
            candidate_coords,
            anchor_coords=coords,
            fallback_coords=coords,
            temperature=self.transport_local_temperature,
        )
        return refined_coords

    def _run_transport_propagation(
        self,
        query_token_map: torch.Tensor,
        key_token_map: torch.Tensor,
        valid_key_map: torch.Tensor,
        coords: torch.Tensor,
        base_coords: torch.Tensor,
        query_mask_map: torch.Tensor,
        sampled_validity: torch.Tensor,
        confidence: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.transport_use_propagation or self.transport_propagation_head is None:
            return coords

        displacement = coords - base_coords
        propagation_inputs = [displacement, query_mask_map, sampled_validity]
        if confidence is not None:
            propagation_inputs.append(confidence)
        proposal_input = torch.cat(propagation_inputs, dim=1)
        proposed_coords = self._propose_transport_candidates(
            self.transport_propagation_head,
            proposal_input,
            coords,
            scale=self.transport_refine_scale,
        )
        candidate_coords = torch.cat([coords.unsqueeze(1), proposed_coords], dim=1)
        propagated_coords, _ = self._select_transport_candidates(
            query_token_map,
            key_token_map,
            valid_key_map,
            query_mask_map,
            candidate_coords,
            anchor_coords=coords,
            fallback_coords=coords,
        )
        return propagated_coords

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
        base_coords_flat = self._get_normalized_token_coords(
            token_hw,
            dtype=coords_flat.dtype,
            device=coords_flat.device,
        ).flatten(start_dim=2).transpose(1, 2).expand(batch_size, -1, -1)
        active_queries = query_mask_flat > 0.5
        if snap_mask_flat is not None:
            active_queries = active_queries & (snap_mask_flat > 0.5)
        valid_keys = key_valid_flat > 0.5
        has_valid_keys = valid_keys.any(dim=1, keepdim=True)
        valid_query_rows = active_queries & has_valid_keys

        deltas = coords_flat.float().unsqueeze(2) - base_coords_flat.float().unsqueeze(1)
        pairwise_dist = deltas.square().sum(dim=-1)
        large_distance = torch.finfo(pairwise_dist.dtype).max
        masked_dist = torch.where(
            valid_keys.unsqueeze(1),
            pairwise_dist,
            torch.full_like(pairwise_dist, large_distance),
        )
        nearest_valid = masked_dist.argmin(dim=-1)
        selected_indices = torch.where(
            valid_query_rows,
            nearest_valid,
            torch.full_like(nearest_valid, -1),
        )
        safe_selected = selected_indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, source_patch_values.shape[-1])
        gathered_values = torch.gather(source_patch_values, 1, safe_selected)
        snapped_values = torch.where(
            valid_query_rows.unsqueeze(-1),
            gathered_values,
            source_patch_values,
        )
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

        coarse_coords = None
        coarse_seed_coords = None
        init_coords = base_coords
        if self.transport_use_coarse_to_fine and self.transport_coarse_ratio > 1 and min(height, width) > 1:
            coarse_hw = (
                max(1, height // self.transport_coarse_ratio),
                max(1, width // self.transport_coarse_ratio),
            )
            if coarse_hw != token_hw:
                coarse_query_map = self._pool_transport_map(query_token_map, coarse_hw, mode="avg")
                coarse_key_map = self._pool_transport_map(key_token_map, coarse_hw, mode="avg")
                coarse_query_mask = self._pool_transport_map(query_mask_map, coarse_hw, mode="max")
                coarse_valid_key = self._pool_transport_map(valid_key_map, coarse_hw, mode="max")
                coarse_base_coords = self._get_normalized_token_coords(
                    coarse_hw,
                    dtype=query_token_map.dtype,
                    device=query_token_map.device,
                ).expand(batch_size, -1, -1, -1)
                coarse_init_coords = coarse_base_coords
                if self.transport_use_score_init:
                    coarse_score_state = self._compute_transport_score_init(
                        coarse_query_map,
                        coarse_key_map,
                        coarse_query_mask,
                        coarse_valid_key,
                        coarse_hw,
                    )
                    coarse_score_coords = coarse_score_state["coords"]
                    coarse_active = (coarse_query_mask > 0.5).expand_as(coarse_init_coords)
                    coarse_init_coords = torch.where(coarse_active, coarse_score_coords, coarse_init_coords)
                coarse_init_input = torch.cat([coarse_query_map, coarse_query_mask], dim=1)
                coarse_offsets = torch.tanh(self.transport_init_head(coarse_init_input)) * self.transport_offset_scale
                coarse_coords = torch.clamp(coarse_init_coords + coarse_offsets, -1.0, 1.0)
                coarse_sampled_key_tokens = self._sample_transport_map(coarse_key_map, coarse_coords)
                coarse_sampled_validity = self._sample_transport_map(coarse_valid_key, coarse_coords)
                for _ in range(self.transport_refine_steps):
                    coarse_refine_input = torch.cat(
                        [
                            coarse_query_map,
                            coarse_sampled_key_tokens,
                            coarse_query_mask,
                            coarse_sampled_validity,
                            coarse_coords,
                        ],
                        dim=1,
                    )
                    coarse_delta = torch.tanh(self.transport_refine_head(coarse_refine_input)) * self.transport_refine_scale
                    coarse_coords = torch.clamp(coarse_coords + coarse_delta, -1.0, 1.0)
                    coarse_sampled_key_tokens = self._sample_transport_map(coarse_key_map, coarse_coords)
                    coarse_sampled_validity = self._sample_transport_map(coarse_valid_key, coarse_coords)
                coarse_seed_coords = F.interpolate(coarse_coords, size=token_hw, mode="bilinear", align_corners=True)
                coarse_seed_key_tokens = self._sample_transport_map(key_token_map, coarse_seed_coords)
                coarse_seed_validity = self._sample_transport_map(valid_key_map, coarse_seed_coords)
                coarse_seed_input = torch.cat(
                    [
                        query_token_map,
                        coarse_seed_key_tokens,
                        query_mask_map,
                        coarse_seed_validity,
                        coarse_seed_coords,
                    ],
                    dim=1,
                )
                coarse_seed_candidates = self._propose_transport_candidates(
                    self.transport_candidate_refine_head,
                    coarse_seed_input,
                    coarse_seed_coords,
                    scale=self.transport_refine_scale,
                )
                coarse_seed_coords, _ = self._select_transport_candidates(
                    query_token_map,
                    key_token_map,
                    valid_key_map,
                    query_mask_map,
                    torch.cat([coarse_seed_coords.unsqueeze(1), coarse_seed_candidates], dim=1),
                    anchor_coords=coarse_seed_coords,
                    fallback_coords=base_coords,
                    temperature=self.transport_local_temperature,
                )

        score_init_coords = None
        score_init_strength = None
        score_init_scorer_state = None
        if self.transport_use_score_init:
            score_init_scorer_state = self._compute_transport_score_init(
                query_token_map,
                key_token_map,
                query_mask_map,
                valid_key_map,
                token_hw,
            )
            score_init_coords = score_init_scorer_state["coords"]
            score_init_strength = score_init_scorer_state["strength"]

        init_candidates = [base_coords]
        if coarse_seed_coords is not None:
            init_candidates.append(coarse_seed_coords)
        if score_init_coords is not None:
            init_candidates.append(score_init_coords)
        if len(init_candidates) > 1:
            init_coords, _ = self._select_transport_candidates(
                query_token_map,
                key_token_map,
                valid_key_map,
                query_mask_map,
                torch.stack(init_candidates, dim=1),
                anchor_coords=base_coords,
                fallback_coords=base_coords,
                temperature=self.transport_score_temperature,
            )

        init_input = torch.cat([query_token_map, query_mask_map], dim=1)
        init_offsets = torch.tanh(self.transport_init_head(init_input)) * self.transport_offset_scale
        coords = torch.clamp(init_coords + init_offsets, -1.0, 1.0)

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

        coords = self._run_transport_local_refinement(
            query_token_map,
            key_token_map,
            valid_key_map,
            query_mask_map,
            coords,
            token_hw,
        )
        sampled_key_tokens = self._sample_transport_map(key_token_map, coords)
        sampled_validity = self._sample_transport_map(valid_key_map, coords)

        confidence = self._compute_transport_confidence(
            query_token_map,
            sampled_key_tokens,
            query_mask_map,
            sampled_validity,
            coords,
        )
        for _ in range(self.transport_propagation_steps):
            coords = self._run_transport_propagation(
                query_token_map,
                key_token_map,
                valid_key_map,
                coords,
                base_coords,
                query_mask_map,
                sampled_validity,
                confidence,
            )
            sampled_key_tokens = self._sample_transport_map(key_token_map, coords)
            sampled_validity = self._sample_transport_map(valid_key_map, coords)
            confidence = self._compute_transport_confidence(
                query_token_map,
                sampled_key_tokens,
                query_mask_map,
                sampled_validity,
                coords,
            )

        return {
            "base_coords": base_coords,
            "coords": coords,
            "sampled_key_tokens": sampled_key_tokens,
            "sampled_validity": sampled_validity,
            "coarse_coords": coarse_coords,
            "score_init_coords": score_init_coords,
            "score_init_strength": score_init_strength,
            "score_init_scorer_state": score_init_scorer_state,
            "confidence": confidence,
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
        confidence = transport_state.get("confidence")
        coarse_coords = transport_state.get("coarse_coords")
        score_init_coords = transport_state.get("score_init_coords")
        score_init_strength = transport_state.get("score_init_strength")

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
        confidence_flat = None
        if confidence is not None:
            confidence_flat = confidence.flatten(start_dim=2).transpose(1, 2).squeeze(-1)
        effective_validity_flat = sampled_validity_flat
        if confidence_flat is not None:
            effective_validity_flat = effective_validity_flat * confidence_flat
        fallback_mask = effective_validity_flat < self.transport_fallback_validity_threshold
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
        if confidence_flat is not None:
            aux["transport_confidence"] = confidence_flat
        if coarse_coords is not None:
            aux["transport_coarse_coords"] = coarse_coords.flatten(start_dim=2).transpose(1, 2)
        if score_init_coords is not None:
            aux["transport_score_init_coords"] = score_init_coords.flatten(start_dim=2).transpose(1, 2)
        if score_init_strength is not None:
            aux["transport_score_init_strength"] = score_init_strength.flatten(start_dim=2).transpose(1, 2).squeeze(-1)
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
        masked_queries = query_mask_flat > 0.5
        identity = torch.eye(num_patches, device=coords_flat.device, dtype=value_dtype)
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
        if selected_indices is not None:
            valid_rows = masked_queries & (selected_indices >= 0)
            safe_selected = selected_indices.clamp_min(0)
            replacement_rows = F.one_hot(safe_selected, num_classes=num_patches).to(dtype=value_dtype)
            replacement_rows = replacement_rows * valid_rows.unsqueeze(-1).to(dtype=value_dtype)
            dense_attn = torch.where(masked_queries.unsqueeze(-1), replacement_rows, identity)
            return dense_attn.unsqueeze(1)

        width_scale = max(width - 1, 1)
        height_scale = max(height - 1, 1)
        coords = coords_flat
        if validity_flat is None:
            strength = torch.ones(
                (batch_size, num_patches, 1),
                device=coords.device,
                dtype=value_dtype,
            )
        else:
            strength = validity_flat.to(dtype=value_dtype).unsqueeze(-1)

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
        ).to(dtype=value_dtype)
        replacement_rows = torch.zeros(
            (batch_size, num_patches, num_patches),
            device=coords.device,
            dtype=value_dtype,
        )
        weighted = bilinear_weights * strength * masked_queries.unsqueeze(-1).to(dtype=value_dtype)
        replacement_rows.scatter_add_(2, candidate_indices, weighted)
        dense_attn = torch.where(masked_queries.unsqueeze(-1), replacement_rows, identity)
        return dense_attn.unsqueeze(1)

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
        attention_supervision_entries = None
        if return_aux_entries and self.transport_direct_scorer_supervision:
            attention_supervision_entries = self.build_attention_supervision_entries(
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
        if attention_supervision_entries is not None:
            aux["attention_supervision_entries"] = attention_supervision_entries
            aux["transport_descriptor_supervision"] = True
        sampled_values_flat = aux["transport_values"]
        sampled_validity_flat = aux["transport_effective_validity"]
        coords_flat = aux["transport_coords"]
        selected_indices = aux.get("transport_selected_indices")

        mixed = self._flatten_patch_map(source_patch_map)
        mixed = default_tokens.clone() if default_tokens is not None else mixed.clone()
        query_rows = (query_mask_flat > 0.5).unsqueeze(-1)
        mixed = torch.where(query_rows, sampled_values_flat, mixed)

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
