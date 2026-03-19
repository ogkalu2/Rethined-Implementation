from __future__ import annotations

import torch
import torch.nn.functional as F


class TransportLossMixin:
    def _transport_patch_loss(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = refined_target.new_zeros(())
        metrics = {"transport_patch": 0.0}
        if self.transport_patch_weight <= 0 or attention_aux is None:
            return zero, metrics
        if attention_aux.get("copy_mode") != "transport":
            return zero, metrics

        query_mask_flat = attention_aux.get("query_mask_flat")
        # Prefer the differentiable pre-snap transport samples when available.
        transport_values = attention_aux.get("transport_copy_values")
        if transport_values is None:
            transport_values = attention_aux.get("transport_values")
        transport_validity = attention_aux.get("transport_validity")
        kernel_size = int(attention_aux.get("kernel_size", 0))
        value_patch_size = int(attention_aux.get("value_patch_size", 0))
        value_patch_padding = int(attention_aux.get("value_patch_padding", 0))
        if (
            query_mask_flat is None
            or transport_values is None
            or kernel_size <= 0
            or value_patch_size <= 0
        ):
            return zero, metrics

        target_patches = self._extract_patch_tokens(
            refined_target,
            patch_size=value_patch_size,
            stride=kernel_size,
            padding=value_patch_padding,
        ).detach()
        losses = []
        for batch_idx in range(target_patches.shape[0]):
            masked_queries = query_mask_flat[batch_idx] > 0.5
            if masked_queries.any():
                per_query = (
                    transport_values[batch_idx, masked_queries] - target_patches[batch_idx, masked_queries]
                ).abs().mean(dim=-1)
                if transport_validity is not None:
                    weights = transport_validity[batch_idx, masked_queries].detach().float().clamp_min(1e-3)
                    losses.append((per_query * weights).sum() / weights.sum().clamp_min(1e-6))
                else:
                    losses.append(per_query.mean())
        if not losses:
            return zero, metrics

        loss = torch.stack(losses).mean()
        metrics["transport_patch"] = loss.item()
        return loss, metrics

    def _transport_self_patch_loss(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = refined_target.new_zeros(())
        metrics = {"transport_self_patch": 0.0}
        if self.transport_self_patch_weight <= 0 or attention_aux is None:
            return zero, metrics

        transport_self_aux = attention_aux.get("transport_self_aux")
        kernel_size = int(attention_aux.get("kernel_size", 0))
        value_patch_size = int(attention_aux.get("value_patch_size", 0))
        value_patch_padding = int(attention_aux.get("value_patch_padding", 0))
        if (
            transport_self_aux is None
            or kernel_size <= 0
            or value_patch_size <= 0
        ):
            return zero, metrics

        query_mask_flat = transport_self_aux.get("query_mask_flat")
        transport_values = transport_self_aux.get("transport_copy_values")
        if transport_values is None:
            transport_values = transport_self_aux.get("transport_values")
        transport_validity = transport_self_aux.get("transport_validity")
        if query_mask_flat is None or transport_values is None:
            return zero, metrics

        target_patches = self._extract_patch_tokens(
            refined_target,
            patch_size=value_patch_size,
            stride=kernel_size,
            padding=value_patch_padding,
        ).detach()
        losses = []
        for batch_idx in range(target_patches.shape[0]):
            self_queries = query_mask_flat[batch_idx] > 0.5
            if self_queries.any():
                per_query = (
                    transport_values[batch_idx, self_queries] - target_patches[batch_idx, self_queries]
                ).abs().mean(dim=-1)
                if transport_validity is not None:
                    weights = transport_validity[batch_idx, self_queries].detach().float().clamp_min(1e-3)
                    losses.append((per_query * weights).sum() / weights.sum().clamp_min(1e-6))
                else:
                    losses.append(per_query.mean())
        if not losses:
            return zero, metrics

        loss = torch.stack(losses).mean()
        metrics["transport_self_patch"] = loss.item()
        return loss, metrics

    def _transport_validity_loss(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = refined_target.new_zeros(())
        metrics = {
            "transport_validity": 0.0,
            "transport_valid_ratio": 0.0,
            "transport_fallback_ratio": 0.0,
        }
        if attention_aux is None or attention_aux.get("copy_mode") != "transport":
            return zero, metrics

        query_mask_flat = attention_aux.get("query_mask_flat")
        transport_validity = attention_aux.get("transport_validity")
        fallback_mask = attention_aux.get("transport_fallback_mask")
        if query_mask_flat is None or transport_validity is None:
            return zero, metrics

        masked_queries = query_mask_flat > 0.5
        if not masked_queries.any():
            return zero, metrics

        valid_ratio = transport_validity[masked_queries].float().mean()
        metrics["transport_valid_ratio"] = valid_ratio.item()
        if fallback_mask is not None:
            metrics["transport_fallback_ratio"] = fallback_mask[masked_queries].float().mean().item()
        if self.transport_validity_weight <= 0:
            return zero, metrics

        loss = 1.0 - valid_ratio
        metrics["transport_validity"] = loss.item()
        return loss, metrics

    def _transport_self_validity_loss(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = refined_target.new_zeros(())
        metrics = {
            "transport_self_validity": 0.0,
            "transport_self_valid_ratio": 0.0,
        }
        if attention_aux is None:
            return zero, metrics

        transport_self_aux = attention_aux.get("transport_self_aux")
        if transport_self_aux is None:
            return zero, metrics

        query_mask_flat = transport_self_aux.get("query_mask_flat")
        transport_validity = transport_self_aux.get("transport_validity")
        if query_mask_flat is None or transport_validity is None:
            return zero, metrics

        self_queries = query_mask_flat > 0.5
        if not self_queries.any():
            return zero, metrics

        valid_ratio = transport_validity[self_queries].float().mean()
        metrics["transport_self_valid_ratio"] = valid_ratio.item()
        if self.transport_self_validity_weight <= 0:
            return zero, metrics

        loss = 1.0 - valid_ratio
        metrics["transport_self_validity"] = loss.item()
        return loss, metrics

    def _transport_offset_smoothness_loss(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = refined_target.new_zeros(())
        metrics = {
            "transport_offset_smoothness": 0.0,
            "transport_offset_curvature": 0.0,
        }
        if self.transport_offset_smoothness_weight <= 0 or attention_aux is None:
            return zero, metrics
        if attention_aux.get("copy_mode") != "transport":
            return zero, metrics

        query_mask_flat = attention_aux.get("query_mask_flat")
        coords_flat = attention_aux.get("transport_coords")
        base_coords_flat = attention_aux.get("transport_base_coords")
        kernel_size = int(attention_aux.get("kernel_size", 0))
        if query_mask_flat is None or coords_flat is None or base_coords_flat is None or kernel_size <= 0:
            return zero, metrics

        token_grid_size = int(refined_target.shape[-1] // kernel_size)
        coord_map = coords_flat.transpose(1, 2).contiguous().view(coords_flat.shape[0], 2, token_grid_size, token_grid_size)
        base_coord_map = base_coords_flat.transpose(1, 2).contiguous().view(
            base_coords_flat.shape[0],
            2,
            token_grid_size,
            token_grid_size,
        )
        displacement_map = coord_map - base_coord_map
        query_mask_map = query_mask_flat.view(coords_flat.shape[0], 1, token_grid_size, token_grid_size).to(dtype=coord_map.dtype)
        token_target = F.adaptive_avg_pool2d(refined_target, (token_grid_size, token_grid_size))
        edge_x = torch.exp(
            -self.transport_offset_edge_scale
            * (token_target[:, :, :, 1:] - token_target[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        )
        edge_y = torch.exp(
            -self.transport_offset_edge_scale
            * (token_target[:, :, 1:, :] - token_target[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
        )
        mask_x = query_mask_map[:, :, :, 1:] * query_mask_map[:, :, :, :-1]
        mask_y = query_mask_map[:, :, 1:, :] * query_mask_map[:, :, :-1, :]
        diff_x = (displacement_map[:, :, :, 1:] - displacement_map[:, :, :, :-1]).abs().sum(dim=1, keepdim=True)
        diff_y = (displacement_map[:, :, 1:, :] - displacement_map[:, :, :-1, :]).abs().sum(dim=1, keepdim=True)
        denom_x = (mask_x * edge_x).sum().clamp_min(1e-6)
        denom_y = (mask_y * edge_y).sum().clamp_min(1e-6)
        first_order = 0.5 * (((diff_x * mask_x * edge_x).sum() / denom_x) + ((diff_y * mask_y * edge_y).sum() / denom_y))

        curvature = zero
        if token_grid_size >= 3:
            edge_xx = edge_x[:, :, :, 1:] * edge_x[:, :, :, :-1]
            edge_yy = edge_y[:, :, 1:, :] * edge_y[:, :, :-1, :]
            mask_xx = (
                query_mask_map[:, :, :, 2:]
                * query_mask_map[:, :, :, 1:-1]
                * query_mask_map[:, :, :, :-2]
            )
            mask_yy = (
                query_mask_map[:, :, 2:, :]
                * query_mask_map[:, :, 1:-1, :]
                * query_mask_map[:, :, :-2, :]
            )
            curvature_x = (
                displacement_map[:, :, :, 2:]
                - (2.0 * displacement_map[:, :, :, 1:-1])
                + displacement_map[:, :, :, :-2]
            ).abs().sum(dim=1, keepdim=True)
            curvature_y = (
                displacement_map[:, :, 2:, :]
                - (2.0 * displacement_map[:, :, 1:-1, :])
                + displacement_map[:, :, :-2, :]
            ).abs().sum(dim=1, keepdim=True)
            denom_xx = (mask_xx * edge_xx).sum().clamp_min(1e-6)
            denom_yy = (mask_yy * edge_yy).sum().clamp_min(1e-6)
            curvature = 0.5 * (
                ((curvature_x * mask_xx * edge_xx).sum() / denom_xx)
                + ((curvature_y * mask_yy * edge_yy).sum() / denom_yy)
            )

        loss = first_order + (0.5 * curvature)
        metrics["transport_offset_smoothness"] = loss.item()
        metrics["transport_offset_curvature"] = curvature.item()
        return loss, metrics

    def _transport_cycle_terms(
        self,
        attention_aux: dict[str, object] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if attention_aux is None or attention_aux.get("copy_mode") != "transport":
            return None, None, None

        query_mask_flat = attention_aux.get("query_mask_flat")
        base_coords = attention_aux.get("transport_base_coords")
        cycle_back_coords = attention_aux.get("transport_cycle_back_coords")
        transport_validity = attention_aux.get("transport_validity")
        if (
            query_mask_flat is None
            or base_coords is None
            or cycle_back_coords is None
            or transport_validity is None
        ):
            return None, None, None
        masked_queries = query_mask_flat > 0.5
        if not masked_queries.any():
            return None, None, None

        cycle_error = (cycle_back_coords - base_coords).abs().mean(dim=-1)
        return masked_queries, cycle_error, transport_validity

    def _transport_cycle_consistency_loss(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = refined_target.new_zeros(())
        metrics = {
            "transport_cycle_consistency": 0.0,
            "transport_cycle_error": 0.0,
        }
        masked_queries, cycle_error, transport_validity = self._transport_cycle_terms(attention_aux)
        if masked_queries is None or cycle_error is None or transport_validity is None:
            return zero, metrics

        masked_error = cycle_error[masked_queries]
        metrics["transport_cycle_error"] = masked_error.mean().item()
        if self.transport_cycle_consistency_weight <= 0:
            return zero, metrics

        weights = 0.25 + (0.75 * transport_validity[masked_queries].detach().clamp(0.0, 1.0))
        loss = (masked_error * weights).sum() / weights.sum().clamp_min(1e-6)
        metrics["transport_cycle_consistency"] = loss.item()
        return loss, metrics

    def _transport_confidence_loss(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = refined_target.new_zeros(())
        metrics = {
            "transport_confidence": 0.0,
            "transport_confidence_mean": 0.0,
        }
        if attention_aux is None or attention_aux.get("copy_mode") != "transport":
            return zero, metrics

        transport_confidence = attention_aux.get("transport_confidence")
        masked_queries, cycle_error, transport_validity = self._transport_cycle_terms(attention_aux)
        if (
            transport_confidence is None
            or masked_queries is None
            or cycle_error is None
            or transport_validity is None
        ):
            return zero, metrics

        masked_confidence = transport_confidence[masked_queries].float()
        metrics["transport_confidence_mean"] = masked_confidence.mean().item()
        if self.transport_confidence_weight <= 0:
            return zero, metrics

        target = torch.exp(-4.0 * cycle_error.detach()) * transport_validity.detach().clamp(0.0, 1.0)
        masked_target = target[masked_queries].float()
        loss = F.mse_loss(masked_confidence, masked_target)
        metrics["transport_confidence"] = loss.item()
        return loss, metrics
