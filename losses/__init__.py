from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.perceptual import PerceptualLoss
from losses.transport import TransportLossMixin


def composite_with_known(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return pred * mask + target * (1 - mask)


def _expanded_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == ref.shape[1]:
        return mask
    return mask.expand(-1, ref.shape[1], -1, -1)


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = _expanded_mask(mask.to(dtype=pred.dtype), pred)
    diff = (pred - target).abs() * mask
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(eps)
    return (diff.sum(dim=(1, 2, 3)) / denom).mean()


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = _expanded_mask(mask.to(dtype=pred.dtype), pred)
    diff = (pred - target).pow(2) * mask
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(eps)
    return (diff.sum(dim=(1, 2, 3)) / denom).mean()


def dilate_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def masked_gradient_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = _expanded_mask(mask.to(dtype=pred.dtype), pred)

    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    dx_loss = (pred_dx - target_dx).abs() * mask_dx
    dy_loss = (pred_dy - target_dy).abs() * mask_dy

    dx_denom = mask_dx.sum(dim=(1, 2, 3)).clamp_min(eps)
    dy_denom = mask_dy.sum(dim=(1, 2, 3)).clamp_min(eps)
    dx_mean = dx_loss.sum(dim=(1, 2, 3)) / dx_denom
    dy_mean = dy_loss.sum(dim=(1, 2, 3)) / dy_denom
    return 0.5 * (dx_mean.mean() + dy_mean.mean())


class InpaintingLoss(TransportLossMixin, nn.Module):
    """Training losses for refinement, retrieval, and active transport supervision."""

    SCHEDULED_WEIGHT_NAMES = (
        "retrieval_loss_weight",
        "perceptual_weight",
    )

    def __init__(
        self,
        coarse_l2_weight: float = 1.0,
        refined_l1_weight: float = 1.0,
        refined_query_patch_l1_weight: float = 0.0,
        retrieval_loss_weight: float = 0.0,
        retrieval_hard_ce_weight: float = 0.0,
        retrieval_top1_margin_weight: float = 0.0,
        retrieval_top1_margin: float = 0.0,
        retrieval_coherence_weight: float = 0.0,
        retrieval_coherence_sigma: float = 0.1,
        retrieval_teacher_patch_padding: int = 8,
        retrieval_teacher_temperature: float = 0.07,
        retrieval_target_margin_pct: float = 0.03,
        transport_patch_weight: float = 1.0,
        transport_selection_weight: float = 0.0,
        transport_selection_temperature: float = 1.0,
        transport_validity_weight: float = 0.1,
        transport_offset_smoothness_weight: float = 0.01,
        transport_offset_edge_scale: float = 5.0,
        loss_schedule_focus_steps: int = 0,
        loss_schedule_transition_steps: int = 0,
        retrieval_loss_weight_start: float | None = None,
        perceptual_weight_start: float | None = None,
        perceptual_weight: float = 0.1,
    ):
        super().__init__()
        self.coarse_l2_weight = float(coarse_l2_weight)
        self.refined_l1_weight = float(refined_l1_weight)
        self.refined_query_patch_l1_weight = float(refined_query_patch_l1_weight)
        self.retrieval_loss_weight = float(retrieval_loss_weight)
        self.retrieval_hard_ce_weight = float(retrieval_hard_ce_weight)
        self.retrieval_top1_margin_weight = float(retrieval_top1_margin_weight)
        self.retrieval_top1_margin = float(retrieval_top1_margin)
        self.retrieval_coherence_weight = float(retrieval_coherence_weight)
        self.retrieval_coherence_sigma = float(retrieval_coherence_sigma)
        self.retrieval_teacher_patch_padding = max(0, int(retrieval_teacher_patch_padding))
        self.retrieval_teacher_temperature = float(retrieval_teacher_temperature)
        self.retrieval_target_margin_pct = max(0.0, float(retrieval_target_margin_pct))
        self.transport_patch_weight = float(transport_patch_weight)
        self.transport_selection_weight = float(transport_selection_weight)
        self.transport_selection_temperature = float(transport_selection_temperature)
        self.transport_validity_weight = float(transport_validity_weight)
        self.transport_offset_smoothness_weight = float(transport_offset_smoothness_weight)
        self.transport_offset_edge_scale = float(transport_offset_edge_scale)
        self.loss_schedule_focus_steps = max(0, int(loss_schedule_focus_steps))
        self.loss_schedule_transition_steps = max(0, int(loss_schedule_transition_steps))
        self.perceptual_weight = float(perceptual_weight)
        if self.retrieval_teacher_temperature <= 0:
            raise ValueError("retrieval_teacher_temperature must be positive.")
        if self.retrieval_top1_margin < 0:
            raise ValueError("retrieval_top1_margin must be non-negative.")
        if self.retrieval_coherence_sigma <= 0:
            raise ValueError("retrieval_coherence_sigma must be positive.")
        if self.transport_patch_weight < 0:
            raise ValueError("transport_patch_weight must be non-negative.")
        if self.transport_selection_weight < 0:
            raise ValueError("transport_selection_weight must be non-negative.")
        if self.transport_selection_temperature <= 0:
            raise ValueError("transport_selection_temperature must be positive.")
        if self.transport_validity_weight < 0:
            raise ValueError("transport_validity_weight must be non-negative.")
        if self.transport_offset_smoothness_weight < 0:
            raise ValueError("transport_offset_smoothness_weight must be non-negative.")
        if self.transport_offset_edge_scale < 0:
            raise ValueError("transport_offset_edge_scale must be non-negative.")

        self.perceptual_loss = PerceptualLoss()
        self.current_training_step = 0
        self._base_weight_values = {
            name: float(getattr(self, name))
            for name in self.SCHEDULED_WEIGHT_NAMES
        }
        self._focus_weight_values = {
            "retrieval_loss_weight": (
                self._base_weight_values["retrieval_loss_weight"]
                if retrieval_loss_weight_start is None
                else float(retrieval_loss_weight_start)
            ),
            "perceptual_weight": (
                self._base_weight_values["perceptual_weight"]
                if perceptual_weight_start is None
                else float(perceptual_weight_start)
            ),
        }

    def set_training_step(self, step: int):
        self.current_training_step = max(0, int(step))

    def retrieval_supervision_enabled(self) -> bool:
        return (
            self._get_scheduled_weight("retrieval_loss_weight") > 0
            or self.retrieval_hard_ce_weight > 0
            or self.retrieval_top1_margin_weight > 0
            or self.retrieval_coherence_weight > 0
        )

    def transport_patch_supervision_enabled(self) -> bool:
        return self.transport_patch_weight > 0

    def transport_selection_supervision_enabled(self) -> bool:
        return self.transport_selection_weight > 0

    def transport_validity_supervision_enabled(self) -> bool:
        return self.transport_validity_weight > 0

    def transport_smoothness_supervision_enabled(self) -> bool:
        return self.transport_offset_smoothness_weight > 0

    def _get_scheduled_weight(self, name: str) -> float:
        base_weight = self._base_weight_values[name]
        focus_weight = self._focus_weight_values[name]
        if self.loss_schedule_focus_steps <= 0 and self.loss_schedule_transition_steps <= 0:
            return base_weight
        if self.current_training_step <= self.loss_schedule_focus_steps:
            return focus_weight
        if self.loss_schedule_transition_steps <= 0:
            return base_weight

        transition_progress = (
            (self.current_training_step - self.loss_schedule_focus_steps)
            / float(self.loss_schedule_transition_steps)
        )
        transition_progress = min(max(transition_progress, 0.0), 1.0)
        return focus_weight + transition_progress * (base_weight - focus_weight)

    def _extract_patch_tokens(
        self,
        image: torch.Tensor,
        *,
        patch_size: int,
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        if padding > 0:
            image = F.pad(image, (padding, padding, padding, padding), mode="reflect")
        patches = F.unfold(image, kernel_size=patch_size, stride=stride)
        return patches.transpose(1, 2).contiguous()

    def _query_patch_l1_loss(
        self,
        refined: torch.Tensor,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> torch.Tensor:
        if attention_aux is None:
            return refined.new_zeros(())

        kernel_size = int(attention_aux.get("kernel_size", 0))
        query_mask_flat = attention_aux.get("query_mask_flat")
        if kernel_size <= 0 or query_mask_flat is None:
            return refined.new_zeros(())

        refined_patches = self._extract_patch_tokens(
            refined,
            patch_size=kernel_size,
            stride=kernel_size,
            padding=0,
        )
        target_patches = self._extract_patch_tokens(
            refined_target,
            patch_size=kernel_size,
            stride=kernel_size,
            padding=0,
        )
        per_patch_l1 = (refined_patches - target_patches).abs().mean(dim=-1)
        query_mask = query_mask_flat > 0.5

        losses = []
        for batch_idx in range(per_patch_l1.shape[0]):
            masked_queries = query_mask[batch_idx]
            if masked_queries.any():
                losses.append(per_patch_l1[batch_idx, masked_queries].mean())
        if not losses:
            return refined.new_zeros(())
        return torch.stack(losses).mean()

    def _normalize_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return patch_tokens.float()

    def _normalized_token_coords(
        self,
        indices: torch.Tensor,
        token_hw: tuple[int, int],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        height, width = token_hw
        ys = torch.div(indices, width, rounding_mode="floor").to(dtype=dtype)
        xs = (indices % width).to(dtype=dtype)
        if width > 1:
            xs = xs / float(width - 1)
        if height > 1:
            ys = ys / float(height - 1)
        return torch.stack([xs, ys], dim=-1)

    def _normalized_transport_token_coords(
        self,
        indices: torch.Tensor,
        token_hw: tuple[int, int],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        height, width = token_hw
        ys = torch.div(indices, width, rounding_mode="floor").to(dtype=dtype)
        xs = (indices % width).to(dtype=dtype)
        if width > 1:
            xs = (xs / float(width - 1)) * 2.0 - 1.0
        else:
            xs = torch.full_like(xs, -1.0)
        if height > 1:
            ys = (ys / float(height - 1)) * 2.0 - 1.0
        else:
            ys = torch.full_like(ys, -1.0)
        return torch.stack([xs, ys], dim=-1)

    def _adjacent_query_pairs(
        self,
        query_indices: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_indices.numel() <= 1:
            empty = query_indices.new_empty((0,), dtype=torch.long)
            return empty, empty

        height, width = token_hw
        num_tokens = height * width
        order = torch.full((num_tokens,), -1, device=query_indices.device, dtype=torch.long)
        local_order = torch.arange(query_indices.numel(), device=query_indices.device)
        order[query_indices] = local_order

        right_mask = (query_indices % width) < (width - 1)
        right_sources = local_order[right_mask]
        right_targets = order[query_indices[right_mask] + 1]
        right_valid = right_targets >= 0

        down_mask = (query_indices + width) < num_tokens
        down_sources = local_order[down_mask]
        down_targets = order[query_indices[down_mask] + width]
        down_valid = down_targets >= 0

        src = torch.cat([right_sources[right_valid], down_sources[down_valid]], dim=0)
        dst = torch.cat([right_targets[right_valid], down_targets[down_valid]], dim=0)
        return src, dst

    def _retrieval_coherence_loss(
        self,
        pred_probs: torch.Tensor,
        query_indices: torch.Tensor,
        key_indices: torch.Tensor,
        query_teacher_tokens: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> torch.Tensor:
        if pred_probs.shape[0] <= 1 or pred_probs.shape[-1] <= 0:
            return pred_probs.new_zeros(())

        pair_src, pair_dst = self._adjacent_query_pairs(query_indices, token_hw)
        if pair_src.numel() == 0:
            return pred_probs.new_zeros(())

        key_coords = self._normalized_token_coords(
            key_indices,
            token_hw,
            dtype=pred_probs.dtype,
        )
        expected_source_coords = pred_probs @ key_coords

        query_patch_dists = (
            query_teacher_tokens[pair_src] - query_teacher_tokens[pair_dst]
        ).abs().mean(dim=-1)
        pair_weights = torch.exp(-query_patch_dists / self.retrieval_coherence_sigma)
        coord_diffs = F.smooth_l1_loss(
            expected_source_coords[pair_src],
            expected_source_coords[pair_dst],
            reduction="none",
        ).mean(dim=-1)
        return (coord_diffs * pair_weights).mean()

    def _build_transport_supervision_entries(
        self,
        attention_aux: dict[str, object],
    ) -> list[dict[str, torch.Tensor | tuple[int, int]]] | None:
        if attention_aux.get("copy_mode") != "transport":
            return None

        query_mask_flat = attention_aux.get("query_mask_flat")
        key_valid_flat = attention_aux.get("key_valid_flat")
        transport_coords = attention_aux.get("transport_selection_coords")
        if transport_coords is None:
            transport_coords = attention_aux.get("transport_coords")
        token_hw = attention_aux.get("token_hw")
        if (
            query_mask_flat is None
            or key_valid_flat is None
            or transport_coords is None
            or token_hw is None
        ):
            return None

        token_hw = tuple(token_hw)
        masked_query_mask = query_mask_flat > 0.5
        valid_key_mask = key_valid_flat > 0.5
        supervision_entries: list[dict[str, torch.Tensor | tuple[int, int]]] = []

        for batch_idx in range(transport_coords.shape[0]):
            query_indices = masked_query_mask[batch_idx].nonzero(as_tuple=False).flatten()
            key_indices = valid_key_mask[batch_idx].nonzero(as_tuple=False).flatten()
            entry: dict[str, torch.Tensor | tuple[int, int]] = {
                "query_indices": query_indices,
                "key_indices": key_indices,
                "token_hw": token_hw,
            }
            if query_indices.numel() == 0 or key_indices.numel() == 0:
                entry["raw_logits"] = transport_coords.new_empty((0, 0), dtype=torch.float32)
                supervision_entries.append(entry)
                continue

            query_coords = transport_coords[batch_idx, query_indices].float()
            key_coords = self._normalized_transport_token_coords(
                key_indices,
                token_hw,
                dtype=query_coords.dtype,
            )
            coord_distances = torch.cdist(
                query_coords.unsqueeze(0),
                key_coords.unsqueeze(0),
                p=2,
            ).squeeze(0)
            entry["raw_logits"] = (-coord_distances).to(dtype=torch.float32)
            supervision_entries.append(entry)

        return supervision_entries

    def _attention_supervision_losses(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
        *,
        metric_prefix: str = "retrieval",
        allow_metric_only: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        zero = refined_target.new_zeros(())
        if attention_aux is None:
            return {}, {}
        if not self.retrieval_supervision_enabled() and not allow_metric_only:
            return {}, {}

        loss_terms = {}
        metrics = {}
        kernel_size = int(attention_aux.get("kernel_size", 0))
        query_mask_flat = attention_aux.get("query_mask_flat")
        supervision_entries = attention_aux.get("attention_supervision_entries")
        compute_retrieval_losses = self.retrieval_supervision_enabled()
        if supervision_entries is None and attention_aux.get("copy_mode") == "transport":
            supervision_entries = self._build_transport_supervision_entries(attention_aux)
            compute_retrieval_losses = False
        if (
            kernel_size <= 0
            or query_mask_flat is None
            or supervision_entries is None
        ):
            return loss_terms, metrics

        teacher_patch_size = kernel_size + 2 * self.retrieval_teacher_patch_padding
        teacher_tokens = self._extract_patch_tokens(
            refined_target,
            patch_size=teacher_patch_size,
            stride=kernel_size,
            padding=self.retrieval_teacher_patch_padding,
        )
        teacher_tokens = self._normalize_patch_tokens(teacher_tokens)

        retrieval_losses = []
        retrieval_hard_ce_losses = []
        retrieval_top1_margin_losses = []
        retrieval_coherence_losses = []
        retrieval_recall1_exact = []
        retrieval_recall1 = []
        retrieval_recall8 = []
        retrieval_recall32 = []

        for batch_idx, entry in enumerate(supervision_entries):
            query_indices = entry["query_indices"]
            key_indices = entry["key_indices"]
            raw_logits = entry["raw_logits"]
            ranking_scores = entry.get("ranking_scores")
            pred_probs = entry.get("pred_probs")
            pred_log_probs = entry.get("pred_log_probs")
            if raw_logits.numel() == 0 or query_indices.numel() == 0 or key_indices.numel() == 0:
                continue

            raw_logits = raw_logits.float()
            ranking_scores = raw_logits if ranking_scores is None else ranking_scores.float()
            batch_query_mask = query_mask_flat[batch_idx, query_indices] > 0.5
            query_teacher_tokens = teacher_tokens[batch_idx, query_indices]
            key_teacher_tokens = teacher_tokens[batch_idx, key_indices]
            teacher_distances = torch.cdist(
                query_teacher_tokens.unsqueeze(0),
                key_teacher_tokens.unsqueeze(0),
                p=1,
            ).squeeze(0)
            
            # Keep a strict best-match metric, but allow a small tolerance for near-tied targets.
            min_teacher_dist = teacher_distances.min(dim=-1, keepdim=True).values
            exact_targets_mask = teacher_distances <= (min_teacher_dist + 1e-4)
            valid_targets_mask = teacher_distances <= (
                min_teacher_dist * (1.0 + self.retrieval_target_margin_pct) + 1e-4
            )
            
            teacher_logits = -teacher_distances / max(query_teacher_tokens.shape[-1], 1)
            teacher_logits = teacher_logits / self.retrieval_teacher_temperature
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            if pred_probs is None:
                pred_probs = F.softmax(raw_logits, dim=-1)
            else:
                pred_probs = pred_probs.float()
            if pred_log_probs is None:
                pred_log_probs = F.log_softmax(raw_logits, dim=-1)
            else:
                pred_log_probs = pred_log_probs.float()

            if batch_query_mask.any():
                masked_teacher_probs = teacher_probs[batch_query_mask]
                masked_pred_log_probs = pred_log_probs[batch_query_mask]
                masked_raw_logits = ranking_scores[batch_query_mask]
                masked_exact_targets = exact_targets_mask[batch_query_mask]
                masked_valid_targets = valid_targets_mask[batch_query_mask]
                masked_query_indices = query_indices[batch_query_mask]
                masked_query_teacher_tokens = query_teacher_tokens[batch_query_mask]

                masked_pred_probs = pred_probs[batch_query_mask]
                if compute_retrieval_losses:
                    retrieval_losses.append(
                        (-(masked_teacher_probs * masked_pred_log_probs).sum(dim=-1)).mean()
                    )

                    # Multi-target cross entropy: maximize total mass over all near-equivalent targets.
                    valid_probs_sum = (masked_pred_probs * masked_valid_targets.float()).sum(dim=-1).clamp_min(1e-8)
                    retrieval_hard_ce_losses.append(
                        -valid_probs_sum.log().mean()
                    )

                    if self.retrieval_coherence_weight > 0.0:
                        token_hw = entry.get("token_hw") or attention_aux.get("token_hw")
                        if token_hw is not None:
                            retrieval_coherence_losses.append(
                                self._retrieval_coherence_loss(
                                    masked_pred_probs,
                                    masked_query_indices,
                                    key_indices,
                                    masked_query_teacher_tokens,
                                    tuple(token_hw),
                                )
                            )

                    if self.retrieval_top1_margin_weight > 0.0 and masked_raw_logits.shape[-1] > 1:
                        # Align the margin with the tolerant valid-target set so near-equivalent
                        # teacher matches do not keep the top-1 objective artificially active.
                        has_negative = (~masked_valid_targets).any(dim=-1)
                        if has_negative.any():
                            pos_logits = masked_raw_logits.masked_fill(
                                ~masked_valid_targets,
                                torch.finfo(masked_raw_logits.dtype).min,
                            ).max(dim=-1).values
                            neg_logits = masked_raw_logits.masked_fill(
                                masked_valid_targets,
                                torch.finfo(masked_raw_logits.dtype).min,
                            ).max(dim=-1).values
                            per_query_margin = F.relu(neg_logits - pos_logits + self.retrieval_top1_margin)
                            retrieval_top1_margin_losses.append(per_query_margin[has_negative].mean())

                for top_k, tolerant_metric in (
                    (1, retrieval_recall1),
                    (8, retrieval_recall8),
                    (32, retrieval_recall32),
                ):
                    k = min(top_k, masked_raw_logits.shape[-1])
                    topk = masked_raw_logits.topk(k=k, dim=-1).indices
                    tolerant_metric.append(masked_valid_targets.gather(1, topk).any(dim=-1).float().mean())
                    if top_k == 1:
                        retrieval_recall1_exact.append(
                            masked_exact_targets.gather(1, topk).any(dim=-1).float().mean()
                        )

        if retrieval_losses:
            loss_terms["retrieval"] = torch.stack(retrieval_losses).mean()
        if retrieval_hard_ce_losses:
            loss_terms["retrieval_hard_ce"] = torch.stack(retrieval_hard_ce_losses).mean()
        if retrieval_top1_margin_losses:
            loss_terms["retrieval_top1_margin"] = torch.stack(retrieval_top1_margin_losses).mean()
        if retrieval_coherence_losses:
            loss_terms["retrieval_coherence"] = torch.stack(retrieval_coherence_losses).mean()

        if retrieval_recall1_exact:
            metrics[f"{metric_prefix}_recall1_exact"] = torch.stack(retrieval_recall1_exact).mean().item()
        if retrieval_recall1:
            metrics[f"{metric_prefix}_recall1"] = torch.stack(retrieval_recall1).mean().item()
        if retrieval_recall8:
            metrics[f"{metric_prefix}_recall8"] = torch.stack(retrieval_recall8).mean().item()
        if retrieval_recall32:
            metrics[f"{metric_prefix}_recall32"] = torch.stack(retrieval_recall32).mean().item()
        return loss_terms, metrics

    def attention_supervision_metrics(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> dict[str, float]:
        if not self.retrieval_supervision_enabled():
            return {}
        _, metrics = self._attention_supervision_losses(refined_target, attention_aux)
        return metrics

    def transport_selection_metrics(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> dict[str, float]:
        if attention_aux is None or attention_aux.get("copy_mode") != "transport":
            return {}
        transport_only_aux = dict(attention_aux)
        transport_only_aux.pop("attention_supervision_entries", None)
        _, metrics = self._attention_supervision_losses(
            refined_target,
            transport_only_aux,
            metric_prefix="transport_selection",
            allow_metric_only=True,
        )
        return metrics

    def _transport_selection_loss(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = refined_target.new_zeros(())
        metrics = {"transport_selection_loss": 0.0}
        if attention_aux is None or attention_aux.get("copy_mode") != "transport":
            return zero, metrics

        kernel_size = int(attention_aux.get("kernel_size", 0))
        query_mask_flat = attention_aux.get("query_mask_flat")
        supervision_entries = self._build_transport_supervision_entries(attention_aux)
        if (
            kernel_size <= 0
            or query_mask_flat is None
            or supervision_entries is None
        ):
            return zero, metrics

        teacher_patch_size = kernel_size + 2 * self.retrieval_teacher_patch_padding
        teacher_tokens = self._extract_patch_tokens(
            refined_target,
            patch_size=teacher_patch_size,
            stride=kernel_size,
            padding=self.retrieval_teacher_patch_padding,
        )
        teacher_tokens = self._normalize_patch_tokens(teacher_tokens)

        losses = []
        for batch_idx, entry in enumerate(supervision_entries):
            query_indices = entry["query_indices"]
            key_indices = entry["key_indices"]
            raw_logits = entry["raw_logits"]
            if raw_logits.numel() == 0 or query_indices.numel() == 0 or key_indices.numel() == 0:
                continue

            masked_queries = query_mask_flat[batch_idx, query_indices] > 0.5
            if not masked_queries.any():
                continue

            masked_raw_logits = raw_logits[masked_queries].float()
            query_teacher_tokens = teacher_tokens[batch_idx, query_indices[masked_queries]]
            key_teacher_tokens = teacher_tokens[batch_idx, key_indices]
            teacher_distances = torch.cdist(
                query_teacher_tokens.unsqueeze(0),
                key_teacher_tokens.unsqueeze(0),
                p=1,
            ).squeeze(0)
            min_teacher_dist = teacher_distances.min(dim=-1, keepdim=True).values
            valid_targets_mask = teacher_distances <= (
                min_teacher_dist * (1.0 + self.retrieval_target_margin_pct) + 1e-4
            )

            pred_probs = F.softmax(
                masked_raw_logits / self.transport_selection_temperature,
                dim=-1,
            )
            valid_probs_sum = (pred_probs * valid_targets_mask.float()).sum(dim=-1).clamp_min(1e-8)
            losses.append(-valid_probs_sum.log().mean())

        if not losses:
            return zero, metrics

        loss = torch.stack(losses).mean()
        metrics["transport_selection_loss"] = loss.item()
        return loss, metrics

    def inpainter_loss(
        self,
        coarse_raw: torch.Tensor,
        refined: torch.Tensor,
        coarse_target: torch.Tensor,
        refined_target: torch.Tensor,
        mask: torch.Tensor,
        attention_aux: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        coarse_l2 = masked_mse_loss(coarse_raw, coarse_target, mask)
        refined_l1 = masked_l1_loss(refined, refined_target, mask)
        refined_query_patch_l1 = self._query_patch_l1_loss(refined, refined_target, attention_aux)
        perceptual = self.perceptual_loss(refined, refined_target)

        scheduled_weights = {
            name: self._get_scheduled_weight(name)
            for name in self.SCHEDULED_WEIGHT_NAMES
        }
        zero = refined_target.new_zeros(())
        total = (
            self.coarse_l2_weight * coarse_l2
            + self.refined_l1_weight * refined_l1
            + self.refined_query_patch_l1_weight * refined_query_patch_l1
            + scheduled_weights["perceptual_weight"] * perceptual
        )
        loss_dict = {
            "coarse_l2": coarse_l2.item(),
            "refined_l1": refined_l1.item(),
            "refined_query_patch_l1": refined_query_patch_l1.item(),
            "perceptual": perceptual.item(),
        }
        if self.retrieval_supervision_enabled():
            attention_loss_terms, attention_metrics = self._attention_supervision_losses(refined_target, attention_aux)
            total = (
                total
                + scheduled_weights["retrieval_loss_weight"] * attention_loss_terms.get("retrieval", zero)
                + self.retrieval_hard_ce_weight * attention_loss_terms.get("retrieval_hard_ce", zero)
                + self.retrieval_top1_margin_weight * attention_loss_terms.get("retrieval_top1_margin", zero)
                + self.retrieval_coherence_weight * attention_loss_terms.get("retrieval_coherence", zero)
            )
            if "retrieval" in attention_loss_terms:
                loss_dict["retrieval_loss"] = attention_loss_terms["retrieval"].item()
            if "retrieval_hard_ce" in attention_loss_terms:
                loss_dict["retrieval_hard_ce_loss"] = attention_loss_terms["retrieval_hard_ce"].item()
            if "retrieval_top1_margin" in attention_loss_terms:
                loss_dict["retrieval_top1_margin_loss"] = attention_loss_terms["retrieval_top1_margin"].item()
            if "retrieval_coherence" in attention_loss_terms:
                loss_dict["retrieval_coherence_loss"] = attention_loss_terms["retrieval_coherence"].item()
            loss_dict["weight/retrieval_loss"] = scheduled_weights["retrieval_loss_weight"]
            loss_dict.update(attention_metrics)

        loss_dict.update(self.transport_selection_metrics(refined_target, attention_aux))

        if self.transport_patch_supervision_enabled():
            transport_patch_loss, transport_patch_metrics = self._transport_patch_loss(refined_target, attention_aux)
            total = total + self.transport_patch_weight * transport_patch_loss
            loss_dict["transport_patch"] = transport_patch_metrics["transport_patch"]
        if self.transport_selection_supervision_enabled():
            transport_selection_loss, transport_selection_metrics = self._transport_selection_loss(
                refined_target,
                attention_aux,
            )
            total = total + self.transport_selection_weight * transport_selection_loss
            loss_dict["transport_selection_loss"] = transport_selection_metrics["transport_selection_loss"]
        if self.transport_validity_supervision_enabled():
            transport_validity_loss, transport_validity_metrics = self._transport_validity_loss(refined_target, attention_aux)
            total = total + self.transport_validity_weight * transport_validity_loss
            loss_dict["transport_validity"] = transport_validity_metrics["transport_validity"]
            loss_dict["transport_valid_ratio"] = transport_validity_metrics["transport_valid_ratio"]
            loss_dict["transport_fallback_ratio"] = transport_validity_metrics["transport_fallback_ratio"]
        if self.transport_smoothness_supervision_enabled():
            transport_offset_smoothness_loss, transport_offset_smoothness_metrics = self._transport_offset_smoothness_loss(
                refined_target,
                attention_aux,
            )
            total = total + self.transport_offset_smoothness_weight * transport_offset_smoothness_loss
            loss_dict["transport_offset_smoothness"] = transport_offset_smoothness_metrics["transport_offset_smoothness"]
            loss_dict["transport_offset_curvature"] = transport_offset_smoothness_metrics["transport_offset_curvature"]

        loss_dict["weight/perceptual"] = scheduled_weights["perceptual_weight"]
        loss_dict["inpainter_total"] = total.item()
        return total, loss_dict
