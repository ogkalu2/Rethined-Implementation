from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.perceptual import PerceptualLoss


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


class InpaintingLoss(nn.Module):
    """Training losses for refinement and retrieval supervision."""

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
        retrieval_teacher_patch_padding: int = 8,
        retrieval_teacher_temperature: float = 0.07,
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
        self.retrieval_teacher_patch_padding = max(0, int(retrieval_teacher_patch_padding))
        self.retrieval_teacher_temperature = float(retrieval_teacher_temperature)
        self.loss_schedule_focus_steps = max(0, int(loss_schedule_focus_steps))
        self.loss_schedule_transition_steps = max(0, int(loss_schedule_transition_steps))
        self.perceptual_weight = float(perceptual_weight)
        if self.retrieval_teacher_temperature <= 0:
            raise ValueError("retrieval_teacher_temperature must be positive.")

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

    def _attention_supervision_losses(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        zero = refined_target.new_zeros(())
        loss_terms = {
            "retrieval": zero,
            "retrieval_hard_ce": zero,
        }
        metrics = {
            "retrieval_recall1": 0.0,
            "retrieval_recall8": 0.0,
            "retrieval_recall32": 0.0,
        }
        if attention_aux is None:
            return loss_terms, metrics

        kernel_size = int(attention_aux.get("kernel_size", 0))
        query_mask_flat = attention_aux.get("query_mask_flat")
        supervision_entries = attention_aux.get("attention_supervision_entries")
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
        retrieval_recall1 = []
        retrieval_recall8 = []
        retrieval_recall32 = []

        for batch_idx, entry in enumerate(supervision_entries):
            query_indices = entry["query_indices"]
            key_indices = entry["key_indices"]
            raw_logits = entry["raw_logits"]
            if raw_logits.numel() == 0 or query_indices.numel() == 0 or key_indices.numel() == 0:
                continue

            raw_logits = raw_logits.float()
            batch_query_mask = query_mask_flat[batch_idx, query_indices] > 0.5
            query_teacher_tokens = teacher_tokens[batch_idx, query_indices]
            key_teacher_tokens = teacher_tokens[batch_idx, key_indices]
            teacher_distances = torch.cdist(
                query_teacher_tokens.unsqueeze(0),
                key_teacher_tokens.unsqueeze(0),
                p=1,
            ).squeeze(0)
            
            # Find all target patches that are essentially identical to the absolute best patch
            # We consider any patch with an L1 distance within 5% of the minimum distance to be a valid target
            min_teacher_dist = teacher_distances.min(dim=-1, keepdim=True).values
            valid_targets_mask = teacher_distances <= (min_teacher_dist * 1.05 + 1e-4)
            
            teacher_logits = -teacher_distances / max(query_teacher_tokens.shape[-1], 1)
            teacher_logits = teacher_logits / self.retrieval_teacher_temperature
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            pred_log_probs = F.log_softmax(raw_logits, dim=-1)

            if batch_query_mask.any():
                masked_teacher_probs = teacher_probs[batch_query_mask]
                masked_pred_log_probs = pred_log_probs[batch_query_mask]
                masked_raw_logits = raw_logits[batch_query_mask]
                masked_valid_targets = valid_targets_mask[batch_query_mask]

                retrieval_losses.append(
                    (-(masked_teacher_probs * masked_pred_log_probs).sum(dim=-1)).mean()
                )
                
                # Multi-target Cross Entropy: maximize the sum of probabilities of all identical valid patches
                masked_pred_probs = F.softmax(masked_raw_logits, dim=-1)
                valid_probs_sum = (masked_pred_probs * masked_valid_targets.float()).sum(dim=-1).clamp_min(1e-8)
                retrieval_hard_ce_losses.append(
                    -valid_probs_sum.log().mean()
                )
                
                for top_k, metric_name in ((1, retrieval_recall1), (8, retrieval_recall8), (32, retrieval_recall32)):
                    k = min(top_k, masked_raw_logits.shape[-1])
                    topk = masked_raw_logits.topk(k=k, dim=-1).indices
                    is_correct = masked_valid_targets.gather(1, topk).any(dim=-1)
                    metric_name.append(
                        is_correct.float().mean()
                    )

        if retrieval_losses:
            loss_terms["retrieval"] = torch.stack(retrieval_losses).mean()
        if retrieval_hard_ce_losses:
            loss_terms["retrieval_hard_ce"] = torch.stack(retrieval_hard_ce_losses).mean()

        if retrieval_recall1:
            metrics["retrieval_recall1"] = torch.stack(retrieval_recall1).mean().item()
        if retrieval_recall8:
            metrics["retrieval_recall8"] = torch.stack(retrieval_recall8).mean().item()
        if retrieval_recall32:
            metrics["retrieval_recall32"] = torch.stack(retrieval_recall32).mean().item()
        return loss_terms, metrics

    def attention_supervision_metrics(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> dict[str, float]:
        _, metrics = self._attention_supervision_losses(refined_target, attention_aux)
        return metrics

    def generator_loss(
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
        attention_loss_terms, attention_metrics = self._attention_supervision_losses(refined_target, attention_aux)
        perceptual = self.perceptual_loss(refined, refined_target)

        scheduled_weights = {
            name: self._get_scheduled_weight(name)
            for name in self.SCHEDULED_WEIGHT_NAMES
        }
        total = (
            self.coarse_l2_weight * coarse_l2
            + self.refined_l1_weight * refined_l1
            + self.refined_query_patch_l1_weight * refined_query_patch_l1
            + scheduled_weights["retrieval_loss_weight"] * attention_loss_terms["retrieval"]
            + self.retrieval_hard_ce_weight * attention_loss_terms["retrieval_hard_ce"]
            + scheduled_weights["perceptual_weight"] * perceptual
        )
        loss_dict = {
            "coarse_l2": coarse_l2.item(),
            "refined_l1": refined_l1.item(),
            "refined_query_patch_l1": refined_query_patch_l1.item(),
            "retrieval_loss": attention_loss_terms["retrieval"].item(),
            "retrieval_hard_ce_loss": attention_loss_terms["retrieval_hard_ce"].item(),
            "perceptual": perceptual.item(),
            "generator_total": total.item(),
        }
        loss_dict.update({
            f"weight/{name.removesuffix('_weight')}": value
            for name, value in scheduled_weights.items()
        })
        loss_dict.update(attention_metrics)
        return total, loss_dict
