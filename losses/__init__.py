"""Paper-aligned losses for RETHINED training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import NativeGaussianBlur2d
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


class FocalFrequencyLoss(nn.Module):
    """Compact Focal Frequency Loss implementation for image restoration."""

    def __init__(self, alpha: float = 1.0, log_matrix: bool = False, eps: float = 1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.log_matrix = bool(log_matrix)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_freq = torch.fft.fft2(pred.float(), norm="ortho")
        target_freq = torch.fft.fft2(target.float(), norm="ortho")
        distance = (pred_freq.real - target_freq.real).pow(2) + (pred_freq.imag - target_freq.imag).pow(2)

        weight = distance.detach().clamp_min(self.eps)
        if self.log_matrix:
            weight = torch.log1p(weight)
        weight = weight.pow(self.alpha)
        weight = weight / weight.mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        return (weight * distance).mean()


class InpaintingLoss(nn.Module):
    """Paper-aligned generator and discriminator losses."""

    def __init__(
        self,
        coarse_l2_weight: float = 1.0,
        coarse_blur_l1_weight: float = 0.0,
        coarse_gradient_weight: float = 0.0,
        coarse_perceptual_weight: float = 0.05,
        refined_l1_weight: float = 1.0,
        refined_query_patch_l1_weight: float = 0.0,
        patch_teacher_weight: float = 0.0,
        patch_teacher_temperature: float = 0.1,
        patch_teacher_confidence_threshold: float = 0.2,
        patch_teacher_margin_threshold: float = 0.03,
        candidate_rerank_weight: float = 0.0,
        candidate_rerank_temperature: float = 1.0,
        candidate_rerank_confidence_threshold: float = 0.2,
        candidate_rerank_margin_threshold: float = 0.03,
        frequency_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        adversarial_weight: float = 0.02,
        adversarial_mode: str = "hinge",
        focal_alpha: float = 1.0,
        focal_log_matrix: bool = False,
    ):
        super().__init__()
        self.coarse_l2_weight = float(coarse_l2_weight)
        self.coarse_blur_l1_weight = float(coarse_blur_l1_weight)
        self.coarse_gradient_weight = float(coarse_gradient_weight)
        self.coarse_perceptual_weight = float(coarse_perceptual_weight)
        self.refined_l1_weight = float(refined_l1_weight)
        self.refined_query_patch_l1_weight = float(refined_query_patch_l1_weight)
        self.patch_teacher_weight = float(patch_teacher_weight)
        self.patch_teacher_temperature = float(patch_teacher_temperature)
        self.patch_teacher_confidence_threshold = float(patch_teacher_confidence_threshold)
        self.patch_teacher_margin_threshold = float(patch_teacher_margin_threshold)
        self.candidate_rerank_weight = float(candidate_rerank_weight)
        self.candidate_rerank_temperature = float(candidate_rerank_temperature)
        self.candidate_rerank_confidence_threshold = float(candidate_rerank_confidence_threshold)
        self.candidate_rerank_margin_threshold = float(candidate_rerank_margin_threshold)
        self.frequency_weight = float(frequency_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.adversarial_weight = float(adversarial_weight)
        self.adversarial_mode = str(adversarial_mode).lower()
        if self.adversarial_mode not in {"bce", "hinge"}:
            raise ValueError(
                f"Unsupported adversarial_mode: {adversarial_mode}. Expected 'bce' or 'hinge'."
            )

        self.frequency_loss = FocalFrequencyLoss(alpha=focal_alpha, log_matrix=focal_log_matrix)
        self.perceptual_loss = PerceptualLoss()
        self.coarse_blur = NativeGaussianBlur2d((7, 7), sigma=(2.01, 2.01))
        if self.patch_teacher_temperature <= 0:
            raise ValueError("patch_teacher_temperature must be positive.")
        if self.patch_teacher_weight < 0:
            raise ValueError("patch_teacher_weight must be non-negative.")
        if self.candidate_rerank_temperature <= 0:
            raise ValueError("candidate_rerank_temperature must be positive.")
        if self.candidate_rerank_weight < 0:
            raise ValueError("candidate_rerank_weight must be non-negative.")

    def _normalize_descriptors(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.float()
        tokens = tokens - tokens.mean(dim=-1, keepdim=True)
        return F.normalize(tokens, dim=-1, eps=1e-6)

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

    def _patch_teacher_loss(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = refined_target.new_zeros(())
        metrics = {
            "patch_teacher": 0.0,
            "patch_teacher_accuracy": 0.0,
            "patch_teacher_supervised_ratio": 0.0,
        }
        if self.patch_teacher_weight <= 0 or attention_aux is None:
            return zero, metrics

        query_mask_flat = attention_aux.get("query_mask_flat")
        key_valid_flat = attention_aux.get("key_valid_flat")
        query_matching_tokens = attention_aux.get("query_matching_tokens")
        key_matching_tokens = attention_aux.get("key_matching_tokens")
        matching_tokens = attention_aux.get("matching_tokens")
        kernel_size = int(attention_aux.get("kernel_size", 0))
        value_patch_size = int(attention_aux.get("value_patch_size", 0))
        value_patch_padding = int(attention_aux.get("value_patch_padding", 0))
        if query_matching_tokens is None:
            query_matching_tokens = matching_tokens
        if key_matching_tokens is None:
            key_matching_tokens = matching_tokens
        if (
            query_mask_flat is None
            or key_valid_flat is None
            or query_matching_tokens is None
            or key_matching_tokens is None
            or kernel_size <= 0
            or value_patch_size <= 0
        ):
            return zero, metrics

        target_patches = self._extract_patch_tokens(
            refined_target,
            patch_size=value_patch_size,
            stride=kernel_size,
            padding=value_patch_padding,
        )
        teacher_patch_tokens = self._normalize_descriptors(target_patches.detach())
        descriptor_query_tokens = self._normalize_descriptors(query_matching_tokens)
        descriptor_key_tokens = self._normalize_descriptors(key_matching_tokens)

        loss_terms = []
        supervised_queries = 0
        total_masked_queries = 0
        correct_matches = 0

        for batch_idx in range(descriptor_query_tokens.shape[0]):
            query_indices = (query_mask_flat[batch_idx] > 0.5).nonzero(as_tuple=False).flatten()
            key_indices = (key_valid_flat[batch_idx] > 0.5).nonzero(as_tuple=False).flatten()
            total_masked_queries += int(query_indices.numel())
            if query_indices.numel() == 0 or key_indices.numel() == 0:
                continue

            with torch.no_grad():
                query_patch_tokens = teacher_patch_tokens[batch_idx, query_indices]
                key_patch_tokens = teacher_patch_tokens[batch_idx, key_indices]
                teacher_scores = torch.matmul(query_patch_tokens, key_patch_tokens.transpose(0, 1))
                if key_indices.numel() > 1:
                    top2 = teacher_scores.topk(k=2, dim=-1)
                    teacher_best = top2.values[:, 0]
                    teacher_margin = top2.values[:, 0] - top2.values[:, 1]
                else:
                    top2 = teacher_scores.topk(k=1, dim=-1)
                    teacher_best = top2.values[:, 0]
                    teacher_margin = torch.full_like(teacher_best, float("inf"))
                valid_teacher = (
                    (teacher_best >= self.patch_teacher_confidence_threshold)
                    & (teacher_margin >= self.patch_teacher_margin_threshold)
                )
                if not valid_teacher.any():
                    continue
                teacher_labels = top2.indices[:, 0][valid_teacher]

            query_tokens = descriptor_query_tokens[batch_idx, query_indices[valid_teacher]]
            key_tokens = descriptor_key_tokens[batch_idx, key_indices]
            logits = torch.matmul(query_tokens, key_tokens.transpose(0, 1))
            logits = logits / self.patch_teacher_temperature
            loss_terms.append(F.cross_entropy(logits, teacher_labels))
            predictions = logits.argmax(dim=-1)
            supervised_queries += int(valid_teacher.sum().item())
            correct_matches += int((predictions == teacher_labels).sum().item())

        if not loss_terms:
            return zero, metrics

        loss = torch.stack(loss_terms).mean()
        metrics["patch_teacher"] = loss.item()
        metrics["patch_teacher_accuracy"] = correct_matches / max(supervised_queries, 1)
        metrics["patch_teacher_supervised_ratio"] = supervised_queries / max(total_masked_queries, 1)
        return loss, metrics

    def _candidate_rerank_loss(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = refined_target.new_zeros(())
        metrics = {
            "candidate_rerank": 0.0,
            "candidate_rerank_accuracy": 0.0,
            "candidate_rerank_supervised_ratio": 0.0,
        }
        if self.candidate_rerank_weight <= 0 or attention_aux is None:
            return zero, metrics

        query_mask_flat = attention_aux.get("query_mask_flat")
        candidate_indices = attention_aux.get("candidate_indices")
        candidate_logits = attention_aux.get("candidate_logits")
        candidate_valid_mask = attention_aux.get("candidate_valid_mask")
        kernel_size = int(attention_aux.get("kernel_size", 0))
        value_patch_size = int(attention_aux.get("value_patch_size", 0))
        value_patch_padding = int(attention_aux.get("value_patch_padding", 0))
        if (
            query_mask_flat is None
            or candidate_indices is None
            or candidate_logits is None
            or candidate_valid_mask is None
            or kernel_size <= 0
            or value_patch_size <= 0
        ):
            return zero, metrics

        target_patches = self._extract_patch_tokens(
            refined_target,
            patch_size=value_patch_size,
            stride=kernel_size,
            padding=value_patch_padding,
        )
        teacher_patch_tokens = self._normalize_descriptors(target_patches.detach())

        loss_terms = []
        supervised_queries = 0
        total_masked_queries = 0
        correct_matches = 0

        for batch_idx in range(teacher_patch_tokens.shape[0]):
            query_indices = (query_mask_flat[batch_idx] > 0.5).nonzero(as_tuple=False).flatten()
            total_masked_queries += int(query_indices.numel())
            if query_indices.numel() == 0:
                continue

            batch_candidate_indices = candidate_indices[batch_idx, query_indices]
            batch_candidate_logits = candidate_logits[batch_idx, query_indices]
            batch_candidate_valid = candidate_valid_mask[batch_idx, query_indices]
            if not batch_candidate_valid.any():
                continue

            safe_candidate_indices = batch_candidate_indices.clamp_min(0)
            query_patch_tokens = teacher_patch_tokens[batch_idx, query_indices]
            candidate_patch_tokens = teacher_patch_tokens[batch_idx, safe_candidate_indices]
            teacher_scores = (query_patch_tokens.unsqueeze(1) * candidate_patch_tokens).sum(dim=-1)
            teacher_scores = teacher_scores.masked_fill(
                ~batch_candidate_valid,
                torch.finfo(teacher_scores.dtype).min,
            )

            valid_counts = batch_candidate_valid.sum(dim=-1)
            teacher_best = teacher_scores.max(dim=-1).values
            teacher_top2 = teacher_scores.topk(k=min(2, teacher_scores.shape[-1]), dim=-1).values
            teacher_margin = torch.full_like(teacher_best, float("inf"))
            if teacher_top2.shape[-1] > 1:
                teacher_margin = teacher_top2[:, 0] - teacher_top2[:, 1]
            valid_teacher = (
                (valid_counts > 0)
                & (teacher_best >= self.candidate_rerank_confidence_threshold)
                & (teacher_margin >= self.candidate_rerank_margin_threshold)
            )
            if not valid_teacher.any():
                continue

            logits = batch_candidate_logits[valid_teacher] / self.candidate_rerank_temperature
            valid_mask = batch_candidate_valid[valid_teacher]
            logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
            teacher_labels = teacher_scores[valid_teacher].argmax(dim=-1)
            loss_terms.append(F.cross_entropy(logits, teacher_labels))
            predictions = logits.argmax(dim=-1)
            supervised_queries += int(valid_teacher.sum().item())
            correct_matches += int((predictions == teacher_labels).sum().item())

        if not loss_terms:
            return zero, metrics

        loss = torch.stack(loss_terms).mean()
        metrics["candidate_rerank"] = loss.item()
        metrics["candidate_rerank_accuracy"] = correct_matches / max(supervised_queries, 1)
        metrics["candidate_rerank_supervised_ratio"] = supervised_queries / max(total_masked_queries, 1)
        return loss, metrics

    def generator_loss(
        self,
        coarse_raw: torch.Tensor,
        refined: torch.Tensor,
        coarse_target: torch.Tensor,
        refined_target: torch.Tensor,
        mask: torch.Tensor,
        fake_logits: list[torch.Tensor] | None = None,
        attention_aux: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        coarse_composite = composite_with_known(coarse_raw, coarse_target, mask)
        coarse_blurred = self.coarse_blur(coarse_composite)
        target_blurred = self.coarse_blur(coarse_target)
        coarse_grad_mask = dilate_mask(mask)
        coarse_l2 = masked_mse_loss(coarse_raw, coarse_target, mask)
        coarse_blur_l1 = masked_l1_loss(coarse_blurred, target_blurred, mask)
        coarse_gradient = masked_gradient_l1_loss(coarse_blurred, target_blurred, coarse_grad_mask)
        coarse_perceptual = self.perceptual_loss(coarse_composite, coarse_target)
        refined_l1 = masked_l1_loss(refined, refined_target, mask)
        refined_query_patch_l1 = self._query_patch_l1_loss(refined, refined_target, attention_aux)
        patch_teacher_loss, patch_teacher_metrics = self._patch_teacher_loss(refined_target, attention_aux)
        candidate_rerank_loss, candidate_rerank_metrics = self._candidate_rerank_loss(
            refined_target,
            attention_aux,
        )
        frequency = self.frequency_loss(refined, refined_target)
        perceptual = self.perceptual_loss(refined, refined_target)

        adversarial = refined.new_zeros(())
        if fake_logits is not None:
            if self.adversarial_mode == "hinge":
                scale_losses = [(-l).mean() for l in fake_logits]
            else:
                scale_losses = [
                    F.binary_cross_entropy_with_logits(l, torch.ones_like(l))
                    for l in fake_logits
                ]
            adversarial = torch.stack(scale_losses).mean()

        total = (
            self.coarse_l2_weight * coarse_l2
            + self.coarse_blur_l1_weight * coarse_blur_l1
            + self.coarse_gradient_weight * coarse_gradient
            + self.coarse_perceptual_weight * coarse_perceptual
            + self.refined_l1_weight * refined_l1
            + self.refined_query_patch_l1_weight * refined_query_patch_l1
            + self.patch_teacher_weight * patch_teacher_loss
            + self.candidate_rerank_weight * candidate_rerank_loss
            + self.frequency_weight * frequency
            + self.perceptual_weight * perceptual
            + self.adversarial_weight * adversarial
        )
        loss_dict = {
            "coarse_l2": coarse_l2.item(),
            "coarse_blur_l1": coarse_blur_l1.item(),
            "coarse_gradient": coarse_gradient.item(),
            "coarse_perceptual": coarse_perceptual.item(),
            "refined_l1": refined_l1.item(),
            "refined_query_patch_l1": refined_query_patch_l1.item(),
            "patch_teacher": patch_teacher_metrics["patch_teacher"],
            "patch_teacher_accuracy": patch_teacher_metrics["patch_teacher_accuracy"],
            "patch_teacher_supervised_ratio": patch_teacher_metrics["patch_teacher_supervised_ratio"],
            "candidate_rerank": candidate_rerank_metrics["candidate_rerank"],
            "candidate_rerank_accuracy": candidate_rerank_metrics["candidate_rerank_accuracy"],
            "candidate_rerank_supervised_ratio": candidate_rerank_metrics["candidate_rerank_supervised_ratio"],
            "frequency": frequency.item(),
            "perceptual": perceptual.item(),
            "adversarial_g": adversarial.item(),
            "generator_total": total.item(),
        }
        return total, loss_dict

    def discriminator_loss(
        self,
        real_logits: list[torch.Tensor],
        fake_logits: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.adversarial_mode == "hinge":
            real_loss = torch.stack([
                F.relu(1 - l).mean()
                for l in real_logits
            ]).mean()
            fake_loss = torch.stack([
                F.relu(1 + l).mean()
                for l in fake_logits
            ]).mean()
        else:
            real_loss = torch.stack([
                F.binary_cross_entropy_with_logits(l, torch.ones_like(l))
                for l in real_logits
            ]).mean()
            fake_loss = torch.stack([
                F.binary_cross_entropy_with_logits(l, torch.zeros_like(l))
                for l in fake_logits
            ]).mean()
        total = 0.5 * (real_loss + fake_loss)
        loss_dict = {
            "adversarial_d_real": real_loss.item(),
            "adversarial_d_fake": fake_loss.item(),
            "discriminator_total": total.item(),
        }
        return total, loss_dict
