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
        frequency_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        adversarial_weight: float = 0.02,
        attention_supervision_weight: float = 0.0,
        attention_supervision_temperature: float = 0.15,
        attention_supervision_hard_targets: bool = False,
        attention_supervision_confidence_threshold: float = 0.0,
        attention_supervision_margin_threshold: float = 0.0,
        attention_supervision_start_step: int = 0,
        attention_supervision_full_step: int = 0,
        attention_coherence_weight: float = 0.0,
        attention_coherence_similarity_threshold: float = 0.0,
        attention_coherence_start_step: int = 0,
        attention_coherence_full_step: int = 0,
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
        self.frequency_weight = float(frequency_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.adversarial_weight = float(adversarial_weight)
        self.attention_supervision_weight = float(attention_supervision_weight)
        self.attention_supervision_temperature = float(attention_supervision_temperature)
        if self.attention_supervision_temperature <= 0:
            raise ValueError("attention_supervision_temperature must be positive.")
        self.attention_supervision_hard_targets = bool(attention_supervision_hard_targets)
        self.attention_supervision_confidence_threshold = float(attention_supervision_confidence_threshold)
        self.attention_supervision_margin_threshold = float(attention_supervision_margin_threshold)
        self.attention_supervision_start_step = int(attention_supervision_start_step)
        self.attention_supervision_full_step = int(attention_supervision_full_step)
        self.attention_coherence_weight = float(attention_coherence_weight)
        self.attention_coherence_similarity_threshold = float(attention_coherence_similarity_threshold)
        self.attention_coherence_start_step = int(attention_coherence_start_step)
        self.attention_coherence_full_step = int(attention_coherence_full_step)
        self.adversarial_mode = str(adversarial_mode).lower()
        if self.adversarial_mode not in {"bce", "hinge"}:
            raise ValueError(
                f"Unsupported adversarial_mode: {adversarial_mode}. Expected 'bce' or 'hinge'."
            )
        for name, value in (
            ("attention_supervision_confidence_threshold", self.attention_supervision_confidence_threshold),
            ("attention_supervision_margin_threshold", self.attention_supervision_margin_threshold),
            ("attention_coherence_similarity_threshold", self.attention_coherence_similarity_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")

        self.frequency_loss = FocalFrequencyLoss(alpha=focal_alpha, log_matrix=focal_log_matrix)
        self.perceptual_loss = PerceptualLoss()
        self.coarse_blur = NativeGaussianBlur2d((7, 7), sigma=(2.01, 2.01))

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

    def _normalize_patch_tokens(self, patches: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        centered = patches - patches.mean(dim=-1, keepdim=True)
        scaled = centered / centered.std(dim=-1, keepdim=True).clamp_min(eps)
        return F.normalize(scaled, dim=-1, eps=eps)

    def _coords_from_indices(self, indices: torch.Tensor, grid_size: int) -> torch.Tensor:
        return torch.stack((indices // grid_size, indices % grid_size), dim=-1).to(dtype=torch.float32)

    def _teacher_patch_descriptors(
        self,
        refined_target: torch.Tensor,
        *,
        patch_size: int,
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        rgb_patches = self._extract_patch_tokens(
            refined_target,
            patch_size=patch_size,
            stride=stride,
            padding=padding,
        )
        high_freq = refined_target - self.coarse_blur(refined_target)
        high_freq_patches = self._extract_patch_tokens(
            high_freq,
            patch_size=patch_size,
            stride=stride,
            padding=padding,
        )
        rgb_desc = self._normalize_patch_tokens(rgb_patches)
        high_freq_desc = self._normalize_patch_tokens(high_freq_patches)
        return F.normalize(torch.cat([rgb_desc, high_freq_desc], dim=-1), dim=-1)

    def _scheduled_weight(
        self,
        base_weight: float,
        step: int | None,
        *,
        start_step: int,
        full_step: int,
    ) -> float:
        if base_weight == 0.0:
            return 0.0
        if step is None:
            return base_weight

        start_step = max(0, int(start_step))
        full_step = max(0, int(full_step))
        if full_step <= start_step:
            return base_weight if step >= start_step else 0.0
        if step <= start_step:
            return 0.0
        if step >= full_step:
            return base_weight
        scale = float(step - start_step) / float(full_step - start_step)
        return base_weight * scale

    def _confidence_mask(self, teacher_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if teacher_probs.numel() == 0:
            empty = teacher_probs.new_zeros((teacher_probs.shape[0],), dtype=torch.bool)
            return empty, teacher_probs.new_zeros((teacher_probs.shape[0],))

        confidence, _ = teacher_probs.max(dim=-1)
        keep_mask = confidence >= self.attention_supervision_confidence_threshold
        if teacher_probs.shape[-1] > 1 and self.attention_supervision_margin_threshold > 0:
            top2 = teacher_probs.topk(k=2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            keep_mask = keep_mask & (margin >= self.attention_supervision_margin_threshold)
        return keep_mask, confidence

    def _weighted_neighbor_loss(
        self,
        offsets: torch.Tensor,
        query_desc: torch.Tensor,
        index_a: torch.Tensor,
        index_b: torch.Tensor,
    ) -> torch.Tensor | None:
        valid = (index_a >= 0) & (index_b >= 0)
        if not valid.any():
            return None

        idx_a = index_a[valid]
        idx_b = index_b[valid]
        pair_similarity = ((query_desc[idx_a] * query_desc[idx_b]).sum(dim=-1) + 1.0) * 0.5
        weights = (
            (pair_similarity - self.attention_coherence_similarity_threshold)
            / max(1.0 - self.attention_coherence_similarity_threshold, 1e-6)
        ).clamp_(0.0, 1.0)
        if weights.sum().item() <= 0:
            return None

        pair_offset_delta = (offsets[idx_a] - offsets[idx_b]).abs().mean(dim=-1)
        return (pair_offset_delta * weights).sum() / weights.sum().clamp_min(1e-6)

    def _attention_auxiliary_losses(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
        *,
        step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        zero = refined_target.new_zeros(())
        metrics = {
            "attention_supervision": 0.0,
            "attention_supervision_coverage": 0.0,
            "attention_candidate_match": 0.0,
            "attention_offset_coherence": 0.0,
            "attention_supervision_weight": self._scheduled_weight(
                self.attention_supervision_weight,
                step,
                start_step=self.attention_supervision_start_step,
                full_step=self.attention_supervision_full_step,
            ),
            "attention_coherence_weight": self._scheduled_weight(
                self.attention_coherence_weight,
                step,
                start_step=self.attention_coherence_start_step,
                full_step=self.attention_coherence_full_step,
            ),
        }
        if attention_aux is None:
            return zero, zero, metrics

        matching_candidates = attention_aux.get("matching_candidates")
        if not matching_candidates:
            return zero, zero, metrics

        descriptor_bank = self._teacher_patch_descriptors(
            refined_target,
            patch_size=int(attention_aux["value_patch_size"]),
            stride=int(attention_aux["kernel_size"]),
            padding=int(attention_aux["value_patch_padding"]),
        )
        grid_size = int(attention_aux["token_grid_size"])

        supervision_losses = []
        supervision_coverages = []
        match_accuracies = []
        coherence_losses = []

        for batch_idx, batch_detail in enumerate(matching_candidates):
            query_indices = batch_detail["query_indices"]
            candidate_key_indices = batch_detail["candidate_key_indices"]
            candidate_logits = batch_detail["candidate_logits"]
            if query_indices.numel() == 0 or candidate_key_indices.numel() == 0:
                continue

            descriptor_tokens = descriptor_bank[batch_idx]
            query_desc = descriptor_tokens.index_select(0, query_indices)
            candidate_desc = descriptor_tokens.index_select(
                0,
                candidate_key_indices.reshape(-1),
            ).view(candidate_key_indices.shape[0], candidate_key_indices.shape[1], -1)
            patch_similarity = (query_desc.unsqueeze(1) * candidate_desc).sum(dim=-1)
            teacher_probs = F.softmax(
                patch_similarity / self.attention_supervision_temperature,
                dim=-1,
            ).detach()
            confident_mask, teacher_confidence = self._confidence_mask(teacher_probs)
            supervision_coverages.append(confident_mask.to(dtype=torch.float32).mean())
            target_best = teacher_probs.argmax(dim=-1)
            if confident_mask.any():
                student_logits = candidate_logits[confident_mask].float()
                sample_weights = teacher_confidence[confident_mask].detach()
                if self.attention_supervision_hard_targets:
                    per_query_loss = F.cross_entropy(
                        student_logits,
                        target_best[confident_mask],
                        reduction="none",
                    )
                else:
                    log_probs = F.log_softmax(student_logits, dim=-1)
                    per_query_loss = F.kl_div(
                        log_probs,
                        teacher_probs[confident_mask],
                        reduction="none",
                    ).sum(dim=-1)
                supervision_losses.append(
                    (per_query_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
                )

                predicted_best = student_logits.argmax(dim=-1)
                match_accuracies.append(
                    (predicted_best == target_best[confident_mask]).to(dtype=torch.float32).mean()
                )

            probs = F.softmax(candidate_logits.float(), dim=-1)
            query_coords = self._coords_from_indices(query_indices, grid_size).to(device=refined_target.device)
            candidate_coords = self._coords_from_indices(
                candidate_key_indices.reshape(-1),
                grid_size,
            ).to(device=refined_target.device).view(candidate_key_indices.shape[0], candidate_key_indices.shape[1], 2)
            expected_coords = (probs.unsqueeze(-1) * candidate_coords).sum(dim=1)
            offsets = expected_coords - query_coords

            query_coords_int = query_coords.to(dtype=torch.long)
            index_map = torch.full(
                (grid_size, grid_size),
                -1,
                device=refined_target.device,
                dtype=torch.long,
            )
            index_map[query_coords_int[:, 0], query_coords_int[:, 1]] = torch.arange(
                query_indices.numel(),
                device=refined_target.device,
            )

            pair_losses = []
            horizontal_a = index_map[:, :-1].reshape(-1)
            horizontal_b = index_map[:, 1:].reshape(-1)
            horizontal_loss = self._weighted_neighbor_loss(offsets, query_desc, horizontal_a, horizontal_b)
            if horizontal_loss is not None:
                pair_losses.append(horizontal_loss)

            vertical_a = index_map[:-1, :].reshape(-1)
            vertical_b = index_map[1:, :].reshape(-1)
            vertical_loss = self._weighted_neighbor_loss(offsets, query_desc, vertical_a, vertical_b)
            if vertical_loss is not None:
                pair_losses.append(vertical_loss)

            if pair_losses:
                coherence_losses.append(torch.stack(pair_losses).mean())

        supervision_loss = torch.stack(supervision_losses).mean() if supervision_losses else zero
        coherence_loss = torch.stack(coherence_losses).mean() if coherence_losses else zero
        metrics["attention_supervision"] = supervision_loss.item()
        metrics["attention_supervision_coverage"] = (
            torch.stack(supervision_coverages).mean().item() if supervision_coverages else 0.0
        )
        metrics["attention_candidate_match"] = torch.stack(match_accuracies).mean().item() if match_accuracies else 0.0
        metrics["attention_offset_coherence"] = coherence_loss.item()
        return supervision_loss, coherence_loss, metrics

    def generator_loss(
        self,
        coarse_raw: torch.Tensor,
        refined: torch.Tensor,
        coarse_target: torch.Tensor,
        refined_target: torch.Tensor,
        mask: torch.Tensor,
        fake_logits: list[torch.Tensor] | None = None,
        attention_aux: dict[str, object] | None = None,
        *,
        step: int | None = None,
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

        attention_supervision, attention_coherence, attention_metrics = self._attention_auxiliary_losses(
            refined_target,
            attention_aux,
            step=step,
        )
        attention_supervision_weight = attention_metrics["attention_supervision_weight"]
        attention_coherence_weight = attention_metrics["attention_coherence_weight"]

        total = (
            self.coarse_l2_weight * coarse_l2
            + self.coarse_blur_l1_weight * coarse_blur_l1
            + self.coarse_gradient_weight * coarse_gradient
            + self.coarse_perceptual_weight * coarse_perceptual
            + self.refined_l1_weight * refined_l1
            + self.frequency_weight * frequency
            + self.perceptual_weight * perceptual
            + self.adversarial_weight * adversarial
            + attention_supervision_weight * attention_supervision
            + attention_coherence_weight * attention_coherence
        )
        loss_dict = {
            "coarse_l2": coarse_l2.item(),
            "coarse_blur_l1": coarse_blur_l1.item(),
            "coarse_gradient": coarse_gradient.item(),
            "coarse_perceptual": coarse_perceptual.item(),
            "refined_l1": refined_l1.item(),
            "frequency": frequency.item(),
            "perceptual": perceptual.item(),
            "adversarial_g": adversarial.item(),
            "attention_supervision": attention_metrics["attention_supervision"],
            "attention_supervision_coverage": attention_metrics["attention_supervision_coverage"],
            "attention_candidate_match": attention_metrics["attention_candidate_match"],
            "attention_offset_coherence": attention_metrics["attention_offset_coherence"],
            "attention_supervision_weight": attention_supervision_weight,
            "attention_coherence_weight": attention_coherence_weight,
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
