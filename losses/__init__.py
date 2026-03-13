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
        attention_coherence_weight: float = 0.0,
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
        self.attention_coherence_weight = float(attention_coherence_weight)
        self.adversarial_mode = str(adversarial_mode).lower()
        if self.adversarial_mode not in {"bce", "hinge"}:
            raise ValueError(
                f"Unsupported adversarial_mode: {adversarial_mode}. Expected 'bce' or 'hinge'."
            )

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

    def _attention_auxiliary_losses(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        zero = refined_target.new_zeros(())
        metrics = {
            "attention_supervision": 0.0,
            "attention_candidate_match": 1.0,
            "attention_offset_coherence": 0.0,
        }
        if attention_aux is None:
            return zero, zero, metrics

        matching_candidates = attention_aux.get("matching_candidates")
        if not matching_candidates:
            return zero, zero, metrics

        patch_tokens = self._extract_patch_tokens(
            refined_target,
            patch_size=int(attention_aux["value_patch_size"]),
            stride=int(attention_aux["kernel_size"]),
            padding=int(attention_aux["value_patch_padding"]),
        )
        grid_size = int(attention_aux["token_grid_size"])

        supervision_losses = []
        match_accuracies = []
        coherence_losses = []

        for batch_idx, batch_detail in enumerate(matching_candidates):
            query_indices = batch_detail["query_indices"]
            candidate_key_indices = batch_detail["candidate_key_indices"]
            candidate_logits = batch_detail["candidate_logits"]
            if query_indices.numel() == 0 or candidate_key_indices.numel() == 0:
                continue

            patch_bank = patch_tokens[batch_idx]
            query_patches = patch_bank.index_select(0, query_indices)
            candidate_patches = patch_bank.index_select(
                0,
                candidate_key_indices.reshape(-1),
            ).view(candidate_key_indices.shape[0], candidate_key_indices.shape[1], -1)

            query_desc = self._normalize_patch_tokens(query_patches)
            candidate_desc = self._normalize_patch_tokens(candidate_patches.view(-1, candidate_patches.shape[-1]))
            candidate_desc = candidate_desc.view_as(candidate_patches)
            patch_similarity = (query_desc.unsqueeze(1) * candidate_desc).sum(dim=-1)
            target_probs = F.softmax(
                patch_similarity / self.attention_supervision_temperature,
                dim=-1,
            ).detach()
            log_probs = F.log_softmax(candidate_logits.float(), dim=-1)
            supervision_losses.append(F.kl_div(log_probs, target_probs, reduction="batchmean"))

            predicted_best = candidate_logits.argmax(dim=-1)
            target_best = target_probs.argmax(dim=-1)
            match_accuracies.append((predicted_best == target_best).to(dtype=torch.float32).mean())

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
            horizontal_valid = (horizontal_a >= 0) & (horizontal_b >= 0)
            if horizontal_valid.any():
                diff = offsets[horizontal_a[horizontal_valid]] - offsets[horizontal_b[horizontal_valid]]
                pair_losses.append(diff.abs().mean())

            vertical_a = index_map[:-1, :].reshape(-1)
            vertical_b = index_map[1:, :].reshape(-1)
            vertical_valid = (vertical_a >= 0) & (vertical_b >= 0)
            if vertical_valid.any():
                diff = offsets[vertical_a[vertical_valid]] - offsets[vertical_b[vertical_valid]]
                pair_losses.append(diff.abs().mean())

            if pair_losses:
                coherence_losses.append(torch.stack(pair_losses).mean())

        supervision_loss = torch.stack(supervision_losses).mean() if supervision_losses else zero
        coherence_loss = torch.stack(coherence_losses).mean() if coherence_losses else zero
        metrics["attention_supervision"] = supervision_loss.item()
        metrics["attention_candidate_match"] = (
            torch.stack(match_accuracies).mean().item() if match_accuracies else 1.0
        )
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
        )

        total = (
            self.coarse_l2_weight * coarse_l2
            + self.coarse_blur_l1_weight * coarse_blur_l1
            + self.coarse_gradient_weight * coarse_gradient
            + self.coarse_perceptual_weight * coarse_perceptual
            + self.refined_l1_weight * refined_l1
            + self.frequency_weight * frequency
            + self.perceptual_weight * perceptual
            + self.adversarial_weight * adversarial
            + self.attention_supervision_weight * attention_supervision
            + self.attention_coherence_weight * attention_coherence
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
            "attention_candidate_match": attention_metrics["attention_candidate_match"],
            "attention_offset_coherence": attention_metrics["attention_offset_coherence"],
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
