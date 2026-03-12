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
        self.adversarial_mode = str(adversarial_mode).lower()
        if self.adversarial_mode not in {"bce", "hinge"}:
            raise ValueError(
                f"Unsupported adversarial_mode: {adversarial_mode}. Expected 'bce' or 'hinge'."
            )

        self.frequency_loss = FocalFrequencyLoss(alpha=focal_alpha, log_matrix=focal_log_matrix)
        self.perceptual_loss = PerceptualLoss()
        self.coarse_blur = NativeGaussianBlur2d((7, 7), sigma=(2.01, 2.01))

    def generator_loss(
        self,
        coarse_raw: torch.Tensor,
        refined: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        fake_logits: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        coarse_composite = composite_with_known(coarse_raw, target, mask)
        coarse_blurred = self.coarse_blur(coarse_composite)
        target_blurred = self.coarse_blur(target)
        coarse_grad_mask = dilate_mask(mask)
        coarse_l2 = masked_mse_loss(coarse_raw, target, mask)
        coarse_blur_l1 = masked_l1_loss(coarse_blurred, target_blurred, mask)
        coarse_gradient = masked_gradient_l1_loss(coarse_blurred, target_blurred, coarse_grad_mask)
        coarse_perceptual = self.perceptual_loss(coarse_composite, target)
        refined_l1 = masked_l1_loss(refined, target, mask)
        frequency = self.frequency_loss(refined, target)
        perceptual = self.perceptual_loss(refined, target)

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
