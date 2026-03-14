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
        retrieval_loss_weight: float = 0.0,
        retrieval_teacher_patch_padding: int = 8,
        retrieval_teacher_temperature: float = 0.07,
        boundary_identity_weight: float = 0.0,
        coordinate_loss_weight: float = 0.0,
        coherence_loss_weight: float = 0.0,
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
        self.retrieval_loss_weight = float(retrieval_loss_weight)
        self.retrieval_teacher_patch_padding = max(0, int(retrieval_teacher_patch_padding))
        self.retrieval_teacher_temperature = float(retrieval_teacher_temperature)
        self.boundary_identity_weight = float(boundary_identity_weight)
        self.coordinate_loss_weight = float(coordinate_loss_weight)
        self.coherence_loss_weight = float(coherence_loss_weight)
        self.frequency_weight = float(frequency_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.adversarial_weight = float(adversarial_weight)
        self.adversarial_mode = str(adversarial_mode).lower()
        if self.adversarial_mode not in {"bce", "hinge"}:
            raise ValueError(
                f"Unsupported adversarial_mode: {adversarial_mode}. Expected 'bce' or 'hinge'."
            )
        if self.retrieval_teacher_temperature <= 0:
            raise ValueError("retrieval_teacher_temperature must be positive.")

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
        patch_tokens = patch_tokens.float()
        patch_tokens = patch_tokens - patch_tokens.mean(dim=-1, keepdim=True)
        return F.normalize(patch_tokens, dim=-1, eps=1e-6)

    def _token_coords(
        self,
        indices: torch.Tensor,
        token_hw: tuple[int, int],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        height, width = token_hw
        ys = torch.div(indices, width, rounding_mode="floor").to(dtype=dtype)
        xs = (indices % width).to(dtype=dtype)
        if height > 1:
            ys = ys / float(height - 1)
        if width > 1:
            xs = xs / float(width - 1)
        return torch.stack([xs, ys], dim=-1)

    def _relative_coord_coherence_loss(
        self,
        pred_coords: torch.Tensor,
        teacher_coords: torch.Tensor,
        query_indices: torch.Tensor,
        token_hw: tuple[int, int],
    ) -> torch.Tensor | None:
        if query_indices.numel() < 2:
            return None

        num_tokens = token_hw[0] * token_hw[1]
        width = token_hw[1]
        local_lookup = torch.full((num_tokens,), -1, dtype=torch.long, device=query_indices.device)
        local_indices = torch.arange(query_indices.numel(), dtype=torch.long, device=query_indices.device)
        local_lookup[query_indices] = local_indices

        qx = query_indices % width
        pair_src = []
        pair_dst = []

        right_local = torch.full_like(query_indices, -1)
        right_mask = qx + 1 < width
        if right_mask.any():
            right_global = query_indices[right_mask] + 1
            right_local[right_mask] = local_lookup[right_global]
        valid_right = right_local >= 0
        if valid_right.any():
            pair_src.append(local_indices[valid_right])
            pair_dst.append(right_local[valid_right])

        down_local = torch.full_like(query_indices, -1)
        down_mask = query_indices + width < num_tokens
        if down_mask.any():
            down_global = query_indices[down_mask] + width
            down_local[down_mask] = local_lookup[down_global]
        valid_down = down_local >= 0
        if valid_down.any():
            pair_src.append(local_indices[valid_down])
            pair_dst.append(down_local[valid_down])

        if not pair_src:
            return None

        src = torch.cat(pair_src, dim=0)
        dst = torch.cat(pair_dst, dim=0)
        pred_delta = pred_coords[src] - pred_coords[dst]
        teacher_delta = teacher_coords[src] - teacher_coords[dst]
        return F.l1_loss(pred_delta, teacher_delta)

    def _attention_supervision_losses(
        self,
        refined_target: torch.Tensor,
        attention_aux: dict[str, object] | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        zero = refined_target.new_zeros(())
        loss_terms = {
            "retrieval": zero,
            "boundary_identity": zero,
            "coordinate": zero,
            "coherence": zero,
        }
        metrics = {
            "retrieval_recall1": 0.0,
            "retrieval_recall8": 0.0,
            "retrieval_recall32": 0.0,
            "retrieval_coord_error": 0.0,
            "boundary_identity_acc": 0.0,
        }
        if attention_aux is None:
            return loss_terms, metrics

        kernel_size = int(attention_aux.get("kernel_size", 0))
        query_mask_flat = attention_aux.get("query_mask_flat")
        supervision_entries = attention_aux.get("attention_supervision_entries")
        token_hw = attention_aux.get("token_hw")
        if (
            kernel_size <= 0
            or query_mask_flat is None
            or supervision_entries is None
            or token_hw is None
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
        boundary_losses = []
        coordinate_losses = []
        coherence_losses = []
        retrieval_recall1 = []
        retrieval_recall8 = []
        retrieval_recall32 = []
        retrieval_coord_errors = []
        boundary_accs = []

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
            teacher_logits = torch.matmul(query_teacher_tokens, key_teacher_tokens.transpose(0, 1))
            teacher_logits = teacher_logits / self.retrieval_teacher_temperature
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            boundary_mask = ~batch_query_mask
            if boundary_mask.any():
                boundary_query_indices = query_indices[boundary_mask]
                target_positions = torch.searchsorted(key_indices, boundary_query_indices)
                valid_targets = target_positions < key_indices.numel()
                if valid_targets.any():
                    valid_positions = target_positions[valid_targets]
                    valid_targets = valid_targets.clone()
                    valid_targets[valid_targets.clone()] = (
                        key_indices[valid_positions] == boundary_query_indices[valid_targets]
                    )
                if valid_targets.any():
                    teacher_probs_boundary = torch.zeros_like(teacher_probs[boundary_mask][valid_targets])
                    teacher_probs_boundary.scatter_(1, target_positions[valid_targets].unsqueeze(-1), 1.0)
                    teacher_probs[boundary_mask.nonzero(as_tuple=False).flatten()[valid_targets]] = teacher_probs_boundary
            pred_log_probs = F.log_softmax(raw_logits, dim=-1)
            pred_probs = pred_log_probs.exp()

            key_coords = self._token_coords(key_indices, token_hw, dtype=pred_probs.dtype)
            pred_coords = torch.matmul(pred_probs, key_coords)
            teacher_coords = torch.matmul(teacher_probs, key_coords)
            coordinate_losses.append(F.l1_loss(pred_coords, teacher_coords))

            coherence_loss = self._relative_coord_coherence_loss(
                pred_coords,
                teacher_coords,
                query_indices,
                token_hw,
            )
            if coherence_loss is not None:
                coherence_losses.append(coherence_loss)

            if batch_query_mask.any():
                masked_teacher_probs = teacher_probs[batch_query_mask]
                masked_pred_log_probs = pred_log_probs[batch_query_mask]
                masked_pred_probs = pred_probs[batch_query_mask]
                masked_raw_logits = raw_logits[batch_query_mask]
                masked_teacher_best = masked_teacher_probs.argmax(dim=-1)
                retrieval_losses.append(
                    (-(masked_teacher_probs * masked_pred_log_probs).sum(dim=-1)).mean()
                )
                for top_k, metric_name in ((1, retrieval_recall1), (8, retrieval_recall8), (32, retrieval_recall32)):
                    k = min(top_k, masked_raw_logits.shape[-1])
                    topk = masked_raw_logits.topk(k=k, dim=-1).indices
                    metric_name.append(
                        (topk == masked_teacher_best.unsqueeze(-1)).any(dim=-1).float().mean()
                    )
                masked_teacher_coords = torch.matmul(masked_teacher_probs, key_coords)
                masked_pred_coords = torch.matmul(masked_pred_probs, key_coords)
                retrieval_coord_errors.append(
                    (masked_pred_coords - masked_teacher_coords).abs().sum(dim=-1).mean()
                )

            if boundary_mask.any():
                boundary_query_indices = query_indices[boundary_mask]
                boundary_logits = raw_logits[boundary_mask]
                target_positions = torch.searchsorted(key_indices, boundary_query_indices)
                valid_targets = target_positions < key_indices.numel()
                if valid_targets.any():
                    valid_positions = target_positions[valid_targets]
                    valid_targets = valid_targets.clone()
                    valid_targets[valid_targets.clone()] = (
                        key_indices[valid_positions] == boundary_query_indices[valid_targets]
                    )
                if valid_targets.any():
                    valid_boundary_logits = boundary_logits[valid_targets]
                    valid_boundary_targets = target_positions[valid_targets]
                    boundary_losses.append(F.cross_entropy(valid_boundary_logits, valid_boundary_targets))
                    boundary_accs.append(
                        (valid_boundary_logits.argmax(dim=-1) == valid_boundary_targets).float().mean()
                    )

        if retrieval_losses:
            loss_terms["retrieval"] = torch.stack(retrieval_losses).mean()
        if boundary_losses:
            loss_terms["boundary_identity"] = torch.stack(boundary_losses).mean()
        if coordinate_losses:
            loss_terms["coordinate"] = torch.stack(coordinate_losses).mean()
        if coherence_losses:
            loss_terms["coherence"] = torch.stack(coherence_losses).mean()

        if retrieval_recall1:
            metrics["retrieval_recall1"] = torch.stack(retrieval_recall1).mean().item()
        if retrieval_recall8:
            metrics["retrieval_recall8"] = torch.stack(retrieval_recall8).mean().item()
        if retrieval_recall32:
            metrics["retrieval_recall32"] = torch.stack(retrieval_recall32).mean().item()
        if retrieval_coord_errors:
            metrics["retrieval_coord_error"] = torch.stack(retrieval_coord_errors).mean().item()
        if boundary_accs:
            metrics["boundary_identity_acc"] = torch.stack(boundary_accs).mean().item()
        return loss_terms, metrics

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
        attention_loss_terms, attention_metrics = self._attention_supervision_losses(refined_target, attention_aux)
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
            + self.retrieval_loss_weight * attention_loss_terms["retrieval"]
            + self.boundary_identity_weight * attention_loss_terms["boundary_identity"]
            + self.coordinate_loss_weight * attention_loss_terms["coordinate"]
            + self.coherence_loss_weight * attention_loss_terms["coherence"]
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
            "retrieval_loss": attention_loss_terms["retrieval"].item(),
            "boundary_identity_loss": attention_loss_terms["boundary_identity"].item(),
            "coordinate_loss": attention_loss_terms["coordinate"].item(),
            "coherence_loss": attention_loss_terms["coherence"].item(),
            "frequency": frequency.item(),
            "perceptual": perceptual.item(),
            "adversarial_g": adversarial.item(),
            "generator_total": total.item(),
        }
        loss_dict.update(attention_metrics)
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
