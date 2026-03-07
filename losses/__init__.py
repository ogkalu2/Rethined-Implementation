"""Combined inpainting loss for RETHINED training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.perceptual import PerceptualLoss, StyleLoss


class InpaintingLoss(nn.Module):
    """Combined loss for RETHINED: masked L1 + perceptual.

    Style loss stays configurable, but defaults to off in V6 because its
    weighting is completely unspecified by the paper and a large value is a
    weak guess for this refinement architecture.
    """

    def __init__(
        self,
        l1_hole_weight: float = 6.0,
        l1_valid_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        style_weight: float = 0.0,
        gate_selective_weight: float = 0.0,
        gate_selective_margin: float = 0.0,
        sample_improvement_weight: float = 0.0,
        sample_improvement_margin: float = 0.0,
        selector_choice_weight: float = 0.0,
        selector_choice_margin: float = 0.0,
        head_oracle_weight: float = 0.0,
        head_oracle_margin: float = 0.0,
    ):
        super().__init__()
        self.l1_hole_weight = l1_hole_weight
        self.l1_valid_weight = l1_valid_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.gate_selective_weight = gate_selective_weight
        self.gate_selective_margin = gate_selective_margin
        self.sample_improvement_weight = sample_improvement_weight
        self.sample_improvement_margin = sample_improvement_margin
        self.selector_choice_weight = selector_choice_weight
        self.selector_choice_margin = selector_choice_margin
        self.head_oracle_weight = head_oracle_weight
        self.head_oracle_margin = head_oracle_margin

        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()

        # Share VGG extractor between perceptual and style loss
        self.style_loss.vgg = self.perceptual_loss.vgg

    def masked_l1(self, pred, target, mask):
        """Compute L1 loss separately for hole and valid regions.

        Args:
            pred: (B, 3, H, W) predicted image
            target: (B, 3, H, W) ground truth
            mask: (B, 1, H, W) binary mask, 1=hole, 0=valid
        """
        hole_loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        hole_loss = hole_loss / (mask.sum() * pred.shape[1] + 1e-8)

        valid_mask = 1 - mask
        valid_loss = F.l1_loss(pred * valid_mask, target * valid_mask, reduction="sum")
        valid_loss = valid_loss / (valid_mask.sum() * pred.shape[1] + 1e-8)

        return self.l1_hole_weight * hole_loss + self.l1_valid_weight * valid_loss

    def forward(self, coarse, refined, target, mask):
        """Compute total loss.

        Args:
            coarse: (B, 3, H, W) coarse model output
            refined: (B, 3, H, W) refined model output
            target: (B, 3, H, W) ground truth
            mask: (B, 1, H, W) binary mask, 1=hole

        Returns:
            total_loss, loss_dict
        """
        # L1 on both outputs
        l1_coarse = self.masked_l1(coarse, target, mask)
        l1_refined = self.masked_l1(refined, target, mask)

        # Perceptual + style on refined output only
        loss_perceptual = self.perceptual_loss(refined, target)
        if self.style_weight > 0:
            loss_style = self.style_loss(refined, target)
        else:
            loss_style = refined.new_zeros(())

        total = (
            l1_coarse
            + l1_refined
            + self.perceptual_weight * loss_perceptual
            + self.style_weight * loss_style
        )

        loss_dict = {
            "l1_coarse": l1_coarse.item(),
            "l1_refined": l1_refined.item(),
            "perceptual": loss_perceptual.item(),
            "style": loss_style.item(),
            "total": total.item(),
        }

        return total, loss_dict
