from __future__ import annotations

import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    """A lightweight focal frequency loss over the refined LR prediction."""

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if self.eps <= 0:
            raise ValueError("eps must be positive.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_freq = torch.fft.rfft2(pred.float(), norm="ortho")
        target_freq = torch.fft.rfft2(target.float(), norm="ortho")
        freq_distance = (pred_freq - target_freq).abs().square()

        weights = freq_distance.detach()
        if self.alpha != 1.0:
            weights = weights.pow(self.alpha)
        weights = weights / weights.amax(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return (weights * freq_distance).mean()
