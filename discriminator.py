from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinglePatchDiscriminator(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3, 
        base_channels: int = 64, 
        n_layers: int = 3
    ):
        super().__init__()
        channels = int(base_channels)
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        in_ch = channels
        for layer_idx in range(1, int(n_layers)):
            out_ch = min(base_channels * (2 ** layer_idx), 512)
            stride = 1 if layer_idx == n_layers - 1 else 2
            layers.extend(
                [
                    nn.utils.spectral_norm(
                        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_ch = out_ch

        layers.append(
            nn.utils.spectral_norm(
                nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1)
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        num_scales: int = 1
    ):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            SinglePatchDiscriminator(in_channels, base_channels, n_layers) 
            for _ in range(num_scales)
        ])

    def forward(self, image) -> list[torch.Tensor]:
        """Return one logit map per scale (coarse-to-fine order).

        Each element is a spatial tensor of shape (B, 1, H', W').  Keeping
        them separate lets the caller compute per-scale losses and sum them
        with equal weight, avoiding the implicit fine-scale bias that arises
        from flattening tensors of different spatial sizes before averaging.
        """
        logits = []
        x = image
        for i, D in enumerate(self.discriminators):
            logits.append(D(x))
            if i != self.num_scales - 1:
                x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        return logits