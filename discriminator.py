"""PatchGAN discriminator used for paper-aligned training."""

from __future__ import annotations

from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_layers: int = 3):
        super().__init__()
        channels = int(base_channels)
        layers = [
            nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        in_ch = channels
        for layer_idx in range(1, int(n_layers)):
            out_ch = min(base_channels * (2 ** layer_idx), 512)
            stride = 1 if layer_idx == n_layers - 1 else 2
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_ch = out_ch

        layers.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, image):
        return self.model(image)
