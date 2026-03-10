"""Coarse inpainting backbones."""

import torch
from torch import nn
from torch.nn import functional as F

from blocks import DepthwiseSeparableBlock, RepDepthwiseSeparableBlock


class CoarseEncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block_cls: type[nn.Module] = DepthwiseSeparableBlock):
        super().__init__()
        self.down = block_cls(in_channels, out_channels, stride=2, use_residual=True)
        self.refine = block_cls(out_channels, out_channels, stride=1, use_residual=True)

    def forward(self, x):
        x = self.down(x)
        return self.refine(x)

    def reparameterize(self):
        self.down.reparameterize()
        self.refine.reparameterize()
        return self


class CoarseUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        block_cls: type[nn.Module] = DepthwiseSeparableBlock,
    ):
        super().__init__()
        self.block1 = block_cls(
            in_channels + skip_channels,
            out_channels,
            stride=1,
            use_residual=True,
        )
        self.block2 = block_cls(out_channels, out_channels, stride=1, use_residual=True)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x)
        return self.block2(x)

    def reparameterize(self):
        self.block1.reparameterize()
        self.block2.reparameterize()
        return self


class CoarseModel(nn.Module):
    def __init__(self, channels=None, head_channels: int = 32, use_rep_blocks: bool = False, **kwargs):
        super().__init__()
        if kwargs:
            raise ValueError(f"Unsupported CoarseModel arguments: {sorted(kwargs)}")

        if channels is None:
            channels = [64, 128, 256, 384, 512]
        if len(channels) != 5:
            raise ValueError(f"CoarseModel expects 5 channel values, got {channels}")

        c0, c1, c2, c3, c4 = [int(c) for c in channels]
        self.feature_channels = [c0, c1, c2, c3, c4]
        self.use_rep_blocks = bool(use_rep_blocks)
        block_cls = RepDepthwiseSeparableBlock if self.use_rep_blocks else DepthwiseSeparableBlock

        self.stage0 = CoarseEncoderStage(3, c0, block_cls=block_cls)
        self.stage1 = CoarseEncoderStage(c0, c1, block_cls=block_cls)
        self.stage2 = CoarseEncoderStage(c1, c2, block_cls=block_cls)
        self.stage3 = CoarseEncoderStage(c2, c3, block_cls=block_cls)
        self.stage4 = CoarseEncoderStage(c3, c4, block_cls=block_cls)

        self.up4 = CoarseUpBlock(c4, c3, c3, block_cls=block_cls)
        self.up3 = CoarseUpBlock(c3, c2, c2, block_cls=block_cls)
        self.up2 = CoarseUpBlock(c2, c1, c1, block_cls=block_cls)
        self.up1 = CoarseUpBlock(c1, c0, c0, block_cls=block_cls)
        self.head_channels = int(head_channels)
        self.head_block = block_cls(c0 + 3, self.head_channels, stride=1, use_residual=True)
        self.out_conv = nn.Conv2d(self.head_channels, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = []
        x0 = self.stage0(x)
        features.append(x0)
        x1 = self.stage1(x0)
        features.append(x1)
        x2 = self.stage2(x1)
        features.append(x2)
        x3 = self.stage3(x2)
        features.append(x3)
        x4 = self.stage4(x3)
        features.append(x4)

        out = self.up4(x4, x3)
        out = self.up3(out, x2)
        out = self.up2(out, x1)
        out = self.up1(out, x0)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = torch.cat([out, x], dim=1)
        out = self.head_block(out)
        out = self.sigmoid(self.out_conv(out))
        return out, features

    def reparameterize(self):
        self.stage0.reparameterize()
        self.stage1.reparameterize()
        self.stage2.reparameterize()
        self.stage3.reparameterize()
        self.stage4.reparameterize()
        self.up4.reparameterize()
        self.up3.reparameterize()
        self.up2.reparameterize()
        self.up1.reparameterize()
        self.head_block.reparameterize()
        return self


COARSE_MODEL_REGISTRY = {
    "CoarseModel": CoarseModel,
}
