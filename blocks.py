"""Shared building blocks for the RETHINED model family."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


class NativeGaussianBlur2d(nn.Module):
    """Drop-in replacement for kornia.filters.GaussianBlur2d."""

    def __init__(self, kernel_size: tuple = (7, 7), sigma: tuple = (2.01, 2.01), **kwargs):
        super().__init__()
        ks = kernel_size[0]
        sig = sigma[0]
        x = torch.arange(ks, dtype=torch.float32) - ks // 2
        gauss_1d = torch.exp(-0.5 * (x / sig) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        self.register_buffer("kernel_h", gauss_1d.view(1, 1, 1, -1))
        self.register_buffer("kernel_v", gauss_1d.view(1, 1, -1, 1))
        self.padding = ks // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kh = self.kernel_h.to(dtype=x.dtype).expand(channels, -1, -1, -1)
        kv = self.kernel_v.to(dtype=x.dtype).expand(channels, -1, -1, -1)
        x = F.pad(x, [self.padding] * 4, mode="reflect")
        x = F.conv2d(x, kh, groups=channels)
        x = F.conv2d(x, kv, groups=channels)
        return x


def fuse_conv_bn_pair(conv: nn.Module, bn: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Fuse a Conv2d+BatchNorm2d pair for inference."""
    if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
        return conv, bn
    return fuse_conv_bn_eval(conv.eval(), bn.eval()), nn.Identity()


def fuse_conv_bn_to_weight_bias(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fused kernel and bias tensors for a Conv2d+BN branch."""
    fused = fuse_conv_bn_eval(conv.eval(), bn.eval())
    return fused.weight.detach().clone(), fused.bias.detach().clone()


def pad_kernel_to_size(kernel: torch.Tensor, target_size: int) -> torch.Tensor:
    """Center-pad a smaller spatial kernel to a target square size."""
    current_size = kernel.shape[-1]
    if current_size == target_size:
        return kernel
    if current_size > target_size or (target_size - current_size) % 2 != 0:
        raise ValueError(f"Cannot pad kernel {current_size} to target {target_size}")
    pad = (target_size - current_size) // 2
    return F.pad(kernel, [pad, pad, pad, pad])


def fuse_identity_bn_to_weight_bias(
    num_channels: int,
    bn: nn.BatchNorm2d,
    kernel_size: int,
    groups: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fused kernel and bias tensors for an identity+BN branch."""
    if num_channels % groups != 0:
        raise ValueError(
            f"Identity fusion expects num_channels divisible by groups (got {num_channels}, {groups})"
        )

    input_dim = num_channels // groups
    kernel = torch.zeros(
        (num_channels, input_dim, kernel_size, kernel_size),
        dtype=bn.weight.dtype,
        device=bn.weight.device,
    )
    center = kernel_size // 2
    channel_idx = torch.arange(num_channels, device=kernel.device)
    kernel[channel_idx, channel_idx % input_dim, center, center] = 1.0

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    std = torch.sqrt(var + bn.eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    bias = beta - mean * gamma / std
    return kernel * scale, bias


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.proj = None
        if use_residual and (stride != 1 or in_channels != out_channels):
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.depthwise(x)
        out = self.depthwise_bn(out)
        out = self.activation(out)
        out = self.pointwise(out)
        out = self.pointwise_bn(out)
        if self.use_residual:
            if self.proj is not None:
                residual = self.proj(residual)
            out = out + residual
        return self.activation(out)

    def reparameterize(self):
        self.depthwise, self.depthwise_bn = fuse_conv_bn_pair(self.depthwise, self.depthwise_bn)
        self.pointwise, self.pointwise_bn = fuse_conv_bn_pair(self.pointwise, self.pointwise_bn)
        if isinstance(self.proj, nn.Sequential) and len(self.proj) == 2:
            fused_proj, proj_bn = fuse_conv_bn_pair(self.proj[0], self.proj[1])
            if isinstance(proj_bn, nn.Identity):
                self.proj = fused_proj
        return self


class RepDepthwiseSeparableBlock(nn.Module):
    """Depthwise-separable block with a reparameterizable multi-branch depthwise stage."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_residual: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.use_residual = use_residual
        self.activation = nn.ReLU(inplace=True)

        self.depthwise = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(self.in_channels)

        self.depthwise_scale = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            stride=self.stride,
            padding=0,
            groups=self.in_channels,
            bias=False,
        )
        self.depthwise_scale_bn = nn.BatchNorm2d(self.in_channels)
        nn.init.zeros_(self.depthwise_scale_bn.weight)
        nn.init.zeros_(self.depthwise_scale_bn.bias)

        self.depthwise_identity_bn = None
        if self.stride == 1:
            self.depthwise_identity_bn = nn.BatchNorm2d(self.in_channels)
            nn.init.zeros_(self.depthwise_identity_bn.weight)
            nn.init.zeros_(self.depthwise_identity_bn.bias)

        self.pointwise = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(self.out_channels)

        self.proj = None
        if self.use_residual and (self.stride != 1 or self.in_channels != self.out_channels):
            self.proj = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.depthwise_bn(self.depthwise(x))
        if self.depthwise_scale is not None:
            out = out + self.depthwise_scale_bn(self.depthwise_scale(x))
        if self.depthwise_identity_bn is not None:
            out = out + self.depthwise_identity_bn(x)
        out = self.activation(out)

        out = self.pointwise_bn(self.pointwise(out))
        if self.use_residual:
            if self.proj is not None:
                residual = self.proj(residual)
            out = out + residual
        return self.activation(out)

    def reparameterize(self):
        dw_kernel, dw_bias = fuse_conv_bn_to_weight_bias(self.depthwise, self.depthwise_bn)
        scale_kernel, scale_bias = fuse_conv_bn_to_weight_bias(self.depthwise_scale, self.depthwise_scale_bn)
        dw_kernel = dw_kernel + pad_kernel_to_size(scale_kernel, target_size=3)
        dw_bias = dw_bias + scale_bias
        if self.depthwise_identity_bn is not None:
            id_kernel, id_bias = fuse_identity_bn_to_weight_bias(
                num_channels=self.in_channels,
                bn=self.depthwise_identity_bn,
                kernel_size=3,
                groups=self.in_channels,
            )
            dw_kernel = dw_kernel + id_kernel
            dw_bias = dw_bias + id_bias

        fused_depthwise = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.in_channels,
            bias=True,
        ).to(device=dw_kernel.device, dtype=dw_kernel.dtype)
        fused_depthwise.weight.data.copy_(dw_kernel)
        fused_depthwise.bias.data.copy_(dw_bias)
        self.depthwise = fused_depthwise
        self.depthwise_bn = nn.Identity()
        self.depthwise_scale = None
        self.depthwise_scale_bn = None
        self.depthwise_identity_bn = None

        self.pointwise, self.pointwise_bn = fuse_conv_bn_pair(self.pointwise, self.pointwise_bn)
        if isinstance(self.proj, nn.Sequential) and len(self.proj) == 2:
            fused_proj, proj_bn = fuse_conv_bn_pair(self.proj[0], self.proj[1])
            if isinstance(proj_bn, nn.Identity):
                self.proj = fused_proj
        return self
