"""VGG-based perceptual and style losses for image inpainting.

V6 keeps VGG frozen while preserving gradient flow through the prediction path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from device_utils import get_autocast_device_type


class VGGFeatureExtractor(nn.Module):
    """Extract features from VGG19 at specified layers."""

    def __init__(self, layer_indices=(3, 8, 15, 22), device=None):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList()

        prev = 0
        for idx in sorted(layer_indices):
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:idx + 1]))
            prev = idx + 1

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.eval()

    def forward(self, x):
        """Extract VGG features for a normalized [0, 1] RGB tensor."""
        x = (x - self.mean) / self.std

        features = []
        h = x
        for s in self.slices:
            h = s(h)
            features.append(h)
        return features


def gram_matrix(x):
    """Compute Gram matrix for style loss."""
    with torch.amp.autocast(get_autocast_device_type(x.device), enabled=False):
        b, c, h, w = x.shape
        f = x.float().view(b, c, h * w)
        gram = torch.bmm(f, f.transpose(1, 2))
        return gram / (c * h * w)


class PerceptualLoss(nn.Module):
    """Perceptual loss: L1 between VGG features of prediction and target."""

    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.vgg = VGGFeatureExtractor()
        self.weights = weights

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        with torch.no_grad():
            target_features = self.vgg(target)

        loss = 0
        for w, pf, tf in zip(self.weights, pred_features, target_features):
            loss += w * F.l1_loss(pf, tf)
        return loss


class StyleLoss(nn.Module):
    """Style loss: L1 between Gram matrices of VGG features."""

    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.vgg = VGGFeatureExtractor()
        self.weights = weights

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        with torch.no_grad():
            target_features = self.vgg(target)

        loss = 0
        for w, pf, tf in zip(self.weights, pred_features, target_features):
            loss += w * F.l1_loss(gram_matrix(pf), gram_matrix(tf))
        return loss
