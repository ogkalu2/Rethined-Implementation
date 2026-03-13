"""VGG-based perceptual losses for image inpainting.

V6 keeps VGG frozen while preserving gradient flow through the prediction path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """Extract features from VGG19 at specified layers."""

    def __init__(self, layer_indices=(3, 8, 15, 22)):
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


class PerceptualLoss(nn.Module):
    """Perceptual loss: L1 between VGG features of prediction and target."""

    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.vgg = VGGFeatureExtractor()
        self.weights = weights

    def forward(self, pred, target, mask=None):
        pred_features = self.vgg(pred)
        with torch.no_grad():
            target_features = self.vgg(target)

        loss = 0
        for w, pf, tf in zip(self.weights, pred_features, target_features):
            if mask is None:
                loss += w * F.l1_loss(pf, tf)
                continue

            feature_mask = F.interpolate(mask.float(), size=pf.shape[-2:], mode="bilinear", align_corners=False)
            feature_mask = feature_mask.expand(-1, pf.shape[1], -1, -1)
            diff = (pf - tf).abs() * feature_mask
            denom = feature_mask.sum(dim=(1, 2, 3)).clamp_min(1e-8)
            loss += w * (diff.sum(dim=(1, 2, 3)) / denom).mean()
        return loss
