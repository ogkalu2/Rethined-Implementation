"""Top-level model composition for the RETHINED implementation."""

from torch import nn

from coarse import COARSE_MODEL_REGISTRY
from patchmatch import PatchInpainting


class InpaintingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        coarse_class_name = config["coarse_model"].get("class", "CoarseModel")
        coarse_class = COARSE_MODEL_REGISTRY.get(coarse_class_name)
        if coarse_class is None:
            raise ValueError(f"Unsupported coarse model class: {coarse_class_name}")

        self.coarse_model = coarse_class(**config["coarse_model"]["parameters"])
        inpainter_params = dict(config["inpainter"]["params"])

        if inpainter_params.get("concat_features", True):
            feature_i = inpainter_params.get("feature_i", 3)
            feature_channels = self.coarse_model.feature_channels
            if feature_i < 0 or feature_i >= len(feature_channels):
                raise ValueError(
                    f"model.inpainter.feature_i={feature_i} is out of range for coarse feature maps"
                )

            inferred_feature_dim = feature_channels[feature_i]
            configured_feature_dim = inpainter_params.get("feature_dim")
            if configured_feature_dim is None:
                inpainter_params["feature_dim"] = inferred_feature_dim
            elif configured_feature_dim != inferred_feature_dim:
                raise ValueError(
                    "model.inpainter.feature_dim does not match the selected coarse feature map: "
                    f"got {configured_feature_dim}, expected {inferred_feature_dim} for feature_i={feature_i}"
                )

        self.inpainter = PatchInpainting(**inpainter_params, model=self.coarse_model)

    def forward(self, image, mask, value_image=None, return_aux=False):
        return self.inpainter(image, mask, value_image=value_image, return_aux=return_aux)

    def reparameterize(self):
        """Apply any model-specific inference-time reparameterization."""
        if hasattr(self.coarse_model, "reparameterize"):
            self.coarse_model.reparameterize()
        if hasattr(self.inpainter, "reparameterize"):
            self.inpainter.reparameterize()
        return self

__all__ = ["InpaintingModel"]
