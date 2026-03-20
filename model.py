import torch
from torch import nn
from torch.nn import functional as F
from typing import Any

from coarse import COARSE_MODEL_REGISTRY
from upscale import AttentionUpscaling
from patchmatch import PatchInpainting


class InpaintingModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        coarse_cfg = config["coarse_model"]
        coarse_enabled = coarse_cfg.get("enabled", True)
        inpainter_params = dict(config["inpainter"]["params"])

        self.coarse_model = None
        if coarse_enabled:
            coarse_class_name = coarse_cfg.get("class", "CoarseModel")
            coarse_class = COARSE_MODEL_REGISTRY.get(coarse_class_name)
            if coarse_class is None:
                raise ValueError(f"Unsupported coarse model class: {coarse_class_name}")
            self.coarse_model = coarse_class(**coarse_cfg["parameters"])
        elif inpainter_params.get("match_coarse_rgb", True) or inpainter_params.get("concat_features", True):
            raise ValueError("Coarse model cannot be disabled while match_coarse_rgb or concat_features is enabled.")

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
        self.hr_upscaler = AttentionUpscaling(self.inpainter)

    def _prefilter_downsample(self, image: torch.Tensor, out_size: int) -> torch.Tensor:
        if image.shape[-2:] == (out_size, out_size):
            return image
        image = self.inpainter.final_gaussian_blur(image)
        return F.interpolate(image, size=(out_size, out_size), mode="bicubic", align_corners=False)

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        value_image: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        return self.inpainter(image, mask, value_image=value_image, return_aux=return_aux)

    def predict(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        value_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_image_size = int(self.inpainter.image_size)
        if image.shape[-2:] == (model_image_size, model_image_size):
            refined, _, _ = self.inpainter(image, mask, value_image=value_image, return_aux=False)
            return refined

        known_hr = image if value_image is None else value_image
        refine_target_lr = F.interpolate(
            known_hr,
            size=(model_image_size, model_image_size),
            mode="bicubic",
            align_corners=False,
        )
        image_lr = self._prefilter_downsample(known_hr, model_image_size)
        mask_lr = F.interpolate(mask, size=(model_image_size, model_image_size), mode="nearest")
        mask_lr = (mask_lr > 0.5).to(image_lr.dtype)
        masked_lr = image_lr * (1 - mask_lr)

        refined_lr, attn_map, _ = self.inpainter(
            masked_lr,
            mask_lr,
            value_image=refine_target_lr,
            return_aux=False,
        )
        refined_lr = refined_lr.clamp(0, 1)

        masked_hr = known_hr * (1 - mask)
        final_hr = self.hr_upscaler(masked_hr, refined_lr, attn_map, mask_hr=mask)
        return final_hr.clamp(0, 1)

    def reparameterize(self) -> "InpaintingModel":
        """Apply any model-specific inference-time reparameterization."""
        if self.coarse_model is not None and hasattr(self.coarse_model, "reparameterize"):
            self.coarse_model.reparameterize()
        if hasattr(self.inpainter, "reparameterize"):
            self.inpainter.reparameterize()
        return self

__all__ = ["InpaintingModel"]
