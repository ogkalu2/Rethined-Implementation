import torch
from torch import nn
from torch.nn import functional as F

from coarse import COARSE_MODEL_REGISTRY
from hr import AttentionUpscaling
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
        self.hr_upscaler = AttentionUpscaling(self.inpainter)

    def _prefilter_downsample(self, image: torch.Tensor, out_size: int) -> torch.Tensor:
        if image.shape[-2:] == (out_size, out_size):
            return image
        image = self.inpainter.final_gaussian_blur(image)
        return F.interpolate(image, size=(out_size, out_size), mode="bicubic", align_corners=False)

    def forward(
        self, 
        image, 
        mask, 
        value_image=None, 
        return_aux=False, 
        return_final=False,
        ):

        model_image_size = int(self.inpainter.image_size)
        # In eval mode, default to returning a single final image tensor.
        # Training keeps the multi-output path unless explicitly overridden.
        use_final_path = return_final or ((not self.training) and (not return_aux))

        if not use_final_path:
            return self.inpainter(image, mask, value_image=value_image, return_aux=return_aux)

        if return_aux:
            raise ValueError("return_aux is not supported when using the final-image HR path.")

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

    def reparameterize(self):
        """Apply any model-specific inference-time reparameterization."""
        if hasattr(self.coarse_model, "reparameterize"):
            self.coarse_model.reparameterize()
        if hasattr(self.inpainter, "reparameterize"):
            self.inpainter.reparameterize()
        return self

__all__ = ["InpaintingModel"]
