"""Paper-aligned NeuralPatchMatch refinement modules."""

from .attention import MultiHeadAttention
from .inpainting import PatchInpainting

MultiHeadAttention.__module__ = __name__
PatchInpainting.__module__ = __name__

__all__ = ["MultiHeadAttention", "PatchInpainting"]
