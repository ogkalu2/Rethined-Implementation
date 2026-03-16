"""LaMa-style free-form mask generation for image inpainting.

Generates random masks with controllable coverage (30-50%).
Mask convention: 1 = hole (to be inpainted), 0 = valid region.
"""

import numpy as np
import cv2


class FreeFormMaskGenerator:
    """Generate random free-form masks using brush strokes and rectangles.

    Based on the mask generation strategy described in LaMa and used by RETHINED.
    """

    def __init__(
        self,
        image_size: int = 256,
        min_coverage: float = 0.3,
        max_coverage: float = 0.5,
        max_strokes: int = 10,
        max_stroke_width: int = 40,
        max_vertices: int = 12,
        max_retries: int = 20,
    ):
        self.image_size = image_size
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.max_strokes = max_strokes
        self.max_stroke_width = max_stroke_width
        self.max_vertices = max_vertices
        self.max_retries = max_retries

    def _random_stroke(self, mask: np.ndarray, rng: np.random.RandomState):
        """Draw a single random brush stroke (polyline with random width)."""
        h, w = mask.shape
        num_vertices = rng.randint(2, self.max_vertices + 1)
        width = rng.randint(5, self.max_stroke_width + 1)

        # Generate random vertices with smooth bezier-like movement
        points = []
        start_x = rng.randint(0, w)
        start_y = rng.randint(0, h)
        points.append((start_x, start_y))

        for _ in range(num_vertices - 1):
            # Random walk from previous point
            dx = rng.randint(-w // 3, w // 3 + 1)
            dy = rng.randint(-h // 3, h // 3 + 1)
            x = np.clip(points[-1][0] + dx, 0, w - 1)
            y = np.clip(points[-1][1] + dy, 0, h - 1)
            points.append((int(x), int(y)))

        # Draw polyline
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=width)

        # Draw circles at each vertex for rounded joints
        for pt in points:
            cv2.circle(mask, pt, width // 2, color=1, thickness=-1)

    def _random_rectangle(self, mask: np.ndarray, rng: np.random.RandomState):
        """Draw a random rectangle."""
        h, w = mask.shape
        rect_w = rng.randint(w // 8, w // 2)
        rect_h = rng.randint(h // 8, h // 2)
        x = rng.randint(0, w - rect_w + 1)
        y = rng.randint(0, h - rect_h + 1)
        mask[y:y + rect_h, x:x + rect_w] = 1

    def __call__(self, rng: np.random.RandomState | None = None) -> np.ndarray:
        """Generate a random mask.

        Returns:
            np.ndarray: Binary mask of shape (H, W), dtype float32.
                        1 = hole, 0 = valid.
        """
        if rng is None:
            rng = np.random.RandomState()

        for _ in range(self.max_retries):
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

            num_strokes = rng.randint(3, self.max_strokes + 1)
            for _ in range(num_strokes):
                if rng.random() < 0.7:
                    self._random_stroke(mask, rng)
                else:
                    self._random_rectangle(mask, rng)

            coverage = mask.mean()
            if self.min_coverage <= coverage <= self.max_coverage:
                return mask

            # If coverage too low, add more strokes
            if coverage < self.min_coverage:
                extra_strokes = int((self.min_coverage - coverage) * 10) + 2
                for _ in range(extra_strokes):
                    self._random_stroke(mask, rng)
                    if rng.random() < 0.3:
                        self._random_rectangle(mask, rng)
                coverage = mask.mean()
                if coverage <= self.max_coverage:
                    return mask

        # Fallback: generate a center block mask with random noise
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        target = rng.uniform(self.min_coverage, self.max_coverage)
        side = int(np.sqrt(target) * self.image_size)
        offset_x = (self.image_size - side) // 2
        offset_y = (self.image_size - side) // 2
        mask[offset_y:offset_y + side, offset_x:offset_x + side] = 1
        return mask
