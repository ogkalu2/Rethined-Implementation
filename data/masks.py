"""Random mask generation for inpainting.

Mask convention: 1 = hole (to be inpainted), 0 = valid region.
"""

import math

import cv2
import numpy as np


class FreeFormMaskGenerator:
    """Generate either classic free-form masks or sparse elongated object masks."""

    VALID_STYLES = {"freeform", "sparse_fashion"}

    def __init__(
        self,
        image_size: int = 256,
        working_size: int | None = None,
        min_coverage: float = 0.3,
        max_coverage: float = 0.5,
        max_strokes: int = 10,
        max_stroke_width: int = 40,
        max_vertices: int = 12,
        max_retries: int = 20,
        style: str = "freeform",
        min_capsule_length_ratio: float = 0.035,
        max_capsule_length_ratio: float = 0.16,
        min_capsule_width_ratio: float = 0.008,
        max_capsule_width_ratio: float = 0.035,
        min_component_count: int = 10,
        max_component_count: int = 52,
        articulated_probability: float = 0.45,
        connector_probability: float = 0.35,
        overlap_tolerance: float = 0.08,
    ):
        self.output_size = int(image_size)
        self.image_size = min(self.output_size, int(working_size or self.output_size))
        self.image_area = self.image_size * self.image_size
        self.min_coverage = float(min_coverage)
        self.max_coverage = float(max_coverage)
        self.max_strokes = int(max_strokes)
        self.max_stroke_width = int(max_stroke_width)
        self.max_vertices = int(max_vertices)
        self.max_retries = int(max_retries)
        self.style = str(style).lower()
        self.min_capsule_length_ratio = float(min_capsule_length_ratio)
        self.max_capsule_length_ratio = float(max_capsule_length_ratio)
        self.min_capsule_width_ratio = float(min_capsule_width_ratio)
        self.max_capsule_width_ratio = float(max_capsule_width_ratio)
        self.min_component_count = int(min_component_count)
        self.max_component_count = int(max_component_count)
        self.articulated_probability = float(articulated_probability)
        self.connector_probability = float(connector_probability)
        self.overlap_tolerance = float(overlap_tolerance)

        if self.style not in self.VALID_STYLES:
            raise ValueError(f"mask style must be one of {sorted(self.VALID_STYLES)}, got {style!r}")
        if not (0.0 < self.min_coverage <= self.max_coverage < 1.0):
            raise ValueError("Mask coverage must satisfy 0 < min_coverage <= max_coverage < 1.")

    def _random_stroke(self, mask: np.ndarray, rng: np.random.RandomState):
        """Draw a single free-form brush stroke."""
        h, w = mask.shape
        num_vertices = rng.randint(2, self.max_vertices + 1)
        width = rng.randint(5, self.max_stroke_width + 1)

        points = []
        start_x = rng.randint(0, w)
        start_y = rng.randint(0, h)
        points.append((start_x, start_y))

        for _ in range(num_vertices - 1):
            dx = rng.randint(-w // 3, w // 3 + 1)
            dy = rng.randint(-h // 3, h // 3 + 1)
            x = np.clip(points[-1][0] + dx, 0, w - 1)
            y = np.clip(points[-1][1] + dy, 0, h - 1)
            points.append((int(x), int(y)))

        pts = np.array(points, dtype=np.int32)
        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=width)
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

    def _generate_freeform_mask(self, rng: np.random.RandomState) -> np.ndarray:
        for _ in range(self.max_retries):
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

            num_strokes = rng.randint(3, self.max_strokes + 1)
            for _ in range(num_strokes):
                if rng.random() < 0.7:
                    self._random_stroke(mask, rng)
                else:
                    self._random_rectangle(mask, rng)

            coverage = float(np.count_nonzero(mask)) / self.image_area
            if self.min_coverage <= coverage <= self.max_coverage:
                return mask

            if coverage < self.min_coverage:
                extra_strokes = int((self.min_coverage - coverage) * 10) + 2
                for _ in range(extra_strokes):
                    self._random_stroke(mask, rng)
                    if rng.random() < 0.3:
                        self._random_rectangle(mask, rng)
                coverage = float(np.count_nonzero(mask)) / self.image_area
                if coverage <= self.max_coverage:
                    return mask

        return self._fallback_block_mask(rng)

    def _ratio_to_px(self, ratio: float, minimum: int = 1) -> int:
        return max(minimum, int(round(ratio * self.image_size)))

    def _clip_point(self, x: float, y: float) -> tuple[int, int]:
        max_idx = self.image_size - 1
        return (
            int(np.clip(round(x), 0, max_idx)),
            int(np.clip(round(y), 0, max_idx)),
        )

    def _draw_capsule(
        self,
        mask: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        width: int,
    ):
        width = max(1, int(width))
        cv2.line(mask, start, end, color=1, thickness=width)
        radius = max(1, width // 2)
        cv2.circle(mask, start, radius, color=1, thickness=-1)
        cv2.circle(mask, end, radius, color=1, thickness=-1)

    def _draw_blob(
        self,
        mask: np.ndarray,
        center: tuple[int, int],
        major_axis: int,
        minor_axis: int,
        angle_deg: float,
    ):
        axes = (max(1, int(major_axis)), max(1, int(minor_axis)))
        cv2.ellipse(mask, center, axes, float(angle_deg), 0.0, 360.0, color=1, thickness=-1)

    def _sample_sparse_anchor(
        self,
        mask: np.ndarray,
        rng: np.random.RandomState,
        radius: int,
        has_existing_pixels: bool,
    ) -> tuple[int, int]:
        max_idx = self.image_size - 1
        radius = max(1, int(radius))
        low = radius
        high = max(max_idx - radius + 1, radius + 1)
        if not has_existing_pixels:
            return int(rng.randint(low, high)), int(rng.randint(low, high))
        for _ in range(24):
            x = int(rng.randint(low, high))
            y = int(rng.randint(low, high))
            y0 = max(0, y - radius)
            y1 = min(self.image_size, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(self.image_size, x + radius + 1)
            window = mask[y0:y1, x0:x1]
            if float(window.sum()) <= self.overlap_tolerance * window.size:
                return x, y
        return int(rng.randint(0, self.image_size)), int(rng.randint(0, self.image_size))

    def _capsule_dims(self, rng: np.random.RandomState) -> tuple[int, int]:
        length = rng.randint(
            self._ratio_to_px(self.min_capsule_length_ratio, minimum=6),
            self._ratio_to_px(self.max_capsule_length_ratio, minimum=7) + 1,
        )
        width = rng.randint(
            self._ratio_to_px(self.min_capsule_width_ratio, minimum=3),
            self._ratio_to_px(self.max_capsule_width_ratio, minimum=4) + 1,
        )
        return length, min(width, max(3, length // 2))

    def _draw_sparse_component(
        self,
        component: np.ndarray,
        occupancy_mask: np.ndarray,
        has_existing_pixels: bool,
        rng: np.random.RandomState,
    ):
        base_length, base_width = self._capsule_dims(rng)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        anchor = self._sample_sparse_anchor(
            occupancy_mask,
            rng,
            radius=base_length,
            has_existing_pixels=has_existing_pixels,
        )

        if rng.random() < self.articulated_probability:
            segment_count = rng.randint(2, 5)
            cursor = anchor
            for segment_idx in range(segment_count):
                length_scale = rng.uniform(0.65, 1.0) if segment_idx == 0 else rng.uniform(0.45, 0.85)
                width_scale = rng.uniform(0.75, 1.15)
                length = max(4, int(round(base_length * length_scale)))
                width = max(2, int(round(base_width * width_scale)))
                next_point = self._clip_point(
                    cursor[0] + math.cos(angle) * length,
                    cursor[1] + math.sin(angle) * length,
                )
                self._draw_capsule(component, cursor, next_point, width)
                if rng.random() < 0.35:
                    blob_major = max(2, int(round(width * rng.uniform(0.8, 1.6))))
                    blob_minor = max(2, int(round(width * rng.uniform(0.5, 1.1))))
                    self._draw_blob(
                        component,
                        next_point,
                        major_axis=blob_major,
                        minor_axis=blob_minor,
                        angle_deg=rng.uniform(0.0, 180.0),
                    )
                if rng.random() < self.connector_probability:
                    thin_width = max(1, width // 4)
                    connector_angle = angle + rng.uniform(-1.3, 1.3)
                    connector_len = max(5, int(round(length * rng.uniform(0.35, 0.75))))
                    connector_end = self._clip_point(
                        next_point[0] + math.cos(connector_angle) * connector_len,
                        next_point[1] + math.sin(connector_angle) * connector_len,
                    )
                    cv2.line(component, next_point, connector_end, color=1, thickness=thin_width)
                    if rng.random() < 0.5:
                        self._draw_blob(
                            component,
                            connector_end,
                            major_axis=max(2, thin_width * 2),
                            minor_axis=max(1, thin_width),
                            angle_deg=rng.uniform(0.0, 180.0),
                        )
                cursor = next_point
                angle += rng.uniform(-1.05, 1.05)
        else:
            end = self._clip_point(
                anchor[0] + math.cos(angle) * base_length,
                anchor[1] + math.sin(angle) * base_length,
            )
            self._draw_capsule(component, anchor, end, base_width)
            if rng.random() < 0.5:
                self._draw_blob(
                    component,
                    anchor if rng.random() < 0.5 else end,
                    major_axis=max(2, int(round(base_width * rng.uniform(0.7, 1.3)))),
                    minor_axis=max(2, int(round(base_width * rng.uniform(0.5, 1.1)))),
                    angle_deg=rng.uniform(0.0, 180.0),
                )
            if rng.random() < self.connector_probability:
                connector_angle = angle + rng.uniform(-1.4, 1.4)
                connector_len = max(4, int(round(base_length * rng.uniform(0.25, 0.6))))
                connector_end = self._clip_point(
                    end[0] + math.cos(connector_angle) * connector_len,
                    end[1] + math.sin(connector_angle) * connector_len,
                )
                cv2.line(component, end, connector_end, color=1, thickness=max(1, base_width // 4))

    def _generate_sparse_object_mask(self, rng: np.random.RandomState) -> np.ndarray:
        target_coverage = rng.uniform(self.min_coverage, self.max_coverage)
        max_components = rng.randint(self.min_component_count, self.max_component_count + 1)
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        filled_pixels = 0

        for _ in range(max_components * 3):
            current_coverage = filled_pixels / self.image_area
            if current_coverage >= target_coverage:
                break

            component = np.zeros_like(mask)
            self._draw_sparse_component(component, mask, filled_pixels > 0, rng)
            new_pixels = int(np.count_nonzero(component & (1 - mask)))
            if new_pixels <= 0:
                continue
            candidate_filled = filled_pixels + new_pixels
            candidate_coverage = candidate_filled / self.image_area
            gain = candidate_coverage - current_coverage
            remaining = target_coverage - current_coverage
            if candidate_coverage > self.max_coverage and current_coverage >= self.min_coverage:
                continue
            if gain > remaining * 1.8 and current_coverage >= self.min_coverage * 0.6:
                continue
            mask |= component
            filled_pixels = candidate_filled

        coverage = filled_pixels / self.image_area
        if coverage < self.min_coverage:
            for _ in range(max_components):
                component = np.zeros_like(mask)
                self._draw_sparse_component(component, mask, filled_pixels > 0, rng)
                new_pixels = int(np.count_nonzero(component & (1 - mask)))
                if new_pixels <= 0:
                    continue
                candidate_filled = filled_pixels + new_pixels
                candidate_coverage = candidate_filled / self.image_area
                if candidate_coverage > self.max_coverage:
                    continue
                mask |= component
                filled_pixels = candidate_filled
                coverage = candidate_coverage
                if coverage >= self.min_coverage:
                    break

        coverage = filled_pixels / self.image_area
        if self.min_coverage <= coverage <= self.max_coverage:
            return mask
        if filled_pixels > 0:
            return mask
        return self._fallback_block_mask(rng)

    def _fallback_block_mask(self, rng: np.random.RandomState) -> np.ndarray:
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        target = rng.uniform(self.min_coverage, self.max_coverage)
        side = int(np.sqrt(target) * self.image_size)
        offset_x = (self.image_size - side) // 2
        offset_y = (self.image_size - side) // 2
        mask[offset_y:offset_y + side, offset_x:offset_x + side] = 1
        return mask

    def __call__(self, rng: np.random.RandomState | None = None) -> np.ndarray:
        """Generate a random binary mask."""
        if rng is None:
            rng = np.random.RandomState()
        if self.style == "sparse_fashion":
            mask = self._generate_sparse_object_mask(rng)
        else:
            mask = self._generate_freeform_mask(rng)
        if self.image_size != self.output_size:
            mask = cv2.resize(mask, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.float32, copy=False)
