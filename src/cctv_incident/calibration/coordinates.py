"""Geometry in decoded video pixels; vertex order is never inferred from position."""

from dataclasses import dataclass

import cv2
import numpy as np

from .homography import Calibration, checked_points, transform_points


def validate_quad(points, image_size=None, min_area=1.0):
    points = checked_points(points)
    if len(points) != 4:
        raise ValueError("Expected four vertices in P1, P2, P3, P4 order")
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    tolerance = max(1e-9, float(np.max(np.linalg.norm(edges, axis=1))) ** 2 * 1e-8)
    if not (np.all(turns > tolerance) or np.all(turns < -tolerance)):
        raise ValueError("Vertices must be convex and ordered without crossings or flat corners")
    area = (
        abs(
            float(
                np.sum(
                    points[:, 0] * np.roll(points[:, 1], -1)
                    - points[:, 1] * np.roll(points[:, 0], -1)
                )
            )
        )
        / 2
    )
    span = np.ptp(points, axis=0)
    if area < min_area or area / max(float(np.max(span)) ** 2, 1e-9) < 1e-6:
        raise ValueError("Calibration quadrilateral has insufficient area or is nearly collinear")
    if image_size is not None:
        width, height = image_size
        if (points < 0).any() or (points > [width - 1, height - 1]).any():
            raise ValueError("Calibration vertices fall outside the decoded image")
    return points


def rectangle_destination(width, length):
    if not np.isfinite([width, length]).all() or min(width, length) <= 0:
        raise ValueError("Width and length must be finite positive distances")
    return [[0.0, 0.0], [float(width), 0.0], [float(width), float(length)], [0.0, float(length)]]


def validate_roi(points, image_size):
    """Allow concave road polygons, but reject duplicate or crossing edges."""
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not 3 <= len(points) <= 64
        or not np.isfinite(points).all()
        or len(np.unique(points, axis=0)) != len(points)
    ):
        raise ValueError("La ROI richiede almeno tre vertici distinti e finiti")
    if (points < 0).any() or (points > np.asarray(image_size) - 1).any():
        raise ValueError("ROI falls outside the decoded image")
    if abs(cv2.contourArea(points.astype(np.float32))) < 1:
        raise ValueError("La ROI ha area insufficiente")

    def orientation(a, b, c):
        edge, delta = b - a, c - a
        return edge[0] * delta[1] - edge[1] * delta[0]

    count = len(points)
    for i in range(count):
        a, b = points[i], points[(i + 1) % count]
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            c, d = points[j], points[(j + 1) % count]
            if (
                np.maximum(np.minimum(a, b), np.minimum(c, d))
                > np.minimum(np.maximum(a, b), np.maximum(c, d))
            ).any():
                continue
            if (
                orientation(a, b, c) * orientation(a, b, d) <= 0
                and orientation(c, d, a) * orientation(c, d, b) <= 0
            ):
                raise ValueError("I lati della ROI non devono incrociarsi")
    return points


def full_image_roi(image_size):
    width, height = image_size
    if min(width, height) < 2:
        raise ValueError("Decoded image must be at least two pixels on each axis")
    return [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]]


@dataclass(frozen=True)
class ImageTransform:
    """Original → processed/display pixels, including optional display margins.

    Sizes are actual array dimensions. The origin is (0, 0); x/y scale by the
    corresponding dimension ratio, without sorting or clipping coordinates.
    """

    original_size: tuple[int, int]
    target_size: tuple[int, int]
    scale: tuple[float, float]
    offset: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        sizes = (*self.original_size, *self.target_size)
        if len(sizes) != 4 or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 2 for v in sizes
        ):
            raise ValueError("Image dimensions must be integer sizes of at least two pixels")
        if not np.isfinite([*self.scale, *self.offset]).all() or min(self.scale) <= 0:
            raise ValueError("Image transform requires finite positive scales and finite offsets")

    @classmethod
    def resize(cls, original_size, target_size):
        if min(original_size) < 2:
            raise ValueError("Image dimensions must be at least two pixels")
        return cls(
            tuple(original_size),
            tuple(target_size),
            (target_size[0] / original_size[0], target_size[1] / original_size[1]),
        )

    @property
    def matrix(self):
        sx, sy = self.scale
        x, y = self.offset
        return np.array([[sx, 0, x], [0, sy, y], [0, 0, 1]], dtype=float)

    def forward(self, points):
        return transform_points(self.matrix, points)

    def inverse(self, points):
        return transform_points(np.linalg.inv(self.matrix), points)

    def adapt_calibration(self, calibration):
        if calibration.image_size != self.original_size:
            raise ValueError("Transform original size differs from calibration")
        data = calibration.model_dump()
        data.update(
            image_size=self.target_size,
            source_points_px=self.forward(calibration.source_points_px).tolist(),
            roi_px=self.forward(calibration.roi_px).tolist(),
            homography=(np.asarray(calibration.homography) @ np.linalg.inv(self.matrix)).tolist(),
        )
        return Calibration.model_validate(data)

    def adapt_reference(self, image):
        if (image.shape[1], image.shape[0]) != self.original_size:
            raise ValueError("Reference image differs from transform original size")
        if self.offset == (0.0, 0.0) and self.scale == (
            self.target_size[0] / self.original_size[0],
            self.target_size[1] / self.original_size[1],
        ):
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        return cv2.warpAffine(image, self.matrix[:2], self.target_size)

    def adapt_motion_threshold(self, threshold_px):
        """Conservative scalar threshold for the existing camera-motion guard."""
        if not np.isfinite(threshold_px) or threshold_px <= 0:
            raise ValueError("Motion threshold must be finite and positive")
        return float(threshold_px * min(self.scale))
