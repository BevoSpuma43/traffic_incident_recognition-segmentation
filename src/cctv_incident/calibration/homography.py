from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml
from pydantic import Field, model_validator

from cctv_incident.config import Settings


def checked_points(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError("At least four x/y correspondences are required")
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Points must be finite and distinct")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("Collinear calibration points")
    return points


def transform_points(matrix, points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError("Point projects onto the homography horizon")
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def order_quad(points):
    points = checked_points(points)
    if len(points) != 4:
        raise ValueError("Expected four vertices")
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    if len(hull) != 4:
        raise ValueError("Quadrilateral must be convex")
    # Clockwise in image coordinates, beginning at the upper-left vertex.
    angles = np.arctan2(hull[:, 1] - hull[:, 1].mean(), hull[:, 0] - hull[:, 0].mean())
    ordered = hull[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)


class Calibration(Settings):
    camera_id: str
    image_size: tuple[int, int]
    method: Literal["manual", "auto_assisted", "image_reference"] = "manual"
    units: Literal["m", "canonical", "px"] = "m"
    source_points_px: list[list[float]]
    destination_points: list[list[float]]
    homography: list[list[float]]
    roi_px: list[list[float]]
    confidence: float = Field(1, ge=0, le=1)
    reprojection_error: float = Field(0, ge=0)
    accepted: bool = True
    valid: bool = True
    reference_image: str | None = None

    @model_validator(mode="after")
    def geometry(self):
        src = checked_points(self.source_points_px)
        dst = checked_points(self.destination_points)
        if src.shape != dst.shape:
            raise ValueError("Correspondence counts differ")
        h = np.asarray(self.homography, dtype=float)
        if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) < 3:
            raise ValueError("Degenerate homography")
        if np.linalg.cond(h) > 1e12:
            raise ValueError("Ill-conditioned homography")
        if min(self.image_size) <= 0 or len(self.roi_px) < 3:
            raise ValueError("Invalid image size or road ROI")
        checked_points(self.roi_px) if len(self.roi_px) >= 4 else None
        roi = np.asarray(self.roi_px, np.float32)
        if not np.isfinite(roi).all() or abs(cv2.contourArea(roi)) < 1:
            raise ValueError("Degenerate ROI")
        return self

    def transform(self, point):
        if not self.valid:
            raise ValueError("Calibration invalidated: recalibrate the camera")
        return transform_points(self.homography, [point])[0]

    def metric_valid(self, min_confidence=0.55):
        return (
            self.valid and self.accepted and self.units == "m" and self.confidence >= min_confidence
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )


def estimate_calibration(
    camera_id, image_size, source, destination, units="m", roi=None, method="manual", accepted=True
):
    src, dst = checked_points(source), checked_points(destination)
    if src.shape != dst.shape:
        raise ValueError("Correspondence counts differ")
    if len(src) == 4:
        if len(cv2.convexHull(src.astype(np.float32))) != 4:
            raise ValueError("Four source points must form a convex quadrilateral")
        h = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        inliers = np.ones(4, dtype=bool)
    else:
        h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 0.25 if units == "m" else 2.0)
        if h is None or inliers is None or inliers.sum() < 4:
            raise ValueError("Homography estimation failed")
        inliers = inliers.ravel().astype(bool)
    if h is None or np.linalg.matrix_rank(h) < 3:
        raise ValueError("Degenerate homography")
    error = float(np.linalg.norm(transform_points(h, src)[inliers] - dst[inliers], axis=1).mean())
    roi = cv2.convexHull(src.astype(np.float32)).reshape(-1, 2).tolist() if roi is None else roi
    return Calibration(
        camera_id=camera_id,
        image_size=image_size,
        units=units,
        source_points_px=src.tolist(),
        destination_points=dst.tolist(),
        homography=h.tolist(),
        roi_px=roi,
        method=method,
        accepted=accepted,
        reprojection_error=error,
        confidence=float(inliers.mean() * np.exp(-error)),
    )


def load_calibration(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "schema_version" in data:
        from .records import to_runtime_calibration
        from .repository import load_record

        return to_runtime_calibration(load_record(path))
    return Calibration.model_validate(data)


def image_reference(camera_id, image_size):
    """Identity in pixels; this is an image coordinate system, not a road calibration."""
    width, height = image_size
    points = [
        [0.0, 0.0],
        [float(width - 1), 0.0],
        [float(width - 1), float(height - 1)],
        [0.0, float(height - 1)],
    ]
    return Calibration(
        camera_id=camera_id,
        image_size=image_size,
        method="image_reference",
        units="px",
        source_points_px=points,
        destination_points=points,
        homography=np.eye(3).tolist(),
        roi_px=points,
        confidence=0.0,
    )
