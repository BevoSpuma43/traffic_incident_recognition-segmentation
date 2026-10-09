"""Experimental metric road plane from a shared, assumed vehicle cuboid.

Fits camera perspective to multiple image boxes; box corners are never treated
as ground correspondences. Assumes a level road, fixed camera, no roll, square
pixels, image-centred principal point and a common vehicle heading modulo pi.
"""

import cv2
import numpy as np
from pydantic import Field
from scipy.optimize import least_squares

from .homography import transform_points
from .records import RecordModel

ALGORITHM_VERSION = "mean-vehicle-cuboid-v1"
REFERENCE_PAPER = "https://arxiv.org/abs/1702.06451"


class VehicleFitError(ValueError):
    """Rejected perspective, retaining numerical evidence for review."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class VehicleDimensions(RecordModel):
    length_m: float = Field(4.7, ge=3.0, le=6.5)
    width_m: float = Field(1.8, ge=1.4, le=2.5)
    height_m: float = Field(1.5, ge=1.1, le=2.5)

    def source(self):
        return (
            f"Ipotesi sperimentale di auto tipo: lunghezza {self.length_m:g} m, "
            f"larghezza {self.width_m:g} m, altezza {self.height_m:g} m. "
            "Valori scelti per il progetto, non una media statistica verificata del parco USA. "
            "Scala derivata da un modello 3D semplificato e dalle osservazioni dei veicoli; "
            "non da un rilievo sul posto."
        )


def camera_road_matrix(image_size, focal_ratio, pitch, height):
    width, image_height = image_size
    focal = focal_ratio * width
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, image_height / 2], [0, 0, 1]])
    sine, cosine = np.sin(pitch), np.cos(pitch)
    plane = np.array([[1, 0, 0], [0, -sine, height * cosine], [0, cosine, height * sine]])
    return intrinsics @ plane


def project_cuboids(image_size, camera, positions, dimensions):
    """Return eight projected 3D corners per vehicle; first four are at Z=0."""
    focal_ratio, pitch, height, heading = camera
    offsets = np.array(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    ) * [dimensions.width_m / 2, dimensions.length_m / 2, dimensions.height_m]
    rotation = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
    offsets[:, :2] = offsets[:, :2] @ rotation.T
    world = offsets[None, :, :] + np.c_[positions, np.zeros(len(positions))][:, None, :]
    x, y, z = np.moveaxis(world, -1, 0)
    sine, cosine = np.sin(pitch), np.cos(pitch)
    depth = y * cosine + (height - z) * sine
    focal = focal_ratio * image_size[0]
    return np.stack(
        [
            image_size[0] / 2 + focal * x / depth,
            image_size[1] / 2 + focal * ((height - z) * cosine - y * sine) / depth,
        ],
        axis=-1,
    )


def projected_boxes(corners):
    return np.c_[corners.min(axis=1), corners.max(axis=1)]


def fit_vehicle_camera(boxes, image_size, dimensions=None):
    """Bounded robust fit, rejecting weak coverage, outliers and ambiguous fits.

    The reported residual/spread describe this model fit, not real metric accuracy.
    """
    dimensions = dimensions or VehicleDimensions()
    boxes = np.asarray(boxes, dtype=float)
    size = np.asarray(image_size, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not 5 <= len(boxes) <= 12:
        raise ValueError("Servono da 5 a 12 osservazioni utili di automobili")
    if (
        size.shape != (2,)
        or not np.isfinite(size).all()
        or min(size) < 64
        or not np.isfinite(boxes).all()
    ):
        raise ValueError("Dimensioni immagine o osservazioni non valide")
    spans = boxes[:, 2:] - boxes[:, :2]
    if (spans < 14).any() or (boxes[:, :2] < 0).any() or (boxes[:, 2:] > size - 1).any():
        raise ValueError("Automobili troppo piccole, tagliate o con geometria non valida")
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    if np.ptp(centers[:, 1]) < size[1] * 0.06 or spans[:, 1].max() / spans[:, 1].min() < 1.3:
        raise ValueError("Le auto devono essere osservate a distanze diverse dalla camera")
    if abs(cv2.contourArea(cv2.convexHull(centers.astype(np.float32)))) < np.prod(size) * 0.005:
        raise ValueError(
            "Osservazioni troppo concentrate o allineate: prospettiva non determinabile"
        )
    scales = np.tile(np.maximum(spans, 20), (1, 2))
    minimum_depth = np.hypot(dimensions.length_m, dimensions.width_m) / 2 + 0.5
    lower = np.r_[
        0.5,
        np.deg2rad(5),
        max(2.5, dimensions.height_m + 0.5),
        -np.pi / 2,
        np.tile([-300, minimum_depth], len(boxes)),
    ]
    upper = np.r_[4.0, np.deg2rad(80), 80, np.pi / 2, np.tile([300, 400], len(boxes))]

    def residual(values, selected=None):
        positions = values[4:].reshape(-1, 2)
        predictions = projected_boxes(
            project_cuboids(image_size, values[:4], positions, dimensions)
        )
        errors = (predictions - boxes) / scales
        return errors.ravel() if selected is None else errors[selected].ravel()

    solutions = []
    for focal, pitch_degrees, heading_degrees in (
        (1.0, 25, 0),
        (1.5, 45, 0),
        (2.5, 25, 0),
        (1.0, 25, 45),
        (1.5, 45, -45),
        (2.0, 30, 85),
    ):
        pitch = np.deg2rad(pitch_degrees)
        road = camera_road_matrix(image_size, focal, pitch, 10)
        bottom = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
        positions = transform_points(np.linalg.inv(road), bottom)
        positions[:, 1] += dimensions.length_m / 2
        initial = np.r_[focal, pitch, 10, np.deg2rad(heading_degrees), positions.ravel()]
        initial = np.clip(initial, lower + 1e-5, upper - 1e-5)
        fitted = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.06,
            x_scale="jac",
            max_nfev=160,
        )
        errors = np.mean(np.abs(residual(fitted.x).reshape(-1, 4)), axis=1)
        solutions.append((float(np.median(errors)), fitted, errors))
    solutions.sort(key=lambda item: item[0])
    attempts = {
        "camera_attempts": [
            {
                "focal_ratio": float(solution.x[0]),
                "pitch_degrees": float(np.rad2deg(solution.x[1])),
                "camera_height_m": float(solution.x[2]),
                "heading_degrees": float(np.rad2deg(solution.x[3])),
                "median_relative_box_error": score,
                "optimizer_success": bool(solution.success),
            }
            for score, solution, _ in solutions
        ]
    }
    converged = [item for item in solutions if item[1].success]
    if not converged:
        raise VehicleFitError("La stima della camera non converge nel budget disponibile", attempts)
    median_error, fitted, errors = converged[0]
    inliers = errors < 0.18
    if median_error > 0.12 or inliers.sum() < 5 or inliers.mean() < 0.65:
        raise VehicleFitError(
            "Le sagome non sono compatibili con il modello di auto e un unico piano stradale",
            attempts,
        )
    # A fit at the allowed camera boundary is generally a forced, weak solution.
    fraction = (fitted.x[:3] - lower[:3]) / (upper[:3] - lower[:3])
    if (fraction < 0.003).any() or (fraction > 0.997).any():
        raise VehicleFitError(
            "La stima della camera raggiunge i limiti del modello; riferimenti insufficienti",
            attempts,
        )
    road = camera_road_matrix(image_size, *fitted.x[:3])
    homogeneous = np.c_[centers[inliers], np.ones(inliers.sum())] @ np.linalg.inv(road).T
    if (homogeneous[:, 2] <= 0).any():
        raise VehicleFitError(
            "Le osservazioni attraversano l'orizzonte del piano stradale", attempts
        )
    mapped = transform_points(np.linalg.inv(road), centers[inliers])
    distances = np.linalg.norm(mapped[1:] - mapped[:-1], axis=1)
    informative = distances > 0.5
    alternatives = []
    for score, solution, _ in solutions:
        if solution is fitted:
            continue
        if score > median_error + 0.025:
            continue
        alternative = camera_road_matrix(image_size, *solution.x[:3])
        try:
            points = transform_points(np.linalg.inv(alternative), centers[inliers])
        except ValueError:
            continue
        ratios = (
            np.linalg.norm(points[1:] - points[:-1], axis=1)[informative] / distances[informative]
        )
        if len(ratios):
            alternatives.append(float(np.median(ratios)))
    spread = max([1.0, *alternatives]) / min([1.0, *alternatives])
    if spread > 1.8:
        raise VehicleFitError(
            "Più prospettive incompatibili spiegano le auto: scala troppo ambigua",
            {**attempts, "alternative_scale_ratio": spread},
        )
    corners = project_cuboids(image_size, fitted.x[:4], fitted.x[4:].reshape(-1, 2), dimensions)
    return {
        "road_to_image": road,
        "camera": fitted.x[:4],
        "positions_m": fitted.x[4:].reshape(-1, 2),
        "corners_px": corners,
        "inliers": inliers,
        "relative_box_errors": errors,
        "median_relative_box_error": median_error,
        "alternative_scale_ratio": spread,
        "optimizer_success": bool(fitted.success),
        **attempts,
    }
