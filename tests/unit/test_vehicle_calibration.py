import numpy as np
import pytest

from cctv_incident.calibration.homography import transform_points
from cctv_incident.calibration.vehicle_geometry import (
    VehicleDimensions,
    VehicleFitError,
    camera_road_matrix,
    fit_vehicle_camera,
)
from cctv_incident.calibration.vehicle_search import _road_rectangle

SIZE = (1280, 960)
POSITIONS = np.array([[-5, 18], [-4, 25], [-2, 40], [2, 21], [5, 33], [10, 60]], float)


def known_scene():
    """Independent pinhole projection of a known 3D scene, with metric probes."""
    pitch, yaw = np.deg2rad([18, 18])
    camera_rotation = np.array(
        [[1, 0, 0], [0, -np.sin(pitch), -np.cos(pitch)], [0, np.cos(pitch), -np.sin(pitch)]]
    )
    intrinsics = np.array([[1280, 0, 640], [0, 1280, 480], [0, 0, 1]])
    heading = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    offsets = (
        np.array(
            [
                [-0.9, -2.35, 0],
                [0.9, -2.35, 0],
                [0.9, 2.35, 0],
                [-0.9, 2.35, 0],
                [-0.9, -2.35, 1.5],
                [0.9, -2.35, 1.5],
                [0.9, 2.35, 1.5],
                [-0.9, 2.35, 1.5],
            ]
        )
        @ heading.T
    )
    boxes = []
    for x, y in POSITIONS:
        points = offsets + [x, y, -9]
        projected = points @ camera_rotation.T @ intrinsics.T
        projected = projected[:, :2] / projected[:, 2, None]
        boxes.append(np.r_[projected.min(axis=0), projected.max(axis=0)])
    probes = np.array([[-3, 20], [3, 20], [-3, 35], [3, 35], [0, 50]], float)
    rays = (np.c_[probes, np.full(len(probes), -9)] @ camera_rotation.T) @ intrinsics.T
    pixels = rays[:, :2] / rays[:, 2, None]
    return np.asarray(boxes), probes, pixels


def pairwise_distances(points):
    return np.linalg.norm(points[1:] - points[:-1], axis=1)


def test_3d_fit_recovers_independent_road_distances_not_box_corner_correspondences():
    boxes, probes, pixels = known_scene()
    fit = fit_vehicle_camera(boxes, SIZE)
    recovered = transform_points(np.linalg.inv(fit["road_to_image"]), pixels)
    np.testing.assert_allclose(pairwise_distances(recovered), pairwise_distances(probes), rtol=1e-4)
    assert fit["median_relative_box_error"] < 1e-5
    assert fit["inliers"].all()
    quad, span = _road_rectangle(fit, SIZE, VehicleDimensions(), None)
    # Ground patch differs from a detected silhouette box and has a metric area.
    assert not any(
        np.allclose(quad, [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]) for b in boxes
    )
    assert min(span) > 1


def test_assumed_vehicle_size_changes_metric_scale_by_the_same_factor():
    boxes, probes, pixels = known_scene()
    dimensions = VehicleDimensions(length_m=5.17, width_m=1.98, height_m=1.65)
    fit = fit_vehicle_camera(boxes, SIZE, dimensions)
    recovered = transform_points(np.linalg.inv(fit["road_to_image"]), pixels)
    np.testing.assert_allclose(
        pairwise_distances(recovered), pairwise_distances(probes) * 1.1, rtol=1e-4
    )
    assert "non una media statistica" in dimensions.source()


def test_small_detection_noise_retains_an_approximate_road_scale():
    boxes, probes, pixels = known_scene()
    fit = fit_vehicle_camera(boxes + np.random.default_rng(4).normal(0, 1, boxes.shape), SIZE)
    recovered = transform_points(np.linalg.inv(fit["road_to_image"]), pixels)
    ratios = pairwise_distances(recovered) / pairwise_distances(probes)
    assert np.max(abs(ratios - 1)) < 0.2


@pytest.mark.parametrize("case", ["few", "same_depth", "aligned", "clipped", "nan", "tiny"])
def test_insufficient_or_invalid_observations_are_rejected(case):
    boxes, _, _ = known_scene()
    if case == "few":
        boxes = boxes[:4]
    elif case == "same_depth":
        boxes = np.repeat(boxes[:1], 6, axis=0)
    elif case == "aligned":
        spans = boxes[:, 2:] - boxes[:, :2]
        centers = np.c_[np.full(6, 500), np.linspace(300, 700, 6)]
        boxes = np.c_[centers - spans / 2, centers + spans / 2]
    elif case == "clipped":
        boxes[0, 0] = -1
    elif case == "nan":
        boxes[0, 0] = np.nan
    else:
        boxes[0, 2:] = boxes[0, :2] + 5
    with pytest.raises(ValueError):
        fit_vehicle_camera(boxes, SIZE)


def test_ground_matrix_round_trip_is_metric_on_the_plane():
    road = camera_road_matrix(SIZE, 1.0, np.deg2rad(18), 9)
    points = np.array([[0, 20], [3, 20], [0, 30], [3, 30]])
    projected = transform_points(road, points)
    recovered = transform_points(np.linalg.inv(road), projected)
    np.testing.assert_allclose(recovered, points, atol=1e-9)


def test_road_patch_must_fit_the_selected_roi():
    boxes, _, _ = known_scene()
    fit = fit_vehicle_camera(boxes, SIZE)
    tiny_roi = [[0, 0], [100, 0], [100, 100], [0, 100]]
    with pytest.raises(ValueError, match="ROI"):
        _road_rectangle(fit, SIZE, VehicleDimensions(), tiny_roi)


def test_templates_validate_dimensions():
    with pytest.raises(ValueError):
        VehicleDimensions(width_m=-1)


def test_forced_perspective_from_real_car_boxes_is_rejected_with_diagnostics():
    # Local CCTV detections with varied silhouettes: an excellent box fit still
    # drives pitch to its lower bound and does not establish a metric camera.
    boxes = [
        [779.9, 661.8, 975.4, 853.5],
        [462.1, 438.0, 567.6, 522.9],
        [122.2, 846.9, 318.5, 993.8],
        [378.3, 346.4, 463.9, 424.3],
        [517.9, 457.2, 636.7, 570.4],
        [338.7, 337.1, 417.8, 395.9],
    ]
    with pytest.raises(VehicleFitError, match="limiti del modello") as rejected:
        fit_vehicle_camera(boxes, (1620, 1080))
    attempts = rejected.value.diagnostics["camera_attempts"]
    assert min(row["median_relative_box_error"] for row in attempts) < 0.05
    assert any(abs(row["pitch_degrees"] - 5) < 0.01 for row in attempts)


def test_unfinished_optimization_cannot_produce_a_calibration(monkeypatch):
    import cctv_incident.calibration.vehicle_geometry as geometry

    optimize = geometry.least_squares

    def unfinished(*args, **kwargs):
        result = optimize(*args, **kwargs)
        result.success = False
        return result

    monkeypatch.setattr(geometry, "least_squares", unfinished)
    boxes, _, _ = known_scene()
    with pytest.raises(VehicleFitError, match="non converge") as rejected:
        fit_vehicle_camera(boxes, SIZE)
    assert not any(
        row["optimizer_success"] for row in rejected.value.diagnostics["camera_attempts"]
    )
