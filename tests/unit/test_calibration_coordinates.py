import numpy as np
import pytest

from cctv_incident.calibration import estimate_calibration
from cctv_incident.calibration.coordinates import (
    ImageTransform,
    full_image_roi,
    rectangle_destination,
    validate_quad,
)
from cctv_incident.calibration.homography import transform_points


def test_known_projection_recovers_independent_points_and_metric_distances():
    h = np.array([[0.02, 0.003, -1], [-0.001, 0.05, -2], [0.0001, 0.0003, 1]])
    destination = rectangle_destination(5, 12)
    source = transform_points(np.linalg.inv(h), destination)
    validate_quad(source, (1280, 720))
    calibration = estimate_calibration("camera", (1280, 720), source, destination)
    world = np.array([[1, 2], [4, 7], [2.5, 10]])
    recovered = transform_points(calibration.homography, transform_points(np.linalg.inv(h), world))
    np.testing.assert_allclose(recovered, world, atol=2e-6)
    assert np.linalg.norm(recovered[0] - recovered[1]) == pytest.approx(np.sqrt(34), abs=2e-6)


def test_resize_rounded_dimensions_preserves_world_roi_reference_and_original(calibration):
    before = calibration.model_dump()
    transform = ImageTransform.resize(calibration.image_size, (321, 181))
    adapted = transform.adapt_calibration(calibration)
    original = np.array([[34.2, 67.9], [499.4, 291.2]])
    np.testing.assert_allclose(
        transform_points(adapted.homography, transform.forward(original)),
        transform_points(calibration.homography, original),
        atol=1e-10,
    )
    np.testing.assert_allclose(transform.inverse(adapted.roi_px), calibration.roi_px, atol=1e-10)
    image = np.zeros((360, 640, 3), np.uint8)
    assert transform.adapt_reference(image).shape == (181, 321, 3)
    assert transform.adapt_motion_threshold(8) == pytest.approx(8 * 321 / 640)
    assert calibration.model_dump() == before


def test_display_margins_are_removed_before_saving_video_points(calibration):
    transform = ImageTransform((640, 360), (1000, 600), (1.25, 1.25), (100, 75))
    np.testing.assert_allclose(transform.forward([[80, 40]]), [[200, 125]])
    np.testing.assert_allclose(transform.inverse([[200, 125]]), [[80, 40]])
    adapted = transform.adapt_calibration(calibration)
    np.testing.assert_allclose(adapted.transform((200, 125)), calibration.transform((80, 40)))


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [90, 90], [90, 0], [0, 90]],
        [[0, 0], [90, 0], [20, 20], [0, 90]],
        [[0, 0], [90, 0], [90, 0], [0, 90]],
        [[0, 0], [1, 0], [2, 0], [3, 0]],
        [[0, 0], [90, 0], [90, 1e-8], [0, 1e-8]],
        [[0, 0], [90, 0], [90, float("nan")], [0, 90]],
        [[0, 0], [90, 0], [90, float("inf")], [0, 90]],
        [[0, 0], [100, 0], [100, 90], [0, 90]],
    ],
)
def test_invalid_quads_rejected(points):
    with pytest.raises(ValueError):
        validate_quad(points, (100, 100))


@pytest.mark.parametrize("width,length", [(0, 2), (-1, 2), (2, float("nan")), (float("inf"), 2)])
def test_invalid_metric_distances(width, length):
    with pytest.raises(ValueError, match="finite positive"):
        rectangle_destination(width, length)


def test_original_size_mismatch_rejected(calibration):
    with pytest.raises(ValueError, match="original size"):
        ImageTransform.resize((1280, 720), (640, 360)).adapt_calibration(calibration)
    assert full_image_roi((100, 80)) == [[0, 0], [99, 0], [99, 79], [0, 79]]
