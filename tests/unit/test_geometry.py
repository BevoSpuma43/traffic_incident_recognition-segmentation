import numpy as np
import pytest
from pydantic import ValidationError

from cctv_incident.calibration import estimate_calibration, load_calibration
from cctv_incident.calibration.assisted import propose_calibration
from cctv_incident.calibration.homography import Calibration, order_quad, transform_points
from cctv_incident.ground_point import ground_point


def test_identity_order_and_persistence(tmp_path):
    points = [[0, 0], [100, 0], [100, 100], [0, 100]]
    np.testing.assert_allclose(order_quad([points[i] for i in [2, 0, 3, 1]]), points)
    calibration = estimate_calibration("camera", (100, 100), points, points)
    np.testing.assert_allclose(calibration.transform((30, 50)), [30, 50])
    path = tmp_path / "camera.yaml"
    calibration.save(path)
    assert load_calibration(path).model_dump() == calibration.model_dump()


def test_degenerate_rejected(calibration):
    points = [[0, 0], [1, 1], [2, 2], [3, 3]]
    with pytest.raises(ValueError, match="Collinear"):
        estimate_calibration("camera", (100, 100), points, points)
    data = calibration.model_dump()
    data["homography"] = [[0, 0, 0]] * 3
    with pytest.raises(ValidationError):
        Calibration.model_validate(data)


def test_horizon_rejected():
    with pytest.raises(ValueError, match="horizon"):
        transform_points([[1, 0, 0], [0, 1, 0], [0, 1, -10]], [[1, 10]])


def test_ground_point_rejects_outlier():
    mask = np.zeros((100, 100), bool)
    mask[20:70, 30:60] = True
    mask[95, 90] = True
    point, quality = ground_point(mask, np.array([30, 20, 60, 70]))
    assert 43 <= point[0] <= 46
    assert 65 <= point[1] <= 69
    assert quality > 0.9
    assert ground_point(mask, np.array([30, 20, 60, 70]), [[0, 0], [10, 0], [10, 10]])[1] == 0


def test_no_lanes_falls_back():
    proposal, info, mask, _ = propose_calibration(np.zeros((240, 320, 3), np.uint8), "camera")
    assert proposal is None
    assert info["requires_review"]
    assert mask.sum() == 0


def test_nonmetric_disables_detector(calibration):
    calibration.units = "canonical"
    assert not calibration.metric_valid()


def test_two_line_families_produce_reviewable_proposal():
    import cv2

    image = np.full((240, 320, 3), 45, np.uint8)
    cv2.rectangle(image, (40, 100), (280, 215), (240, 240, 240), 3)
    proposal, info, _, _ = propose_calibration(image, "camera", width=10, length=20)
    assert proposal is not None, info
    assert proposal.units == "m"
    assert not proposal.accepted and not proposal.metric_valid()
    assert info["requires_review"]
