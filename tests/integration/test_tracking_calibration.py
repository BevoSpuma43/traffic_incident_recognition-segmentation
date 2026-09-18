import cv2
import numpy as np

from cctv_incident.calibration.quality import CameraMotionGuard
from cctv_incident.tracker import VehicleTracker
from cctv_incident.types import Instance


def test_tracking_mask_association_and_expiry(config):
    tracker = VehicleTracker(config.tracking, 10)
    first = Instance(np.array([10, 10, 40, 40]), np.ones((100, 100), bool), 2, 0.9)
    tracks = tracker.update([first], 0, (100, 100))
    assert len(tracks) == 1 and tracks[0].instance is first
    first_id = tracks[0].track_id
    low = Instance(np.array([12, 10, 42, 40]), first.mask, 2, 0.2)
    tracks = tracker.update([low], 0.1, (100, 100))
    assert tracks[0].track_id == first_id and tracks[0].instance is low
    tracker.update([], 0.2, (100, 100))
    tracker.update([first], 3, (100, 100))
    tracks = tracker.update([first], 3.1, (100, 100))
    assert tracks[0].track_id != first_id


def test_camera_motion_invalidation():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, (320, 480, 3), dtype=np.uint8)
    guard = CameraMotionGuard(image, threshold_px=8)
    assert not guard.update(image)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 18], [0, 1, 12]]), (480, 320))
    assert not guard.update(shifted)
    assert guard.update(shifted)
