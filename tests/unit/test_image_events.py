import numpy as np
import pytest

from cctv_incident.calibration.homography import image_reference
from cctv_incident.event_detector import EventDetector
from cctv_incident.features import compute_pairs
from cctv_incident.image_event_detector import ImageEventDetector
from cctv_incident.trajectories import TrajectoryStore
from cctv_incident.types import Observation


def simulate(config, mode="collision", scale=1):
    reference = image_reference("camera_01", (640, 360))
    store = TrajectoryStore(config.tracking, config.features, "image")
    detector = ImageEventDetector(config.events, config.features, reference.camera_id)
    events, states = [], []
    for timestamp in np.arange(0, 6, 0.1):
        travel = min(timestamp, 2) if mode == "collision" else timestamp
        positions = [100 + 50 * travel, 320 - 50 * travel]
        if mode == "stationary":
            positions = [200, 220]
        observations = []
        for index, x in enumerate(positions):
            y = 100 + (100 if index and mode == "near_miss" else 0)
            point = np.array([x, y]) * scale
            box = np.array([x - 20, y - 20, x + 20, y + 20]) * scale
            observations.append(Observation(index + 1, timestamp, point, tuple(point), box))
        motions = store.update(observations, timestamp)
        pairs = compute_pairs(motions, config.features, "image")
        decision = detector.update(motions, pairs, timestamp, reference)
        events.extend(decision.events)
        states.append(decision.state)
    return events, states


def test_image_collision_temporal_confirmation_and_resize_invariance(config):
    events, states = simulate(config)
    resized, resized_states = simulate(config, scale=0.5)
    assert len(events) == len(resized) == 1
    assert events[0].coordinate_mode == "image"
    assert events[0].calibration_confidence == 0
    assert events[0].confirm_time_s > events[0].impact_time_s >= 1.9
    assert events[0].impact_time_s == resized[0].impact_time_s
    assert states == resized_states
    assert "COOLDOWN" in states


@pytest.mark.parametrize("mode", ["stationary", "near_miss", "passing"])
def test_image_hard_negatives(config, mode):
    assert not simulate(config, mode)[0]


def test_pixel_reference_cannot_enable_metric_detector(config):
    reference = image_reference("camera_01", (640, 360))
    assert not reference.metric_valid()
    detector = EventDetector(config.events, config.features, reference.camera_id)
    assert detector.update({}, [], 0, reference).state == "PAUSED"
    reference.valid = False
    image_detector = ImageEventDetector(config.events, config.features, reference.camera_id)
    assert image_detector.update({}, [], 0, reference).state == "PAUSED"


def test_image_mode_rejects_metric_classifier():
    from cctv_incident.config import Events

    with pytest.raises(ValueError, match="metric classifier"):
        Events(coordinate_mode="image", classifier="models/metric.joblib")
