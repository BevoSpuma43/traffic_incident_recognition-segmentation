from dataclasses import replace

import numpy as np
import pytest

from cctv_incident.calibration.homography import image_reference
from cctv_incident.event_detector import EventDetector
from cctv_incident.features import compute_pairs
from cctv_incident.image_event_detector import ImageEventDetector
from cctv_incident.trajectories import TrajectoryStore
from cctv_incident.types import Motion, Observation


def test_stationary_box_jitter_does_not_count_as_an_approach(config):
    detector = ImageEventDetector(config.events, config.features, "camera")
    reference = image_reference("camera", (640, 360))
    # One old speed spike and subpixel convergence between two stopped vehicles.
    motions = {}
    for key, x in [(1, 120), (2, 150)]:
        point = np.array([x, 140.0])
        observation = Observation(key, 0, point, tuple(point), np.array([x - 20, 100, x + 20, 140]))
        velocity = np.array([2.0, 0.0]) if key == 1 else np.zeros(2)
        motions[key] = Motion(
            observation,
            point,
            velocity,
            np.zeros(2),
            0,
            float(np.linalg.norm(velocity)),
            0,
            0,
            5,
            1,
            40 if key == 1 else 0,
            1,
        )
    events = []
    for timestamp in np.arange(0, 1.5, 0.1):
        current = {
            key: replace(motion, observation=replace(motion.observation, timestamp_s=timestamp))
            for key, motion in motions.items()
        }
        events.extend(
            detector.update(
                current, compute_pairs(current, config.features, "image"), timestamp, reference
            ).events
        )
    assert events == []


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_collision_between_different_sized_vehicles(config, scale):
    detector = ImageEventDetector(config.events, config.features, "camera")
    reference = image_reference("camera", (640, 360))
    store = TrajectoryStore(config.tracking, config.features, "image")
    events = []
    for timestamp in np.arange(0, 6, 0.1):
        travel = min(timestamp, 2)
        observations = []
        for key, x, half_width, half_height in [
            (1, 100 + 50 * travel, 100, 50),
            (2, 410 - 50 * travel, 20, 20),
        ]:
            point = np.array([x, 100.0]) * scale
            box = (
                np.array([x - half_width, 100 - half_height, x + half_width, 100 + half_height])
                * scale
            )
            observations.append(Observation(key, timestamp, point, tuple(point), box))
        motions = store.update(observations, timestamp)
        pairs = compute_pairs(motions, config.features, "image")
        # IoU remains below the original threshold even at contact.
        if timestamp >= 2:
            assert all(pair.bbox_iou < config.events.image.min_bbox_iou for pair in pairs)
        events.extend(detector.update(motions, pairs, timestamp, reference).events)
    assert len(events) == 1
    assert 2 <= events[0].impact_time_s < 2.3
    assert events[0].track_ids == [1, 2]


def test_late_impact_has_time_for_post_impact_confirmation(config):
    detector = EventDetector(config.events, config.features, "camera")
    key = (1, 2)
    for timestamp in np.arange(0, 2.9, 0.1):
        assert not detector._advance(
            {key: (0.8, True, False, False, {"approach"})}, timestamp, 1
        ).events
    detector._advance({key: (1, True, True, False, {"deceleration"})}, 2.9, 1)
    events = []
    for timestamp in np.arange(3.0, 3.7, 0.1):
        events.extend(
            detector._advance({key: (1, False, False, True, {"stop"})}, timestamp, 1).events
        )
    assert len(events) == 1
    assert events[0].impact_time_s == 2.9
    assert events[0].confirm_time_s > config.events.candidate_timeout_s


def test_impact_without_confirmation_still_expires(config):
    detector = EventDetector(config.events, config.features, "camera")
    key = (1, 2)
    for timestamp in np.arange(0, 7, 0.1):
        evidence = {key: (0.8, timestamp < 0.4, timestamp == 0.2, False, {"approach"})}
        assert not detector._advance(evidence, timestamp, 1).events
    assert not detector.candidates


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_ground_point_jump_inside_stationary_boxes_cannot_create_collision(config, scale):
    detector = ImageEventDetector(config.events, config.features, "camera")
    reference = image_reference("camera", (640, 360))
    store = TrajectoryStore(config.tracking, config.features, "image")
    events = []
    peak_speed = peak_deceleration = 0
    for timestamp in np.arange(0, 4, 0.1):
        observations = []
        for key, x in [(1, 200), (2, 220)]:
            # A mask edge moves by 30 px, but both vehicle boxes remain stationary.
            point_x = x + (-15 if timestamp < 1 else 15) if key == 1 else x
            point = np.array([point_x, 120.0]) * scale
            box = np.array([x - 20, 100, x + 20, 140]) * scale
            observations.append(Observation(key, timestamp, point, tuple(point), box))
        motions = store.update(observations, timestamp)
        peak_speed = max(peak_speed, motions[1].speed / scale)
        peak_deceleration = max(peak_deceleration, motions[1].deceleration / scale)
        events.extend(
            detector.update(
                motions, compute_pairs(motions, config.features, "image"), timestamp, reference
            ).events
        )
    # Exercise a large false approach and braking impulse, not just subpixel jitter.
    assert peak_speed > 50
    assert peak_deceleration > 50
    assert events == []
