from dataclasses import replace

import numpy as np
import pytest

from cctv_incident.calibration.homography import image_reference
from cctv_incident.features import compute_pairs
from cctv_incident.image_event_detector import ImageEventDetector
from cctv_incident.types import Motion, Observation


def vehicle(track_id, timestamp, scale=1, *, speed=100, braking=0, turning=0, x=None):
    x = (100 if track_id == 1 else 130) if x is None else x
    position = np.array([x, 120.0]) * scale
    observation = Observation(
        track_id,
        timestamp,
        position,
        tuple(position),
        np.array([x - 20, 80, x + 20, 120]) * scale,
    )
    return Motion(
        observation,
        position,
        np.array([speed, 0.0]) * scale,
        np.zeros(2),
        0,
        speed * scale,
        braking * scale,
        turning,
        2,
        0,
        100 * scale,
        1,
    )


def run_contact(config, *, scale=1, reaction=True, after="slow", step=0.1):
    detector = ImageEventDetector(config.events, config.features, "camera")
    reference = image_reference("camera", (640, 360))
    events = []
    for frame in range(round(2.5 / step)):
        timestamp = round(frame * step, 8)
        if timestamp <= 0.2:
            motions = {
                1: vehicle(1, timestamp, scale, turning=0.8 if reaction else 0),
                2: vehicle(2, timestamp, scale, speed=70, braking=100),
            }
        elif after == "missing":
            motions = {}
        else:
            # The second vehicle keeps moving. Only the first becomes occluded.
            motion = vehicle(2, timestamp, scale, speed=40)
            if after == "outside":
                motion = vehicle(2, timestamp, scale, speed=40, x=400)
            elif after == "late" and timestamp < 1.3:
                motion = vehicle(2, timestamp, scale, speed=70)
            elif after == "brief" and timestamp >= 0.6:
                motion = vehicle(2, timestamp, scale, speed=70)
            elif after == "interrupted" and timestamp == 0.5:
                motion = None
            elif after == "predicted":
                motion = replace(motion, observation=replace(motion.observation, predicted=True))
            motions = {} if motion is None else {2: motion}
        events.extend(
            detector.update(
                motions, compute_pairs(motions, config.features, "image"), timestamp, reference
            ).events
        )
    return detector, events


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_observed_slowdown_confirms_contact_despite_occluded_partner(config, scale):
    _, events = run_contact(config, scale=scale)
    assert len(events) == 1
    assert events[0].track_ids == [1, 2]
    assert events[0].impact_time_s == 0
    assert 0.7 <= events[0].confirm_time_s <= 0.8
    assert "post_impact_slowdown" in events[0].reasons
    assert "occluded_partner" in events[0].reasons
    assert "post_impact_stop" not in events[0].reasons


def test_braking_behind_a_passing_vehicle_is_insufficient(config):
    _, events = run_contact(config, reaction=False)
    assert events == []


@pytest.mark.parametrize("after", ["missing", "outside", "late", "brief", "predicted"])
def test_contact_memory_cannot_confirm_without_sustained_observation(config, after):
    detector, events = run_contact(config, after=after)
    assert events == []
    assert not detector.contacts


def test_missing_observation_restarts_slowdown_confirmation(config):
    # With a short contact lifetime, the two short visible intervals cannot be added.
    config.events.image.occlusion_confirmation_s = 0.6
    _, events = run_contact(config, after="interrupted")
    assert events == []
