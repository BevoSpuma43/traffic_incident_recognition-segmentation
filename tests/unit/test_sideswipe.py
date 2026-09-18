import json
from pathlib import Path

import numpy as np
import pytest

from cctv_incident.calibration.homography import image_reference
from cctv_incident.features import compute_pairs
from cctv_incident.image_event_detector import ImageEventDetector
from cctv_incident.trajectories import TrajectoryStore
from cctv_incident.types import Observation


def replay_contact(config, scale=1, variant="contact"):
    frames = json.loads(
        (Path(__file__).parents[1] / "fixtures/sideswipe-observations.json").read_text()
    )["frames"]
    detector = ImageEventDetector(config.events, config.features, "camera")
    store = TrajectoryStore(config.tracking, config.features, "image")
    reference = image_reference("camera", (1280, 720))
    events = []
    impact_box = None
    for frame in frames:
        t = frame["timestamp_s"]
        observations = []
        for row in frame["observations"]:
            key = row["track_id"]
            if variant == "missing" and key == 22 and t >= 4.667:
                continue
            box, point = np.array(row["bbox"]), np.array(row["point_px"])
            if variant == "no_contact" and key == 22:
                box[[0, 2]] -= 500
                point[0] -= 500
            if variant == "no_rotation" and key == 22:
                center = (box[:2] + box[2:]) / 2
                half_height = (box[2] - box[0]) / (2 * 1.4)
                box[[1, 3]] = center[1] + np.array([-half_height, half_height])
            if variant == "no_partner_reaction" and key == 16:
                if abs(t - 3.7916666667) < 0.01:
                    impact_box = box.copy()
                if impact_box is not None:
                    box = impact_box + np.tile(np.array([141, -85]) * (t - 3.7916666667), 2)
            observations.append(
                Observation(
                    key,
                    t,
                    point * scale,
                    tuple(point * scale),
                    box * scale,
                    row["class_id"],
                    row["confidence"],
                    row["quality"],
                    predicted=variant == "predicted" and key == 22 and t > 4,
                )
            )
        if variant == "invalid_reference":
            reference.valid = not (4.2 < t < 4.4)
        motions = store.update(observations, t)
        events.extend(
            detector.update(
                motions,
                compute_pairs(motions, config.features, "image"),
                t,
                reference,
            ).events
        )
    return events


@pytest.mark.parametrize("scale", [1, 0.5])
def test_lateral_impact_without_a_stop_is_confirmed_at_first_contact(config, scale):
    events = replay_contact(config, scale)
    assert len(events) == 1
    assert events[0].impact_time_s == pytest.approx(3.7916666667)
    assert 4.8 <= events[0].confirm_time_s < 5.2
    assert events[0].track_ids == [16, 22]
    assert "vehicle_rotation" in events[0].reasons
    assert "post_impact_stop" not in events[0].reasons


@pytest.mark.parametrize(
    "variant",
    [
        "no_contact",
        "no_rotation",
        "no_partner_reaction",
        "missing",
        "predicted",
        "invalid_reference",
    ],
)
def test_side_contact_needs_complete_observed_evidence(config, variant):
    assert replay_contact(config, variant=variant) == []


def test_side_contact_cannot_confirm_after_its_window(config):
    config.events.image.sideswipe.confirmation_window_s = 0.4
    assert replay_contact(config) == []
