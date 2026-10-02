import json
from pathlib import Path

import numpy as np
import pytest

from cctv_incident.features import compute_pairs
from cctv_incident.sideswipe_detector import SideswipeDetector
from cctv_incident.trajectories import TrajectoryStore
from cctv_incident.types import Observation


def replay_crossing(config, profile, scale=1, variant="collision"):
    case = json.loads(
        (Path(__file__).parents[1] / "fixtures/crossing-observations.json").read_text()
    )["cases"][profile]
    detector = SideswipeDetector(config.events, config.features, "camera")
    store = TrajectoryStore(config.tracking, config.features, "image")
    moving_id = case["track_ids"][1] if profile == "rapid" else case["track_ids"][0]
    events, cooldowns = [], {}
    did_reset = False
    for frame in case["frames"]:
        t = frame["timestamp_s"]
        observations = []
        if variant == "no_approach" and t < 9:
            continue
        for row in frame["observations"]:
            box, point = np.array(row["bbox"]), np.array(row["point_px"])
            if variant == "missing" and row["track_id"] == moving_id and t > 9.25:
                continue
            if variant == "no_contact" and row["track_id"] == moving_id:
                box[[0, 2]] += 500
                point[0] += 500
            if variant in ("no_rotation", "brief_rotation"):
                if not (variant == "brief_rotation" and abs(t - 10) < 0.01):
                    cy = (box[1] + box[3]) / 2
                    half_height = (box[2] - box[0]) / 2
                    box[[1, 3]] = cy + np.array([-half_height, half_height])
            observations.append(
                Observation(
                    row["track_id"],
                    t,
                    point * scale,
                    tuple(point * scale),
                    box * scale,
                    row["class_id"],
                    row["confidence"],
                    row["quality"],
                    predicted=variant == "predicted" and row["track_id"] == moving_id and t > 9.25,
                )
            )
        if variant == "reset" and t > 9.25 and not did_reset:
            detector.reset()
            did_reset = True
        motions = store.update(observations, t)
        events.extend(
            detector.update(
                motions,
                compute_pairs(motions, config.features, "image"),
                t,
                cooldowns,
            )
        )
    return events


@pytest.mark.parametrize("profile,expected", [("rapid", 9), ("accurate", 9.0833333333)])
@pytest.mark.parametrize("scale", [1, 0.5])
def test_crossing_impact_survives_occlusion_and_is_backdated_to_contact(
    config, profile, expected, scale
):
    events = replay_crossing(config, profile, scale)
    assert len(events) == 1
    assert events[0].impact_time_s == pytest.approx(expected)
    assert 9.5 < events[0].confirm_time_s < 10.5
    assert "crossing_contact" in events[0].reasons
    assert "crossing_approach" in events[0].reasons
    assert "two_vehicle_reaction" not in events[0].reasons


@pytest.mark.parametrize(
    "variant",
    [
        "no_approach",
        "no_contact",
        "no_rotation",
        "brief_rotation",
        "missing",
        "predicted",
        "reset",
    ],
)
def test_crossing_confirmation_requires_complete_observed_evidence(config, variant):
    assert replay_crossing(config, "rapid", variant=variant) == []


def test_crossing_confirmation_expires(config):
    config.events.image.sideswipe.confirmation_window_s = 0.5
    assert replay_crossing(config, "rapid") == []
