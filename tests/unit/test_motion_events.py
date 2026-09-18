import numpy as np
import pytest
from conftest import observation

from cctv_incident.event_detector import EventDetector
from cctv_incident.features import closest_approach, compute_pairs
from cctv_incident.trajectories import TrajectoryStore


def test_derivatives_irregular_timestamps(config):
    store = TrajectoryStore(config.tracking, config.features)
    for timestamp in [0, 0.11, 0.28, 0.5, 0.75]:
        motion = store.update([observation(1, timestamp, 3 * timestamp)], timestamp)[1]
    assert motion.speed == pytest.approx(3)
    np.testing.assert_allclose(motion.acceleration, [0, 0], atol=1e-10)
    assert store.update([observation(1, 0.75, 400)], 0.75) == {}
    motion = store.update([observation(1, 2.0, 50)], 2.0)[1]
    assert motion.quality == 0 and motion.speed == 0


def test_ttc():
    t, d, ttc, converging = closest_approach([10, 0], [-2, 0], 2)
    assert (t, d, ttc, converging) == (5, 0, 4, True)
    assert closest_approach([10, 0], [2, 0], 2)[2] is None
    assert closest_approach([10, 0], [0, 0], 2)[2] is None
    assert closest_approach([10, 10], [-2, 0], 2)[2] is None


def simulation(config, calibration, mode="collision"):
    store = TrajectoryStore(config.tracking, config.features)
    detector = EventDetector(config.events, config.features, calibration.camera_id)
    events, states = [], []
    for timestamp in np.arange(0, 7, 0.1):
        travel = min(timestamp, 2) if mode in ("collision", "poor_quality") else timestamp
        x1, x2 = 5 + 4.5 * travel, 25 - 4.5 * travel
        if mode == "stationary":
            x1, x2 = 14, 16
        y = 5 if mode == "near_miss" else 0
        quality = 0.1 if mode == "poor_quality" else 1
        observations = [
            observation(1, timestamp, x1, quality=quality),
            observation(2, timestamp, x2, y, quality=quality),
        ]
        motions = store.update(observations, timestamp)
        pairs = compute_pairs(motions, config.features)
        decision = detector.update(motions, pairs, timestamp, calibration)
        events.extend(decision.events)
        states.append(decision.state)
    return events, states


def test_collision_confirmed_once(config, calibration):
    events, states = simulation(config, calibration)
    assert len(events) == 1
    assert events[0].impact_time_s >= 1.9
    assert events[0].confirm_time_s > events[0].impact_time_s
    assert "post_impact_stop" in events[0].reasons
    assert "COOLDOWN" in states


@pytest.mark.parametrize("mode", ["near_miss", "stationary", "passing", "poor_quality"])
def test_hard_negatives(config, calibration, mode):
    assert simulation(config, calibration, mode)[0] == []


def test_invalid_calibration_pauses(config, calibration):
    calibration.valid = False
    events, states = simulation(config, calibration)
    assert not events and set(states) == {"PAUSED"}


def test_bounded_histories(config):
    config.tracking.max_tracks = 10
    config.tracking.max_samples = 12
    store = TrajectoryStore(config.tracking, config.features)
    for index in range(1000):
        timestamp = index / 10
        store.update([observation(index, timestamp, 0)], timestamp)
        assert len(store.histories) <= 10
        assert all(len(history.motions) <= 12 for history in store.histories.values())
