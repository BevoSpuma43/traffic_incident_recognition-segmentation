from types import SimpleNamespace

import numpy as np
import pytest

from cctv_incident.track_association import motion_costs
from cctv_incident.tracker import VehicleTracker
from cctv_incident.types import Instance


def test_fast_incoming_vehicle_keeps_its_identity(config):
    tracker = VehicleTracker(config.tracking, 8)
    black_boxes = [
        [84, 478, 253, 574],
        [100, 470, 263, 563],
        [120, 461, 277, 551],
        [139, 452, 290, 541],
        [168, 438, 310, 528],
        [199, 421, 340, 507],
        [231, 401, 381, 484],
        [276, 388, 416, 461],
        [315, 375, 449, 444],
        [339, 366, 474, 432],
        [373, 354, 507, 419],
    ]
    white_boxes = [
        None,
        None,
        [0, 597, 33, 663],
        [0, 521, 138, 640],
        [43, 471, 178, 567],
        [74, 446, 206, 527],
        [105, 417, 246, 489],
        [135, 396, 262, 466],
        [167, 377, 324, 439],
        [190, 366, 341, 422],
        [219, 350, 360, 402],
    ]
    timestamps = [3.292, 3.458, 3.542, 3.667, 3.792, 3.958, 4.042, 4.167, 4.292, 4.458, 4.542]
    white_ids = []
    for t, black, white in zip(timestamps, black_boxes, white_boxes, strict=True):
        instances = [Instance(np.array(black, float), np.ones((1, 1), bool), 2, 0.9)]
        if white is not None:
            instances.append(Instance(np.array(white, float), np.ones((1, 1), bool), 2, 0.9))
        tracks = tracker.update(instances, t, (720, 1280))
        for track in tracks:
            assert any(track.instance is instance for instance in instances)
            if white is not None and track.instance is instances[1]:
                white_ids.append(track.track_id)
    assert len(white_ids) >= 7
    assert len(set(white_ids)) == 1
    assert tracker.update([], 4.667, (720, 1280)) == []


@pytest.mark.parametrize("case", ["ambiguous", "different_class", "stale", "existing_match"])
def test_motion_fallback_rejects_unsafe_associations(case):
    track = SimpleNamespace(
        track_id=1, frame_id=2, is_activated=True, cls=2, xyxy=np.array([0, 0, 40, 40], float)
    )
    detections = [SimpleNamespace(cls=2, xyxy=np.array([30, 0, 70, 40], float))]
    costs = np.array([[0.95]])
    history = {1: [(0.0, np.array([0, 0, 40, 40], float)), (0.1, np.array([15, 0, 55, 40], float))]}
    timestamp = 0.2
    if case == "ambiguous":
        detections.append(SimpleNamespace(cls=2, xyxy=np.array([35, 0, 75, 40], float)))
        costs = np.array([[0.95, 0.95]])
    elif case == "different_class":
        detections[0].cls = 7
    elif case == "stale":
        timestamp = 0.5
    elif case == "existing_match":
        costs[0, 0] = 0.2
    result = motion_costs(costs, [track], detections, history, timestamp, 3, 0.8)
    np.testing.assert_array_equal(result, costs)
