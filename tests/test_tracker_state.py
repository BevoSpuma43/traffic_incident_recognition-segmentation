from __future__ import annotations

import numpy as np

from src.config import AppConfig
from src.models import DetectionResult
from src.tracker_state import VehicleStateStore


def _detection(
    track_id: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    confidence: float = 0.9,
) -> DetectionResult:
    polygon = np.array(
        [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]],
        dtype=np.int32,
    )
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[y1 : y2 + 1, x1 : x2 + 1] = 255
    return DetectionResult(
        track_id=track_id,
        class_id=2,
        class_name="car",
        confidence=confidence,
        polygon=polygon,
        mask=mask,
        bbox=(x1, y1, x2, y2),
    )


def test_new_track_has_no_valid_motion_or_stop_evidence() -> None:
    store = VehicleStateStore(AppConfig())

    state = store.update([_detection(1, 2, 2, 8, 8)], frame_index=0)[0]

    assert state.kinematics_valid is False
    assert state.stopped_frames == 0
    assert state.motion_anchor is not None
    assert state.motion_anchor.y == 8.0


def test_velocity_uses_bottom_center_anchor() -> None:
    config = AppConfig(velocity_ema_alpha=1.0)
    store = VehicleStateStore(config)
    store.update([_detection(1, 2, 2, 8, 8)], frame_index=0)

    state = store.update([_detection(1, 6, 2, 12, 8)], frame_index=1)[0]

    assert state.kinematics_valid is True
    assert state.velocity_x_px == 4.0
    assert state.velocity_y_px == 0.0
    assert state.speed_px == 4.0


def test_long_detection_gap_resets_kinematics_and_stop_counter() -> None:
    config = AppConfig(max_kinematic_gap_frames=1, velocity_ema_alpha=1.0)
    store = VehicleStateStore(config)
    store.update([_detection(1, 2, 2, 8, 8)], frame_index=0)
    store.update([_detection(1, 3, 2, 9, 8)], frame_index=1)

    state = store.update([_detection(1, 20, 2, 26, 8)], frame_index=4)[0]

    assert state.kinematics_valid is False
    assert state.speed_px == 0.0
    assert state.acceleration_px == 0.0
    assert state.stopped_frames == 0


def test_duplicate_track_id_keeps_highest_confidence_detection() -> None:
    store = VehicleStateStore(AppConfig())
    low = _detection(7, 2, 2, 8, 8, confidence=0.4)
    high = _detection(7, 12, 2, 18, 8, confidence=0.9)

    states = store.update([low, high], frame_index=0)

    assert len(states) == 1
    assert states[0].bbox == high.bbox
    assert states[0].confidence == 0.9
