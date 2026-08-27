from __future__ import annotations

from collections import deque

import numpy as np

from src.collision_logic import CollisionDetector
from src.models import Point2D, TrackSample, VehicleState


def _make_mask(
    top: int,
    left: int,
    bottom: int,
    right: int,
    shape: tuple[int, int] = (20, 20),
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _state(
    track_id: int,
    mask: np.ndarray,
    frame_index: int,
    *,
    speed: float,
    acceleration: float = 0.0,
    stopped_frames: int = 0,
    velocity_x: float = 0.0,
    history_speeds: tuple[float, ...] = (),
) -> VehicleState:
    bbox = _bbox(mask)
    anchor = Point2D((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))
    history: deque[TrackSample] = deque(maxlen=15)
    first_frame = frame_index - len(history_speeds) + 1
    for offset, sample_speed in enumerate(history_speeds):
        history.append(
            TrackSample(
                frame_index=first_frame + offset,
                centroid=anchor,
                motion_anchor=anchor,
                speed_px=sample_speed,
                velocity_x_px=velocity_x,
                velocity_y_px=0.0,
                acceleration_px=0.0,
                mask_area=int(np.count_nonzero(mask)),
            )
        )
    return VehicleState(
        track_id=track_id,
        class_id=2,
        class_name="car",
        centroid=anchor,
        prev_centroid=anchor,
        speed_px=speed,
        prev_speed_px=max(speed, speed - acceleration),
        acceleration_px=acceleration,
        polygon=None,
        mask=mask,
        bbox=bbox,
        last_seen_frame=frame_index,
        stopped_frames=stopped_frames,
        motion_anchor=anchor,
        prev_motion_anchor=anchor,
        velocity_x_px=velocity_x,
        velocity_y_px=0.0,
        kinematics_valid=True,
        observed_frames=max(2, len(history_speeds)),
        confidence=0.9,
        history=history,
    )


def _detector(**overrides: object) -> CollisionDetector:
    values: dict[str, object] = {
        "mask_overlap_threshold": 4,
        "mask_overlap_ratio_threshold": 0.01,
        "stopped_frames_threshold": 3,
        "stopped_speed_threshold": 2.5,
        "strong_deceleration_threshold": -4.0,
        "contact_distance_threshold_px": 2.0,
        "mask_dilation_pixels": 1,
        "min_preimpact_speed_px": 3.0,
        "impact_window_frames": 3,
        "collision_confirmation_frames": 2,
        "collision_cooldown_frames": 10,
    }
    values.update(overrides)
    return CollisionDetector(**values)


def test_temporal_contact_is_confirmed_once() -> None:
    detector = _detector()
    mask_a = _make_mask(2, 2, 9, 9)
    mask_b = _make_mask(6, 6, 13, 13)

    first = detector.detect_collisions(
        [
            _state(1, mask_a, 0, speed=0.0, acceleration=-5.0, history_speeds=(6.0, 0.0)),
            _state(2, mask_b, 0, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        frame_index=0,
    )
    confirmed = detector.detect_collisions(
        [
            _state(1, mask_a, 1, speed=0.0, history_speeds=(6.0, 0.0, 0.0)),
            _state(2, mask_b, 1, speed=0.0, history_speeds=(0.0, 0.0, 0.0)),
        ],
        frame_index=1,
    )
    duplicate = detector.detect_collisions(
        [
            _state(1, mask_a, 2, speed=0.0, history_speeds=(6.0, 0.0, 0.0)),
            _state(2, mask_b, 2, speed=0.0, history_speeds=(0.0, 0.0, 0.0)),
        ],
        frame_index=2,
    )

    assert first == []
    assert len(confirmed) == 1
    assert (confirmed[0].track_id_a, confirmed[0].track_id_b) == (1, 2)
    assert confirmed[0].speed_a == 0.0
    assert confirmed[0].overlap_ratio > 0.0
    assert duplicate == []


def test_stationary_overlapping_vehicles_are_not_a_collision() -> None:
    detector = _detector()
    mask_a = _make_mask(2, 2, 9, 9)
    mask_b = _make_mask(6, 6, 13, 13)

    events = []
    for frame_index in range(6):
        events.extend(
            detector.detect_collisions(
                [
                    _state(
                        10,
                        mask_a,
                        frame_index,
                        speed=0.0,
                        stopped_frames=6,
                        history_speeds=(0.0, 0.0, 0.0),
                    ),
                    _state(
                        11,
                        mask_b,
                        frame_index,
                        speed=0.0,
                        stopped_frames=6,
                        history_speeds=(0.0, 0.0, 0.0),
                    ),
                ],
                frame_index,
            )
        )

    assert events == []


def test_touching_masks_do_not_need_exact_overlap() -> None:
    detector = _detector(collision_confirmation_frames=1)
    mask_a = _make_mask(2, 2, 8, 5)
    mask_b = _make_mask(2, 5, 8, 8)

    events = detector.detect_collisions(
        [
            _state(
                20,
                mask_a,
                0,
                speed=0.0,
                acceleration=-5.0,
                velocity_x=5.0,
                history_speeds=(5.0, 0.0),
            ),
            _state(21, mask_b, 0, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        0,
    )

    assert len(events) == 1
    assert events[0].overlap_area == 0


def test_same_track_id_can_never_collide_with_itself() -> None:
    detector = _detector(collision_confirmation_frames=1)
    mask = _make_mask(2, 2, 9, 9)
    state = _state(30, mask, 0, speed=0.0, acceleration=-5.0, history_speeds=(5.0, 0.0))

    assert detector.detect_collisions([state, state], 0) == []


def test_dynamic_anomaly_without_contact_is_not_a_collision() -> None:
    detector = _detector(collision_confirmation_frames=1)
    mask_a = _make_mask(1, 1, 4, 4)
    mask_b = _make_mask(14, 14, 18, 18)

    events = detector.detect_collisions(
        [
            _state(40, mask_a, 0, speed=0.0, acceleration=-6.0, history_speeds=(7.0, 0.0)),
            _state(41, mask_b, 0, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        0,
    )

    assert events == []


def test_dynamic_anomaly_and_contact_may_occur_in_neighboring_frames() -> None:
    detector = _detector(collision_confirmation_frames=1)
    far_a = _make_mask(2, 2, 8, 8)
    far_b = _make_mask(2, 12, 8, 18)
    contact_b = _make_mask(2, 6, 8, 12)

    before_contact = detector.detect_collisions(
        [
            _state(
                50,
                far_a,
                0,
                speed=0.0,
                acceleration=-5.0,
                velocity_x=5.0,
                history_speeds=(6.0, 0.0),
            ),
            _state(51, far_b, 0, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        0,
    )
    at_contact = detector.detect_collisions(
        [
            _state(50, far_a, 1, speed=0.0, history_speeds=(6.0, 0.0, 0.0)),
            _state(51, contact_b, 1, speed=0.0, history_speeds=(0.0, 0.0, 0.0)),
        ],
        1,
    )

    assert before_contact == []
    assert len(at_contact) == 1
