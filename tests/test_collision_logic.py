from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from src.collision_logic import CollisionDetector
from src.kinematics import compute_bbox_scale_px
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
    velocity_y: float = 0.0,
    history_speeds: tuple[float, ...] = (),
    history_anchors: tuple[tuple[float, float], ...] = (),
) -> VehicleState:
    bbox = _bbox(mask)
    anchor = Point2D((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))
    history: deque[TrackSample] = deque(maxlen=15)
    first_frame = frame_index - len(history_speeds) + 1
    for offset, sample_speed in enumerate(history_speeds):
        sample_anchor = (
            Point2D(*history_anchors[offset])
            if offset < len(history_anchors)
            else anchor
        )
        history.append(
            TrackSample(
                frame_index=first_frame + offset,
                centroid=sample_anchor,
                motion_anchor=sample_anchor,
                speed_px=sample_speed,
                velocity_x_px=velocity_x,
                velocity_y_px=velocity_y,
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
        prev_speed_px=max(0.0, speed - acceleration),
        acceleration_px=acceleration,
        polygon=None,
        mask=mask,
        bbox=bbox,
        last_seen_frame=frame_index,
        stopped_frames=stopped_frames,
        motion_anchor=anchor,
        prev_motion_anchor=anchor,
        velocity_x_px=velocity_x,
        velocity_y_px=velocity_y,
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
        "motion_confirmation_frames": 1,
        "approach_confirmation_frames": 1,
        "approach_evidence_window_frames": 3,
        "impact_window_frames": 3,
        "impact_reaction_confirmation_frames": 1,
        "max_contact_candidate_age_frames": 0,
        "collision_confirmation_frames": 2,
        "collision_cooldown_frames": 10,
        "preexisting_contact_release_frames": 0,
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


def test_preexisting_overlap_stays_disarmed_while_queue_is_stationary() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        motion_confirmation_frames=2,
        preexisting_contact_release_frames=3,
    )
    mask_a = _make_mask(2, 2, 9, 9)
    mask_b = _make_mask(6, 6, 13, 13)
    events = []

    for frame_index in range(20):
        events.extend(
            detector.detect_collisions(
                [
                    _state(
                        60,
                        mask_a,
                        frame_index,
                        speed=0.0,
                        acceleration=-6.0,
                        stopped_frames=10,
                        history_speeds=(4.0, 4.0, 0.0),
                    ),
                    _state(
                        61,
                        mask_b,
                        frame_index,
                        speed=0.0,
                        stopped_frames=10,
                        history_speeds=(0.0, 0.0, 0.0),
                    ),
                ],
                frame_index,
            )
        )

    assert events == []
    assert detector.last_diagnostics[0].preexisting_contact is True
    assert detector.last_diagnostics[0].pair_status == "preexisting_contact"


def test_preexisting_overlap_arms_only_after_stable_separation() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        motion_confirmation_frames=2,
        preexisting_contact_release_frames=3,
    )
    mask_a = _make_mask(2, 2, 8, 8)
    contact_b = _make_mask(2, 6, 8, 12)
    far_b = _make_mask(2, 12, 8, 18)

    initial = detector.detect_collisions(
        [
            _state(70, mask_a, 0, speed=0.0, history_speeds=(0.0, 0.0)),
            _state(71, contact_b, 0, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        0,
    )
    separated_events = []
    for frame_index in range(1, 4):
        separated_events.extend(
            detector.detect_collisions(
                [
                    _state(70, mask_a, frame_index, speed=5.0, history_speeds=(5.0, 5.0)),
                    _state(71, far_b, frame_index, speed=0.0, history_speeds=(0.0, 0.0)),
                ],
                frame_index,
            )
        )

    impact = detector.detect_collisions(
        [
            _state(
                70,
                mask_a,
                4,
                speed=0.0,
                acceleration=-6.0,
                velocity_x=5.0,
                history_speeds=(5.0, 5.0, 0.0),
            ),
            _state(71, contact_b, 4, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        4,
    )

    assert initial == []
    assert separated_events == []
    assert len(impact) == 1


def test_pair_with_one_mature_track_is_not_treated_as_preexisting_contact() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        preexisting_contact_release_frames=3,
    )
    mover_mask = _make_mask(2, 2, 9, 9)
    reborn_mask = _make_mask(6, 6, 13, 13)

    events = detector.detect_collisions(
        [
            _state(
                200,
                mover_mask,
                0,
                speed=0.0,
                acceleration=-5.0,
                velocity_x=5.0,
                history_speeds=(6.0,) * 11 + (0.0,),
            ),
            _state(201, reborn_mask, 0, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        0,
    )

    assert len(events) == 1
    assert detector.last_diagnostics[0].preexisting_contact is False


def test_pair_of_two_new_tracks_still_arms_preexisting_contact() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        preexisting_contact_release_frames=3,
    )
    mask_a = _make_mask(2, 2, 9, 9)
    mask_b = _make_mask(6, 6, 13, 13)

    events = detector.detect_collisions(
        [
            _state(
                210,
                mask_a,
                0,
                speed=0.0,
                acceleration=-5.0,
                velocity_x=5.0,
                history_speeds=(6.0, 0.0),
            ),
            _state(211, mask_b, 0, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        0,
    )

    assert events == []
    assert detector.last_diagnostics[0].preexisting_contact is True


def test_preexisting_contact_is_released_after_the_timeout() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        preexisting_contact_release_frames=3,
        preexisting_contact_max_frames=5,
    )
    mask_a = _make_mask(2, 2, 9, 9)
    mask_b = _make_mask(6, 6, 13, 13)

    # La coppia nasce gia a contatto e non si separa mai: senza scadenza
    # resterebbe disarmata per tutta la durata del video.
    disarmed_events = []
    for frame_index in range(5):
        disarmed_events.extend(
            detector.detect_collisions(
                [
                    _state(
                        220,
                        mask_a,
                        frame_index,
                        speed=0.0,
                        stopped_frames=6,
                        history_speeds=(0.0, 0.0),
                    ),
                    _state(
                        221,
                        mask_b,
                        frame_index,
                        speed=0.0,
                        stopped_frames=6,
                        history_speeds=(0.0, 0.0),
                    ),
                ],
                frame_index,
            )
        )
        if frame_index == 3:
            assert detector.last_diagnostics[0].preexisting_contact is True

    assert disarmed_events == []
    assert detector.last_diagnostics[0].preexisting_contact is False

    impact = detector.detect_collisions(
        [
            _state(
                220,
                mask_a,
                5,
                speed=0.0,
                acceleration=-5.0,
                velocity_x=5.0,
                history_speeds=(6.0, 0.0),
            ),
            _state(221, mask_b, 5, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        5,
    )

    assert len(impact) == 1


def test_normalization_is_inert_at_reference_scale_and_fps() -> None:
    """A condizioni di riferimento le soglie devono restare identiche."""
    mask = _make_mask(2, 2, 9, 9)
    bbox = _bbox(mask)
    reference_scale = ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5

    plain = _detector(collision_confirmation_frames=1)
    normalized = _detector(
        collision_confirmation_frames=1,
        kinematic_normalization_enabled=True,
        reference_scale_px=reference_scale,
        reference_fps=25.0,
        video_fps=25.0,
    )

    assert normalized._window(6) == 6
    assert normalized._speed_threshold(3.0, _state(1, mask, 0, speed=0.0)) == 3.0

    other = _make_mask(6, 6, 13, 13)
    states = lambda: [
        _state(1, mask, 0, speed=0.0, acceleration=-5.0, history_speeds=(6.0, 0.0)),
        _state(2, other, 0, speed=0.0, history_speeds=(0.0, 0.0)),
    ]
    assert len(plain.detect_collisions(states(), 0)) == len(
        normalized.detect_collisions(states(), 0)
    )


def test_distant_vehicle_keeps_a_proportionally_lower_speed_threshold() -> None:
    """Un veicolo lontano si muove di pochi pixel: la soglia deve scendere."""
    detector = _detector(
        kinematic_normalization_enabled=True,
        reference_scale_px=90.0,
        reference_fps=15.0,
        video_fps=15.0,
    )
    near = _state(1, _make_mask(0, 0, 19, 19), 0, speed=0.0)
    far = _state(2, _make_mask(8, 8, 11, 11), 0, speed=0.0)

    near_threshold = detector._speed_threshold(3.0, near)
    far_threshold = detector._speed_threshold(3.0, far)

    assert far_threshold < near_threshold
    # Il rapporto fra le soglie segue il rapporto fra le diagonali delle bbox.
    assert far_threshold / near_threshold == pytest.approx(
        compute_bbox_scale_px(far.bbox) / compute_bbox_scale_px(near.bbox)
    )


def test_higher_frame_rate_lowers_speeds_and_stretches_windows() -> None:
    """A fps doppio lo stesso moto fisico produce meta pixel per frame."""
    detector = _detector(
        kinematic_normalization_enabled=True,
        normalize_time_windows=True,
        reference_scale_px=10.0,
        reference_fps=15.0,
        video_fps=30.0,
    )
    vehicle = _state(1, _make_mask(2, 2, 9, 9), 0, speed=0.0)

    # Le soglie di velocita si dimezzano, quelle di accelerazione si riducono
    # di un fattore quattro, le finestre in frame raddoppiano.
    assert detector._speed_threshold(3.0, vehicle) == pytest.approx(
        1.5 * detector._vehicle_scale_factor(vehicle)
    )
    assert detector._acceleration_threshold(-4.0, vehicle) == pytest.approx(
        -1.0 * detector._vehicle_scale_factor(vehicle)
    )
    assert detector._window(5) == 10
    assert detector._window(3) == 6
    # Lo zero conserva il significato di "controllo disattivato".
    assert detector._window(0) == 0


def test_time_windows_are_not_dilated_unless_explicitly_enabled() -> None:
    """Le magnitudini si normalizzano, le finestre no: default misurato."""
    detector = _detector(
        kinematic_normalization_enabled=True,
        reference_scale_px=10.0,
        reference_fps=15.0,
        video_fps=30.0,
    )
    vehicle = _state(1, _make_mask(2, 2, 9, 9), 0, speed=0.0)

    assert detector._window(5) == 5
    assert detector._window(3) == 3
    # La normalizzazione delle magnitudini resta comunque attiva.
    assert detector._speed_threshold(3.0, vehicle) < 3.0


def test_single_motion_spike_does_not_confirm_preimpact_motion() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        motion_confirmation_frames=2,
    )
    mask_a = _make_mask(2, 2, 8, 8)
    far_b = _make_mask(2, 12, 8, 18)
    contact_b = _make_mask(2, 6, 8, 12)
    detector.detect_collisions(
        [
            _state(80, mask_a, 0, speed=0.0, history_speeds=(0.0, 0.0)),
            _state(81, far_b, 0, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        0,
    )

    events = detector.detect_collisions(
        [
            _state(
                80,
                mask_a,
                1,
                speed=0.0,
                acceleration=-6.0,
                velocity_x=5.0,
                history_speeds=(0.0, 7.0, 0.0),
            ),
            _state(81, contact_b, 1, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        1,
    )

    assert events == []
    assert detector.last_diagnostics[0].had_recent_motion is False


def test_parallel_slow_traffic_with_overlap_is_not_a_collision() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        motion_confirmation_frames=2,
        approach_confirmation_frames=3,
        max_contact_candidate_age_frames=5,
    )
    mask_a = _make_mask(2, 2, 9, 9)
    far_b = _make_mask(2, 12, 9, 19)
    overlap_b = _make_mask(6, 6, 13, 13)
    detector.detect_collisions(
        [
            _state(90, mask_a, 0, speed=2.0, velocity_x=2.0, history_speeds=(2.0, 2.0)),
            _state(91, far_b, 0, speed=2.0, velocity_x=2.0, history_speeds=(2.0, 2.0)),
        ],
        0,
    )

    events = []
    for frame_index in range(1, 9):
        events.extend(
            detector.detect_collisions(
                [
                    _state(
                        90,
                        mask_a,
                        frame_index,
                        speed=0.0,
                        velocity_x=2.0,
                        history_speeds=(4.0, 4.0, 0.0),
                    ),
                    _state(
                        91,
                        overlap_b,
                        frame_index,
                        speed=0.0,
                        velocity_x=2.0,
                        history_speeds=(4.0, 4.0, 0.0),
                    ),
                ],
                frame_index,
            )
        )

    assert events == []


def test_late_anomaly_during_persistent_contact_is_ignored() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        max_contact_candidate_age_frames=3,
    )
    mask_a = _make_mask(2, 2, 8, 8)
    far_b = _make_mask(2, 12, 8, 18)
    contact_b = _make_mask(2, 6, 8, 12)
    detector.detect_collisions(
        [
            _state(100, mask_a, 0, speed=0.0, history_speeds=(0.0, 0.0)),
            _state(101, far_b, 0, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        0,
    )
    for frame_index in range(1, 7):
        detector.detect_collisions(
            [
                _state(100, mask_a, frame_index, speed=5.0, velocity_x=5.0, history_speeds=(5.0, 5.0)),
                _state(101, contact_b, frame_index, speed=0.0, history_speeds=(0.0, 0.0)),
            ],
            frame_index,
        )

    events = detector.detect_collisions(
        [
            _state(
                100,
                mask_a,
                7,
                speed=0.0,
                acceleration=-6.0,
                velocity_x=5.0,
                history_speeds=(5.0, 5.0, 0.0),
            ),
            _state(101, contact_b, 7, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        7,
    )

    assert events == []
    assert detector.last_diagnostics[0].contact_age_frames > 3


def test_sustained_convergence_near_contact_can_still_emit() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        motion_confirmation_frames=2,
        approach_confirmation_frames=3,
        approach_evidence_window_frames=1,
        max_contact_candidate_age_frames=5,
    )
    mask_a = _make_mask(2, 2, 8, 8)
    far_b = _make_mask(2, 12, 8, 18)
    contact_b = _make_mask(2, 6, 8, 12)
    detector.detect_collisions(
        [
            _state(110, mask_a, 0, speed=5.0, velocity_x=5.0, history_speeds=(5.0, 5.0)),
            _state(111, far_b, 0, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        0,
    )

    events = []
    for frame_index in range(1, 4):
        events.extend(
            detector.detect_collisions(
                [
                    _state(
                        110,
                        mask_a,
                        frame_index,
                        speed=0.0 if frame_index == 3 else 5.0,
                        acceleration=-6.0 if frame_index == 3 else 0.0,
                        velocity_x=5.0,
                        history_speeds=(5.0, 5.0, 0.0),
                    ),
                    _state(111, contact_b, frame_index, speed=0.0, history_speeds=(0.0, 0.0)),
                ],
                frame_index,
            )
        )

    assert len(events) == 1


def test_single_frame_braking_during_turn_past_stationary_vehicle_is_ignored() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        impact_reaction_confirmation_frames=2,
        stationary_history_frames=5,
        trajectory_history_frames=6,
        trajectory_reaction_lag_frames=1,
    )
    mover_mask = _make_mask(2, 2, 9, 9)
    target_mask = _make_mask(2, 7, 9, 14)
    straight_history = tuple((float(x), 8.0) for x in range(6))

    first = detector.detect_collisions(
        [
            _state(
                120,
                mover_mask,
                0,
                speed=5.0,
                acceleration=-6.0,
                velocity_x=5.0,
                history_speeds=(5.0,) * 6,
                history_anchors=straight_history,
            ),
            _state(121, target_mask, 0, speed=0.0, history_speeds=(0.0,) * 6),
        ],
        0,
    )
    second = detector.detect_collisions(
        [
            _state(
                120,
                mover_mask,
                1,
                speed=5.0,
                velocity_x=5.0,
                history_speeds=(5.0,) * 6,
                history_anchors=straight_history,
            ),
            _state(121, target_mask, 1, speed=0.0, history_speeds=(0.0,) * 6),
        ],
        1,
    )

    assert first == []
    assert second == []
    assert detector.last_diagnostics[0].reaction_frames == 0


def test_late_stationary_target_impact_uses_trajectory_deflection() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        impact_reaction_confirmation_frames=2,
        stationary_history_frames=5,
        trajectory_history_frames=6,
        trajectory_reaction_lag_frames=1,
        trajectory_min_displacement_px=3.0,
        trajectory_deflection_angle_deg=45.0,
        max_contact_candidate_age_frames=2,
    )
    mover_mask = _make_mask(2, 2, 9, 9)
    target_mask = _make_mask(2, 7, 9, 14)
    straight_history = tuple((float(x), 8.0) for x in range(6))

    events = []
    for frame_index in range(5):
        events.extend(
            detector.detect_collisions(
                [
                    _state(
                        130,
                        mover_mask,
                        frame_index,
                        speed=5.0,
                        velocity_x=5.0,
                        history_speeds=(5.0,) * 6,
                        history_anchors=straight_history,
                    ),
                    _state(
                        131,
                        target_mask,
                        frame_index,
                        speed=0.0,
                        history_speeds=(0.0,) * 6,
                    ),
                ],
                frame_index,
            )
        )

    events.extend(
        detector.detect_collisions(
            [
                _state(
                    130,
                    mover_mask,
                    5,
                    speed=5.0,
                    acceleration=-6.0,
                    velocity_x=5.0,
                    history_speeds=(5.0,) * 6,
                    history_anchors=straight_history,
                ),
                _state(131, target_mask, 5, speed=0.0, history_speeds=(0.0,) * 6),
            ],
            5,
        )
    )
    events.extend(
        detector.detect_collisions(
            [
                _state(
                    130,
                    mover_mask,
                    6,
                    speed=5.0,
                    velocity_y=5.0,
                    history_speeds=(5.0,) * 6,
                    history_anchors=straight_history,
                ),
                _state(131, target_mask, 6, speed=0.0, history_speeds=(0.0,) * 6),
            ],
            6,
        )
    )

    assert len(events) == 1
    assert events[0].frame_index == 6
    assert "trajectory_deflection" in events[0].reason
    assert events[0].first_contact_frame == 0


def test_crossing_overlap_without_pair_disruption_is_ignored() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        crossing_history_frames=6,
        crossing_min_angle_deg=45.0,
    )
    vertical = _make_mask(2, 2, 10, 9)
    horizontal = _make_mask(6, 6, 13, 15)
    vertical_history = tuple((5.0, float(y)) for y in range(13, 7, -1))
    horizontal_history = tuple((float(x), 12.0) for x in range(15, 9, -1))

    events = detector.detect_collisions(
        [
            _state(
                140,
                vertical,
                0,
                speed=5.0,
                velocity_y=-5.0,
                history_speeds=(5.0,) * 6,
                history_anchors=vertical_history,
            ),
            _state(
                141,
                horizontal,
                0,
                speed=10.0,
                acceleration=-6.0,
                velocity_x=-10.0,
                history_speeds=(10.0,) * 6,
                history_anchors=horizontal_history,
            ),
        ],
        0,
    )

    assert events == []
    assert detector.last_diagnostics[0].crossing_trajectories is True
    assert detector.last_diagnostics[0].target_impulse is False


def test_crossing_target_impulse_confirms_the_impacted_pair() -> None:
    detector = _detector(
        collision_confirmation_frames=2,
        approach_confirmation_frames=3,
        crossing_history_frames=6,
        crossing_min_angle_deg=45.0,
        target_impulse_acceleration_threshold=8.0,
    )
    target = _make_mask(2, 2, 10, 9)
    mover = _make_mask(6, 6, 13, 15)
    target_history = ((5.0, 8.0), (5.0, 5.0), (5.0, 20.0))
    mover_history = tuple((float(x), 12.0) for x in range(15, 9, -1))

    events = detector.detect_collisions(
        [
            _state(
                150,
                target,
                0,
                speed=20.0,
                acceleration=15.0,
                velocity_y=20.0,
                history_speeds=(5.0, 5.0, 20.0),
                history_anchors=target_history,
            ),
            _state(
                151,
                mover,
                0,
                speed=10.0,
                velocity_x=-10.0,
                history_speeds=(10.0,) * 6,
                history_anchors=mover_history,
            ),
        ],
        0,
    )

    assert len(events) == 1
    assert "target_impulse" in events[0].reason
    assert detector.last_diagnostics[0].crossing_trajectories is True
    assert detector.last_diagnostics[0].target_impulse is True


def test_crossing_dual_stop_after_tracking_gap_is_confirmed() -> None:
    detector = _detector(
        collision_confirmation_frames=2,
        crossing_history_frames=6,
        crossing_dual_stop_bridge_frames=12,
        crossing_dual_stop_min_gap_frames=2,
    )
    approach_a = _make_mask(1, 3, 7, 8)
    approach_b = _make_mask(8, 10, 14, 16)
    contact_a = _make_mask(4, 4, 12, 11)
    contact_b = _make_mask(7, 8, 15, 15)
    vertical_history = ((5.0, 3.0), (5.0, 5.0), (5.0, 7.0))
    horizontal_history = ((16.0, 12.0), (14.0, 12.0), (12.0, 12.0))

    assert detector.detect_collisions(
        [
            _state(
                170,
                approach_a,
                0,
                speed=4.0,
                velocity_y=4.0,
                history_speeds=(4.0,) * 3,
                history_anchors=vertical_history,
            ),
            _state(
                171,
                approach_b,
                0,
                speed=4.0,
                velocity_x=-4.0,
                history_speeds=(4.0,) * 3,
                history_anchors=horizontal_history,
            ),
        ],
        0,
    ) == []

    for frame_index in range(1, 4):
        assert detector.detect_collisions([], frame_index) == []

    events = detector.detect_collisions(
        [
            _state(
                170,
                contact_a,
                4,
                speed=0.0,
                stopped_frames=3,
                history_speeds=(4.0, 4.0, 0.0),
            ),
            _state(
                171,
                contact_b,
                4,
                speed=0.0,
                stopped_frames=3,
                history_speeds=(4.0, 4.0, 0.0),
            ),
        ],
        4,
    )

    assert len(events) == 1
    assert "dual_stop" in events[0].reason
    assert "occlusion_bridge" in events[0].reason
    assert detector.last_diagnostics[0].observation_gap_frames == 3
    assert detector.last_diagnostics[0].bridged_dual_stop is True


def test_crossing_dual_stop_without_tracking_gap_is_ignored() -> None:
    detector = _detector(
        collision_confirmation_frames=1,
        crossing_history_frames=6,
        crossing_dual_stop_bridge_frames=12,
        crossing_dual_stop_min_gap_frames=2,
    )
    approach_a = _make_mask(1, 3, 7, 8)
    approach_b = _make_mask(8, 10, 14, 16)
    contact_a = _make_mask(4, 4, 12, 11)
    contact_b = _make_mask(7, 8, 15, 15)
    vertical_history = ((5.0, 3.0), (5.0, 5.0), (5.0, 7.0))
    horizontal_history = ((16.0, 12.0), (14.0, 12.0), (12.0, 12.0))

    detector.detect_collisions(
        [
            _state(
                180,
                approach_a,
                0,
                speed=4.0,
                velocity_y=4.0,
                history_speeds=(4.0,) * 3,
                history_anchors=vertical_history,
            ),
            _state(
                181,
                approach_b,
                0,
                speed=4.0,
                velocity_x=-4.0,
                history_speeds=(4.0,) * 3,
                history_anchors=horizontal_history,
            ),
        ],
        0,
    )
    events = detector.detect_collisions(
        [
            _state(
                180,
                contact_a,
                1,
                speed=0.0,
                stopped_frames=3,
                history_speeds=(4.0, 4.0, 0.0),
            ),
            _state(
                181,
                contact_b,
                1,
                speed=0.0,
                stopped_frames=3,
                history_speeds=(4.0, 4.0, 0.0),
            ),
        ],
        1,
    )

    assert events == []
    assert detector.last_diagnostics[0].observation_gap_frames == 0
    assert detector.last_diagnostics[0].bridged_dual_stop is False


def test_strong_impact_after_tracking_gap_is_confirmed_immediately() -> None:
    detector = _detector(
        collision_confirmation_frames=2,
        crossing_dual_stop_bridge_frames=12,
        crossing_dual_stop_min_gap_frames=2,
    )
    approach_a = _make_mask(2, 2, 8, 7)
    approach_b = _make_mask(2, 12, 8, 17)
    contact_a = _make_mask(2, 4, 10, 12)
    contact_b = _make_mask(4, 7, 12, 15)

    assert detector.detect_collisions(
        [
            _state(
                190,
                approach_a,
                0,
                speed=6.0,
                velocity_x=6.0,
                history_speeds=(6.0, 6.0),
            ),
            _state(
                191,
                approach_b,
                0,
                speed=6.0,
                velocity_x=-6.0,
                history_speeds=(6.0, 6.0),
            ),
        ],
        0,
    ) == []
    assert detector.detect_collisions([], 1) == []
    assert detector.detect_collisions([], 2) == []

    events = detector.detect_collisions(
        [
            _state(
                190,
                contact_a,
                3,
                speed=1.0,
                acceleration=-6.0,
                history_speeds=(6.0, 6.0, 1.0),
            ),
            _state(
                191,
                contact_b,
                3,
                speed=3.0,
                history_speeds=(6.0, 6.0, 3.0),
            ),
        ],
        3,
    )

    assert len(events) == 1
    assert "occlusion_bridge" in events[0].reason
    assert detector.last_diagnostics[0].observation_gap_frames == 2
    assert detector.last_diagnostics[0].bridged_strong_impact is True


def test_dilated_only_contact_does_not_reuse_stale_dynamic_evidence() -> None:
    detector = _detector(collision_confirmation_frames=2)
    mask_a = _make_mask(2, 2, 8, 5)
    touching_b = _make_mask(2, 5, 8, 8)

    first = detector.detect_collisions(
        [
            _state(
                160,
                mask_a,
                0,
                speed=0.0,
                acceleration=-6.0,
                velocity_x=5.0,
                history_speeds=(5.0, 0.0),
            ),
            _state(161, touching_b, 0, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        0,
    )
    second = detector.detect_collisions(
        [
            _state(160, mask_a, 1, speed=3.0, history_speeds=(5.0, 3.0)),
            _state(161, touching_b, 1, speed=0.0, history_speeds=(0.0, 0.0)),
        ],
        1,
    )

    assert first == []
    assert second == []
