from __future__ import annotations

import math

import numpy as np
import pytest

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


def _reid_config(**overrides: object) -> AppConfig:
    values: dict[str, object] = {
        "track_reid_enabled": True,
        "velocity_ema_alpha": 1.0,
    }
    values.update(overrides)
    return AppConfig(**values)


def _feed_moving_track(store: VehicleStateStore) -> None:
    """Tre frame di un veicolo che avanza di 4 px/frame verso destra."""
    store.update([_detection(1, 2, 2, 8, 8)], frame_index=0)
    store.update([_detection(1, 6, 2, 12, 8)], frame_index=1)
    store.update([_detection(1, 10, 2, 16, 8)], frame_index=2)


def test_reassigned_id_inherits_the_kinematic_history() -> None:
    store = VehicleStateStore(_reid_config())
    _feed_moving_track(store)

    # L'ID 1 sparisce e ne compare uno nuovo dove il veicolo era atteso:
    # e la firma di un cambio di ID durante un'occlusione, non di un veicolo
    # appena entrato in scena.
    state = store.update([_detection(9, 14, 2, 20, 8)], frame_index=3)[0]

    assert state.track_id == 9
    assert state.observed_frames == 4
    assert len(state.history) == 4
    assert state.kinematics_valid is True
    assert state.speed_px == 4.0
    assert [item.track_id for item in store.get_active_states()] == [9]


def test_reid_is_disabled_unless_requested() -> None:
    store = VehicleStateStore(AppConfig(velocity_ema_alpha=1.0))
    _feed_moving_track(store)

    state = store.update([_detection(9, 14, 2, 20, 8)], frame_index=3)[0]

    assert state.observed_frames == 1
    assert state.kinematics_valid is False


def test_reid_ignores_a_track_that_is_still_visible() -> None:
    store = VehicleStateStore(_reid_config())
    _feed_moving_track(store)

    states = store.update(
        [_detection(1, 14, 2, 20, 8), _detection(9, 15, 2, 21, 8)],
        frame_index=3,
    )

    inherited = next(item for item in states if item.track_id == 1)
    newborn = next(item for item in states if item.track_id == 9)
    assert inherited.observed_frames == 4
    assert newborn.observed_frames == 1


def test_reid_does_not_adopt_a_distant_detection() -> None:
    store = VehicleStateStore(_reid_config())
    _feed_moving_track(store)

    state = store.update([_detection(9, 30, 30, 36, 36)], frame_index=3)[0]

    assert state.observed_frames == 1
    assert state.kinematics_valid is False


def test_reid_does_not_adopt_a_much_larger_silhouette() -> None:
    store = VehicleStateStore(_reid_config(track_reid_max_scale_ratio=1.5))
    _feed_moving_track(store)

    state = store.update([_detection(9, 5, 2, 29, 26)], frame_index=3)[0]

    assert state.observed_frames == 1


def test_duplicate_track_id_keeps_highest_confidence_detection() -> None:
    store = VehicleStateStore(AppConfig())
    low = _detection(7, 2, 2, 8, 8, confidence=0.4)
    high = _detection(7, 12, 2, 18, 8, confidence=0.9)

    states = store.update([low, high], frame_index=0)

    assert len(states) == 1
    assert states[0].bbox == high.bbox
    assert states[0].confidence == 0.9


def test_delta_v_needs_two_measured_frames_before_it_means_anything() -> None:
    store = VehicleStateStore(AppConfig())

    first = store.update([_detection(1, 2, 2, 8, 8)], frame_index=0)[0]
    second = store.update([_detection(1, 6, 2, 12, 8)], frame_index=1)[0]

    # Al primo frame non c'e cinematica, al secondo non c'e ancora una velocita
    # precedente: senza questa guardia ogni track nuovo sembrerebbe urtato.
    assert first.delta_v_px == 0.0
    assert second.delta_v_px == 0.0


def test_delta_v_uses_the_raw_velocity_not_the_filtered_one() -> None:
    # Un EMA molto aggressivo: se il delta-V leggesse la velocita filtrata,
    # il valore risulterebbe fortemente smorzato.
    store = VehicleStateStore(AppConfig(velocity_ema_alpha=0.1))
    store.update([_detection(1, 0, 2, 6, 8)], frame_index=0)
    store.update([_detection(1, 10, 2, 16, 8)], frame_index=1)

    # Il veicolo devia di 90 gradi mantenendo lo stesso modulo di velocita.
    state = store.update([_detection(1, 10, 12, 16, 18)], frame_index=2)[0]

    assert state.raw_velocity_x_px == 0.0
    assert state.raw_velocity_y_px == 10.0
    assert state.delta_v_px == pytest.approx(10.0 * math.sqrt(2))


def test_deflection_at_constant_speed_is_invisible_to_scalar_acceleration() -> None:
    store = VehicleStateStore(AppConfig(velocity_ema_alpha=1.0))
    store.update([_detection(1, 0, 2, 6, 8)], frame_index=0)
    store.update([_detection(1, 10, 2, 16, 8)], frame_index=1)

    state = store.update([_detection(1, 10, 12, 16, 18)], frame_index=2)[0]

    # Stesso modulo prima e dopo la deviazione: l'accelerazione scalare non
    # vede nulla, il delta-V vede l'urto. E la ragione d'essere del punto 3.
    assert state.speed_px == pytest.approx(10.0)
    assert state.acceleration_px == pytest.approx(0.0)
    assert state.delta_v_px > 14.0
