import math

import pytest

from src.kinematics import (
    compute_acceleration,
    compute_speed_px,
    update_stopped_counter,
)


def test_compute_speed_px_returns_zero_when_centroid_is_none() -> None:
    assert compute_speed_px(None, (3.0, 4.0)) == pytest.approx(0.0)
    assert compute_speed_px((3.0, 4.0), None) == pytest.approx(0.0)
    assert compute_speed_px(None, None) == pytest.approx(0.0)


def test_compute_speed_px_returns_known_distance_for_3_4_5_triangle() -> None:
    speed = compute_speed_px((0.0, 0.0), (3.0, 4.0))

    assert speed == pytest.approx(5.0)


def test_compute_speed_px_returns_zero_for_identical_centroids() -> None:
    speed = compute_speed_px((2.0, 2.0), (2.0, 2.0))

    assert speed == pytest.approx(0.0)


def test_compute_acceleration_returns_difference_between_speeds() -> None:
    acceleration = compute_acceleration(7.5, 5.0)

    assert acceleration == pytest.approx(2.5)


def test_compute_acceleration_returns_negative_value_for_deceleration() -> None:
    acceleration = compute_acceleration(2.0, 5.0)

    assert acceleration == pytest.approx(-3.0)


def test_update_stopped_counter_increments_when_speed_is_below_threshold() -> None:
    counter = update_stopped_counter(
        stopped_counter=2,
        speed_px=1.5,
        stopped_speed_threshold=2.5,
    )

    assert counter == 3


def test_update_stopped_counter_resets_when_speed_is_above_threshold() -> None:
    counter = update_stopped_counter(
        stopped_counter=4,
        speed_px=3.0,
        stopped_speed_threshold=2.5,
    )

    assert counter == 0


def test_update_stopped_counter_keeps_incrementing_at_threshold_boundary() -> None:
    counter = update_stopped_counter(
        stopped_counter=1,
        speed_px=2.5,
        stopped_speed_threshold=2.5,
    )

    assert counter == 2