import inspect

import numpy as np
import pytest

from src.collision_logic import CollisionDetector
from src.models import Point2D, VehicleState


def _make_mask(
    top: int,
    left: int,
    bottom: int,
    right: int,
    shape: tuple[int, int] = (10, 10),
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    return mask


def _make_point(x: float, y: float) -> Point2D:
    try:
        return Point2D(x=x, y=y)
    except TypeError:
        return Point2D(x, y)


def _make_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _build_vehicle_state(
    track_id: int,
    mask: np.ndarray,
    *,
    speed_px: float = 0.0,
    acceleration: float = 0.0,
    stopped_frames: int = 0,
    centroid: tuple[float, float] = (0.0, 0.0),
) -> VehicleState:
    sig = inspect.signature(VehicleState)
    bbox = _make_bbox_from_mask(mask)
    point = _make_point(*centroid)

    values: dict[str, object] = {
        "track_id": track_id,
        "id": track_id,
        "centroid": point,
        "current_centroid": point,
        "last_centroid": point,
        "mask": mask,
        "binary_mask": mask,
        "current_mask": mask,
        "speed_px": speed_px,
        "speed": speed_px,
        "acceleration": acceleration,
        "acceleration_px": acceleration,
        "stopped_frames": stopped_frames,
        "stopped_counter": stopped_frames,
        "stop_counter": stopped_frames,
        "bbox": bbox,
        "bounding_box": bbox,
        "missing_frames": 0,
        "frames_missing": 0,
        "last_seen_frame": 0,
        "frame_index": 0,
        "confidence": 1.0,
        "class_id": 2,
        "cls_id": 2,
        "label": "car",
        "history": [],
        "collided": False,
    }

    kwargs: dict[str, object] = {}
    for name, parameter in sig.parameters.items():
        if name == "self":
            continue
        if name in values:
            kwargs[name] = values[name]
            continue
        if parameter.default is not inspect._empty:
            continue
        if name.endswith("mask"):
            kwargs[name] = mask
        elif "centroid" in name:
            kwargs[name] = point
        elif "bbox" in name or "box" in name:
            kwargs[name] = bbox
        elif "track" in name or name == "id":
            kwargs[name] = track_id
        elif "speed" in name:
            kwargs[name] = speed_px
        elif "acceleration" in name:
            kwargs[name] = acceleration
        elif "stopped" in name:
            kwargs[name] = stopped_frames
        elif "history" in name:
            kwargs[name] = []
        elif "missing" in name or "frame" in name:
            kwargs[name] = 0
        elif "confidence" in name:
            kwargs[name] = 1.0
        elif "class" in name or "cls" in name:
            kwargs[name] = 2
        elif "label" in name:
            kwargs[name] = "car"
        else:
            kwargs[name] = None

    return VehicleState(**kwargs)


def _build_detector() -> CollisionDetector:
    sig = inspect.signature(CollisionDetector)
    values: dict[str, object] = {
        "mask_overlap_threshold": 4,
        "stopped_frames_threshold": 3,
        "strong_deceleration_threshold": -4.0,
        "stopped_speed_threshold": 2.5,
    }

    kwargs: dict[str, object] = {}
    for name, parameter in sig.parameters.items():
        if name == "self":
            continue
        if name in values:
            kwargs[name] = values[name]
        elif parameter.default is inspect._empty:
            if "threshold" in name:
                kwargs[name] = 0
            else:
                kwargs[name] = None

    return CollisionDetector(**kwargs)


def _extract_events(result: object) -> list[object]:
    if result is None:
        return []
    if isinstance(result, list):
        return result
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, list):
                return item
        return list(result)
    if isinstance(result, dict):
        return list(result.values())
    return [result]


def _run_detection(detector: CollisionDetector, states: list[VehicleState]) -> list[object]:
    states_by_id = {
        int(getattr(state, "track_id", getattr(state, "id"))): state for state in states
    }

    for method_name in ("detect", "detect_collisions", "find_collisions", "__call__"):
        if not hasattr(detector, method_name):
            continue

        method = getattr(detector, method_name)
        sig = inspect.signature(method)

        kwargs: dict[str, object] = {}
        args: list[object] = []

        for name, parameter in sig.parameters.items():
            if name == "self":
                continue
            if name in {"states", "vehicle_states", "tracks", "track_states"}:
                kwargs[name] = states_by_id
            elif name in {"vehicles", "detections"}:
                kwargs[name] = states
            elif name in {"frame_index", "frame_idx"}:
                kwargs[name] = 0
            elif parameter.default is inspect._empty:
                args.append(states_by_id)

        result = method(*args, **kwargs)
        return _extract_events(result)

    raise AssertionError("CollisionDetector does not expose a supported detection method")


def _event_pair(event: object) -> tuple[int, int]:
    candidate_pairs = [
        ("track_id_a", "track_id_b"),
        ("vehicle_id_a", "vehicle_id_b"),
        ("a_track_id", "b_track_id"),
        ("track_a_id", "track_b_id"),
        ("id_a", "id_b"),
    ]

    for left_name, right_name in candidate_pairs:
        if hasattr(event, left_name) and hasattr(event, right_name):
            left = int(getattr(event, left_name))
            right = int(getattr(event, right_name))
            return tuple(sorted((left, right)))

    for pair_name in ("track_ids", "pair", "vehicle_ids", "ids"):
        if hasattr(event, pair_name):
            pair_value = getattr(event, pair_name)
            if isinstance(pair_value, (tuple, list, set)) and len(pair_value) == 2:
                left, right = [int(v) for v in pair_value]
                return tuple(sorted((left, right)))

    raise AssertionError("Unsupported CollisionEvent format")


def test_collision_detected_with_sufficient_overlap_and_both_stopped() -> None:
    detector = _build_detector()
    mask_a = _make_mask(1, 1, 5, 5)
    mask_b = _make_mask(3, 3, 7, 7)

    state_a = _build_vehicle_state(
        1,
        mask_a,
        speed_px=0.0,
        acceleration=0.0,
        stopped_frames=5,
        centroid=(2.0, 2.0),
    )
    state_b = _build_vehicle_state(
        2,
        mask_b,
        speed_px=0.0,
        acceleration=0.0,
        stopped_frames=5,
        centroid=(4.0, 4.0),
    )

    events = _run_detection(detector, [state_a, state_b])

    assert len(events) == 1
    assert _event_pair(events[0]) == (1, 2)


def test_collision_detected_with_sufficient_overlap_and_strong_deceleration() -> None:
    detector = _build_detector()
    mask_a = _make_mask(1, 1, 5, 5)
    mask_b = _make_mask(3, 3, 7, 7)

    state_a = _build_vehicle_state(
        10,
        mask_a,
        speed_px=6.0,
        acceleration=-5.5,
        stopped_frames=0,
        centroid=(2.0, 2.0),
    )
    state_b = _build_vehicle_state(
        11,
        mask_b,
        speed_px=5.5,
        acceleration=-4.5,
        stopped_frames=0,
        centroid=(4.0, 4.0),
    )

    events = _run_detection(detector, [state_a, state_b])

    assert len(events) == 1
    assert _event_pair(events[0]) == (10, 11)


def test_no_collision_detected_without_overlap() -> None:
    detector = _build_detector()
    mask_a = _make_mask(1, 1, 3, 3)
    mask_b = _make_mask(6, 6, 8, 8)

    state_a = _build_vehicle_state(
        20,
        mask_a,
        speed_px=0.0,
        acceleration=0.0,
        stopped_frames=5,
        centroid=(1.5, 1.5),
    )
    state_b = _build_vehicle_state(
        21,
        mask_b,
        speed_px=0.0,
        acceleration=0.0,
        stopped_frames=5,
        centroid=(6.5, 6.5),
    )

    events = _run_detection(detector, [state_a, state_b])

    assert events == []


def test_no_duplicate_collision_events_for_same_pair() -> None:
    detector = _build_detector()
    mask_a = _make_mask(1, 1, 5, 5)
    mask_b = _make_mask(3, 3, 7, 7)

    state_a = _build_vehicle_state(
        30,
        mask_a,
        speed_px=0.0,
        acceleration=0.0,
        stopped_frames=5,
        centroid=(2.0, 2.0),
    )
    state_b = _build_vehicle_state(
        31,
        mask_b,
        speed_px=0.0,
        acceleration=0.0,
        stopped_frames=5,
        centroid=(4.0, 4.0),
    )

    events = _run_detection(detector, [state_a, state_b])

    pairs = [_event_pair(event) for event in events]

    assert len(events) == 1
    assert pairs == [(30, 31)]