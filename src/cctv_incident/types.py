from dataclasses import dataclass, field

import numpy as np


@dataclass
class Instance:
    bbox: np.ndarray
    mask: np.ndarray
    class_id: int
    confidence: float


@dataclass
class Track:
    track_id: int
    instance: Instance


@dataclass
class Observation:
    track_id: int
    timestamp_s: float
    point: np.ndarray
    point_px: tuple[float, float]
    bbox: np.ndarray
    class_id: int = 2
    confidence: float = 1.0
    quality: float = 1.0
    predicted: bool = False


@dataclass
class Motion:
    observation: Observation
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: float
    speed: float
    deceleration: float
    heading_change: float
    age_s: float
    stop_duration_s: float
    prior_speed: float
    quality: float


@dataclass
class PairFeatures:
    track_ids: tuple[int, int]
    distance: float
    relative_speed: float
    closest_time: float
    closest_distance: float
    ttc: float | None
    converging: bool
    bbox_iou: float
    quality: float


@dataclass
class Event:
    event_id: str
    camera_id: str
    start_time_s: float
    impact_time_s: float
    confirm_time_s: float
    track_ids: list[int]
    score: float
    calibration_confidence: float
    reasons: list[str]
    coordinate_mode: str = "metric"
    run_id: str = ""
    clip_id: str = ""
    clip_path: str | None = None
    clip_status: str = "pending"
    clip_start_s: float | None = None
    clip_end_s: float | None = None
    clip_truncated: bool = False


@dataclass
class Decision:
    state: str = "NORMAL"
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    candidate_ids: list[int] = field(default_factory=list)
