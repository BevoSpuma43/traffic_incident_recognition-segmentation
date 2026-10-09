from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Project(Settings):
    root_dir: Path = Path("..")
    seed: int = 42
    output_dir: Path = Path("outputs")


class Video(Settings):
    source: str = "data/samples/demo.mp4"
    clip_id: str | None = None
    max_width: int | None = Field(None, ge=320, le=3840)
    target_fps: float = Field(8, gt=0, le=120)
    queue_size: int = Field(2, ge=1, le=4)
    reconnect_rtsp: bool = True
    max_reconnects: int = Field(3, ge=0)
    timeout_s: float = Field(10, gt=0)
    max_frames: int | None = Field(None, gt=0)


class Perception(Settings):
    model: Path = Path("models/yolo26n-seg.pt")
    backend: Literal["pytorch", "onnx", "openvino", "tensorrt", "synthetic"] = "pytorch"
    device: str = "cpu"
    image_size: int = Field(640, ge=128, le=2048)
    confidence: float = Field(0.1, gt=0, lt=1)
    classes: list[str] = ["car", "truck", "bus", "motorcycle"]
    max_detections: int = Field(100, ge=1, le=1000)
    half: bool = False


class Tracking(Settings):
    motion_matching: bool = True
    track_high_thresh: float = Field(0.3, gt=0, lt=1)
    track_low_thresh: float = Field(0.1, gt=0, lt=1)
    new_track_thresh: float = Field(0.35, gt=0, lt=1)
    match_thresh: float = Field(0.8, gt=0, lt=1)
    fuse_score: bool = True
    lost_seconds: float = Field(1, gt=0)
    history_seconds: float = Field(5, gt=0)
    max_samples: int = Field(300, ge=3)
    max_tracks: int = Field(500, ge=1)

    @model_validator(mode="after")
    def thresholds(self):
        if not self.track_low_thresh <= self.track_high_thresh <= self.new_track_thresh:
            raise ValueError("Expected low <= high <= new track thresholds")
        return self


class CalibrationSettings(Settings):
    camera_id: str = Field("camera_01", pattern=r"^[a-zA-Z0-9_-]+$")
    file: Path = Path("data/calibration/camera_01.yaml")
    expected_record_id: str | None = Field(None, pattern=r"^[0-9a-f]{32}$")
    expected_revision: int | None = Field(None, ge=1)
    min_confidence: float = Field(0.55, ge=0, le=1)
    detect_camera_motion: bool = True
    motion_threshold_px: float = Field(8, gt=0)
    motion_check_s: float = Field(2, gt=0)


class ImageFeatures(Settings):
    pair_radius_diagonals: float = Field(4, gt=0)
    collision_distance_diagonals: float = Field(0.8, gt=0)
    max_speed_diagonals_s: float = Field(12, gt=0)
    stop_speed_diagonals_s: float = Field(0.12, gt=0)


class Features(Settings):
    image: ImageFeatures = Field(default_factory=ImageFeatures)
    ground_point_method: Literal["mask", "box"] = "mask"
    ema_alpha: float = Field(0.35, gt=0, le=1)
    smoothing_reference_s: float = Field(0.125, gt=0)
    max_gap_s: float = Field(0.6, gt=0)
    max_speed_mps: float = Field(70, gt=0)
    stop_speed_mps: float = Field(0.7, ge=0)
    pair_radius_m: float = Field(15, gt=0)
    collision_distance_m: float = Field(2.5, gt=0)
    window_seconds: float = Field(3, gt=0)


class SideswipeEvents(Settings):
    enabled: bool = True
    min_contact_coverage: float = Field(0.03, gt=0, le=1)
    min_shape_change: float = Field(1.6, gt=1)
    min_heading_change_rad: float = Field(0.3, gt=0)
    min_velocity_change_diagonals_s: float = Field(0.5, gt=0)
    min_direction_cosine: float = Field(0.8, ge=-1, le=1)
    confirmation_window_s: float = Field(1.5, gt=0)
    velocity_window_s: float = Field(0.35, gt=0)


class ImageEvents(Settings):
    sideswipe: SideswipeEvents = Field(default_factory=SideswipeEvents)
    min_bbox_iou: float = Field(0.05, ge=0, le=1)
    min_contact_coverage: float = Field(0.1, gt=0, le=1)
    min_approach_speed_diagonals_s: float = Field(0.25, gt=0)
    occlusion_confirmation_s: float = Field(1.0, gt=0)
    post_impact_speed_ratio: float = Field(0.5, gt=0, lt=1)
    min_prior_speed_diagonals_s: float = Field(0.5, gt=0)
    deceleration_diagonals_s2: float = Field(0.8, gt=0)
    min_quality: float = Field(0.25, ge=0, le=1)
    heading_change_rad: float = Field(0.6, gt=0)


class Events(Settings):
    coordinate_mode: Literal["metric", "image"] = "metric"
    image: ImageEvents = Field(default_factory=ImageEvents)
    candidate_threshold: float = Field(0.55, ge=0, le=1)
    confirm_threshold: float = Field(0.8, ge=0, le=1)
    candidate_duration_s: float = Field(0.2, gt=0)
    confirm_duration_s: float = Field(0.4, gt=0)
    candidate_timeout_s: float = Field(3, gt=0)
    cooldown_s: float = Field(8, gt=0)
    evidence_gap_s: float = Field(0.65, gt=0)
    min_track_age_s: float = Field(0.5, ge=0)
    min_quality: float = Field(0.55, ge=0, le=1)
    deceleration_mps2: float = Field(4, gt=0)
    min_prior_speed_mps: float = Field(2, gt=0)
    low_ttc_s: float = Field(1.5, gt=0)
    pre_event_s: float = Field(5, ge=0)
    post_event_s: float = Field(10, ge=0)
    classifier: Path | None = None

    @model_validator(mode="after")
    def thresholds(self):
        if self.coordinate_mode == "image" and self.classifier is not None:
            raise ValueError("The metric classifier cannot be applied to image-coordinate features")
        if self.confirm_threshold < self.candidate_threshold:
            raise ValueError("confirm_threshold must be >= candidate_threshold")
        return self


class Storage(Settings):
    retention_gb: float = Field(10, gt=0)
    buffer_mb: int = Field(256, ge=1)
    max_pending_clips: int = Field(8, ge=1, le=32)
    jpeg_quality: int = Field(80, ge=30, le=100)


class UI(Settings):
    render_masks: bool = True
    render_bird_eye: bool = True


class AppConfig(Settings):
    project: Project = Field(default_factory=Project)
    video: Video = Field(default_factory=Video)
    perception: Perception = Field(default_factory=Perception)
    tracking: Tracking = Field(default_factory=Tracking)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    features: Features = Field(default_factory=Features)
    events: Events = Field(default_factory=Events)
    storage: Storage = Field(default_factory=Storage)
    ui: UI = Field(default_factory=UI)

    @model_validator(mode="after")
    def validate_low_confidence(self):
        if self.perception.confidence > self.tracking.track_low_thresh:
            raise ValueError(
                "perception.confidence must retain ByteTrack low-confidence detections"
            )
        return self


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    path = Path(path).resolve()
    cfg = AppConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    root = (path.parent / cfg.project.root_dir).resolve()
    cfg.project.root_dir = root
    cfg.project.output_dir = (root / cfg.project.output_dir).resolve()
    cfg.perception.model = (root / cfg.perception.model).resolve()
    cfg.calibration.file = (root / cfg.calibration.file).resolve()
    if cfg.events.classifier:
        cfg.events.classifier = (root / cfg.events.classifier).resolve()
    if not cfg.video.source.lower().startswith(("rtsp://", "rtsps://")):
        cfg.video.source = str((root / cfg.video.source).resolve())
    return cfg
