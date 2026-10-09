"""Versioned, immutable editor records; only confirmed metric records become runtime data."""

from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .coordinates import full_image_roi, rectangle_destination, validate_quad, validate_roi
from .homography import Calibration, estimate_calibration, transform_points

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Size = tuple[Annotated[int, Field(ge=2)], Annotated[int, Field(ge=2)]]


def utc_now():
    return datetime.now(timezone.utc)


class RecordModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class VideoIdentity(RecordModel):
    source_path: str
    relative_path: str | None = None
    original_name: str
    sha256: Digest
    size_bytes: int = Field(gt=0)
    image_size: Size
    rotation_degrees: Finite = 0
    first_frame_sha256: Digest


class ReferenceFrame(RecordModel):
    frame_index: int = Field(ge=0)
    timestamp_s: Finite | None = None
    pts: int | None = None
    time_base: str | None = None
    pixel_sha256: Digest
    image_path: str | None = None
    image_sha256: Digest | None = None

    @model_validator(mode="after")
    def file_pair(self):
        if (self.image_path is None) != (self.image_sha256 is None):
            raise ValueError("Reference image path and hash must be provided together")
        return self


class Vertex(RecordModel):
    id: str = Field(pattern=r"^P[1-9][0-9]*$")
    x: Finite
    y: Finite


class Distance(RecordModel):
    value: Positive | None = None
    unit: Literal["m"] = "m"
    origin: Literal["unknown", "measured", "standard", "experimental"] = "unknown"
    source: str | None = None
    preset: str | None = None
    user_confirmed: bool = False

    @model_validator(mode="after")
    def provenance(self):
        if self.user_confirmed and (self.value is None or self.origin == "unknown"):
            raise ValueError("Confirmed distance requires a value and explicit provenance")
        if self.origin in {"standard", "experimental"} and not self.source:
            raise ValueError("Assumed distances require a source or explanation")
        return self


class Automation(RecordModel):
    method: Literal["manual", "auto_assisted"] = "manual"
    algorithm_version: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None
    initial_points_px: tuple[tuple[Finite, Finite], ...] = ()
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ExplicitScale(RecordModel):
    """Provenance of an imported legacy metric coordinate system."""

    origin: Literal["measured", "standard", "experimental"]
    source: str = Field(min_length=1)
    user_confirmed: bool = False


class CalibrationRecord(RecordModel):
    schema_version: Literal[1] = 1
    record_id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[0-9a-f]{32}$")
    revision: int = Field(0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    modified_at: datetime = Field(default_factory=utc_now)
    video: VideoIdentity
    reference: ReferenceFrame
    camera_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    status: Literal["draft", "confirmed", "invalid"] = "draft"
    invalid_reasons: tuple[str, ...] = ()
    geometry_mode: Literal["rectangle", "explicit"] = "rectangle"
    vertices: tuple[Vertex, ...] = Field(default=(), max_length=64)
    roi_px: tuple[tuple[Finite, Finite], ...] | None = None
    width: Distance = Field(default_factory=Distance)
    length: Distance = Field(default_factory=Distance)
    destination_points: tuple[tuple[Finite, Finite], ...] = ()
    explicit_scale: ExplicitScale | None = None
    automation: Automation = Field(default_factory=Automation)
    automatic_acceptance: Literal["experimental_usa_v1"] | None = None
    geometric_quality: Finite = Field(0, ge=0, le=1)
    runtime: Calibration | None = None

    @model_validator(mode="after")
    def consistency(self):
        if self.created_at.tzinfo is None or self.modified_at.tzinfo is None:
            raise ValueError("Record dates must include a timezone")
        if self.modified_at < self.created_at:
            raise ValueError("Modification date precedes creation")
        ids = [v.id for v in self.vertices]
        if len(set(ids)) != len(ids):
            raise ValueError("Vertex IDs must be unique")
        if self.geometry_mode == "rectangle" and not set(ids).issubset({"P1", "P2", "P3", "P4"}):
            raise ValueError("Rectangle vertices must use stable IDs P1–P4")
        if self.geometry_mode == "explicit" and ids != [f"P{i + 1}" for i in range(len(ids))]:
            raise ValueError("Explicit correspondences must retain their P1…Pn order")
        if self.reference.pixel_sha256 != self.video.first_frame_sha256:
            raise ValueError("Reference frame does not belong to the inspected video")
        if self.status != "confirmed" and self.runtime is not None:
            raise ValueError("Draft or invalid records cannot contain a runtime calibration")
        if self.status == "invalid" and not self.invalid_reasons:
            raise ValueError("Invalid records require a reason")
        if self.automatic_acceptance:
            selected = self.automation.diagnostics.get("selected_candidate", {})
            if (
                self.geometry_mode != "rectangle"
                or self.automation.method != "auto_assisted"
                or not self.automation.algorithm_version
                or selected.get("points_px") != self.points_px
                or any(
                    selected.get(name) != getattr(self, name).model_dump(mode="json")
                    or getattr(self, name).origin != "experimental"
                    for name in ("width", "length")
                )
            ):
                raise ValueError(
                    "Automatic acceptance requires the unchanged experimental proposal"
                )
        if self.status == "confirmed":
            if self.runtime is None or self.invalid_reasons:
                raise ValueError(
                    "Confirmed record requires runtime geometry and no invalid reasons"
                )
            expected = _build_runtime(self)
            actual = self.runtime
            if (
                actual.camera_id != expected.camera_id
                or actual.image_size != expected.image_size
                or actual.units != "m"
                or not actual.accepted
                or not actual.valid
                or actual.method != expected.method
                or actual.reference_image != self.reference.image_path
                or actual.confidence != self.geometric_quality
            ):
                raise ValueError("Runtime calibration disagrees with the confirmed record")
            for field in ("source_points_px", "destination_points", "roi_px"):
                a, b = np.asarray(getattr(actual, field)), np.asarray(getattr(expected, field))
                if a.shape != b.shape or not np.allclose(a, b, atol=1e-9, rtol=1e-9):
                    raise ValueError(f"Runtime {field} disagrees with the editor")
            # This checks serialization consistency, not physical metric accuracy.
            src = np.asarray(expected.source_points_px)
            probes = np.vstack([src, (src + np.roll(src, -1, axis=0)) / 2, src.mean(axis=0)])
            if len(src) == 4:
                if not np.allclose(
                    transform_points(actual.homography, probes),
                    transform_points(expected.homography, probes),
                    atol=1e-5,
                    rtol=1e-6,
                ):
                    raise ValueError("Runtime homography disagrees with the editor")
            else:
                # More than four explicit correspondences may contain outliers.
                # RANSAC may select different valid subsets on different calls.
                errors = np.linalg.norm(
                    transform_points(actual.homography, src)
                    - np.asarray(expected.destination_points),
                    axis=1,
                )
                inliers = src[errors <= 0.25]
                if len(inliers) < 4 or np.linalg.matrix_rank(inliers - inliers.mean(axis=0)) < 2:
                    raise ValueError("Runtime homography lacks four consistent correspondences")
        return self

    @property
    def points_px(self):
        vertices = sorted(self.vertices, key=lambda v: int(v.id[1:]))
        return [[v.x, v.y] for v in vertices]


def _build_runtime(record):
    points = np.asarray(record.points_px)
    if points.size and (
        (points < 0).any() or (points > np.asarray(record.video.image_size) - 1).any()
    ):
        raise ValueError("Calibration vertices fall outside the decoded image")
    if record.geometry_mode == "rectangle":
        validate_quad(record.points_px, record.video.image_size)
        for distance in (record.width, record.length):
            if (
                distance.value is None
                or not (distance.user_confirmed or record.automatic_acceptance)
                or distance.origin == "unknown"
            ):
                raise ValueError("Both metric distances require values and confirmed provenance")
        destination = rectangle_destination(record.width.value, record.length.value)
    else:
        if record.explicit_scale is None or not record.explicit_scale.user_confirmed:
            raise ValueError("Explicit metric coordinates require confirmed scale provenance")
        destination = record.destination_points
        if len(record.vertices) == 4:
            validate_quad(record.points_px, record.video.image_size)
    roi = full_image_roi(record.video.image_size) if record.roi_px is None else record.roi_px
    if record.roi_px is not None:
        validate_roi(roi, record.video.image_size)
    calibration = estimate_calibration(
        record.camera_id,
        record.video.image_size,
        record.points_px,
        destination,
        units="m",
        roi=roi,
        method=record.automation.method,
    )
    data = calibration.model_dump()
    data.update(confidence=record.geometric_quality, reference_image=record.reference.image_path)
    return Calibration.model_validate(data)


def confirm_record(record):
    record = CalibrationRecord.model_validate(record.model_dump())
    if record.status == "invalid":
        raise ValueError("An invalid calibration must be corrected before confirmation")
    # A manual confirmation must never inherit acceptance from the automatic policy.
    if record.automatic_acceptance:
        record = edit_record(record)
    data = record.model_dump()
    data.update(
        status="confirmed",
        invalid_reasons=(),
        runtime=_build_runtime(record).model_dump(),
        modified_at=utc_now(),
    )
    return CalibrationRecord.model_validate(data)


def accept_automatic_record(record):
    """Accept an approximate US hypothesis without claiming user confirmation."""
    if record.status != "draft":
        raise ValueError("Automatic acceptance requires a draft")
    data = record.model_dump()
    data["automatic_acceptance"] = "experimental_usa_v1"
    candidate = CalibrationRecord.model_validate(data)
    data.update(
        status="confirmed", runtime=_build_runtime(candidate).model_dump(), modified_at=utc_now()
    )
    return CalibrationRecord.model_validate(data)


def edit_record(record, **changes):
    """Every editor change drops acceptance; quality is never promoted by confirmation."""
    allowed = {
        "vertices",
        "roi_px",
        "width",
        "length",
        "destination_points",
        "explicit_scale",
        "automation",
        "geometric_quality",
        "geometry_mode",
    }
    if set(changes) - allowed:
        raise ValueError("Only editor fields may be changed")
    if (
        set(changes) & {"vertices", "roi_px", "destination_points", "geometry_mode"}
        and "geometric_quality" not in changes
    ):
        changes["geometric_quality"] = 0
    data = record.model_dump()
    data.update(
        changes,
        status="draft",
        runtime=None,
        automatic_acceptance=None,
        invalid_reasons=(),
        modified_at=utc_now(),
    )
    return CalibrationRecord.model_validate(data)


def invalidate_record(record, reason):
    data = record.model_dump()
    data.update(
        status="invalid",
        runtime=None,
        automatic_acceptance=None,
        invalid_reasons=(reason,),
        modified_at=utc_now(),
    )
    return CalibrationRecord.model_validate(data)


def to_runtime_calibration(record):
    record = CalibrationRecord.model_validate(record.model_dump())
    if record.status != "confirmed":
        raise ValueError("A confirmed valid metric record is required")
    return Calibration.model_validate(record.runtime.model_dump())
