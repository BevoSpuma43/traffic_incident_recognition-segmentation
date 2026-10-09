"""Editor state and bounded metric preview, independent of Streamlit reruns."""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .coordinates import rectangle_destination, validate_quad
from .homography import estimate_calibration
from .records import CalibrationRecord, Vertex, edit_record

EDITOR_FIELDS = (
    "vertices",
    "roi_px",
    "width",
    "length",
    "geometry_mode",
    "destination_points",
    "explicit_scale",
    "automation",
    "geometric_quality",
)


@dataclass
class EditorSession:
    fingerprint: tuple
    record: CalibrationRecord
    frame: np.ndarray
    path: Path | None = None
    locked: bool = False
    visually_accepted: bool = False
    epoch: int = 0
    history: list[dict] = field(default_factory=list)
    last_event: str | None = None
    input_error: str | None = None
    proposal_result: object | None = None
    proposal_generation: int = 0
    workflow: str = "manual"
    applied_proposal: tuple[int, str] | None = None
    proposal_error: str | None = None

    def change(self, **changes):
        data = self.record.model_dump()
        new = edit_record(self.record, **changes)
        normalized = new.model_dump()
        if all(data.get(k) == normalized[k] for k in changes):
            return False
        self.history.append({k: data[k] for k in EDITOR_FIELDS})
        self.history = self.history[-50:]
        self.record = new
        if "roi_px" in changes:
            self.proposal_result = None
        self.visually_accepted = False
        self.input_error = None
        self.epoch += 1
        return True

    def undo(self):
        if not self.history:
            return
        self.record = edit_record(self.record, **self.history.pop())
        self.proposal_result = None
        self.visually_accepted = False
        self.input_error = None
        self.epoch += 1

    def receive(self, payload):
        if not isinstance(payload, dict) or self.locked or not payload.get("event_id"):
            return False
        if payload.get("event_id") == self.last_event:
            return False
        record = self.record
        expected = (record.record_id, record.video.sha256, record.revision, self.epoch)
        actual = tuple(payload.get(k) for k in ("record_id", "source_sha256", "revision", "epoch"))
        if actual != expected:
            return False
        self.last_event = payload.get("event_id")
        if payload.get("mode") == "roi":
            roi = np.asarray(payload["roi_px"], dtype=float)
            if roi.size and (
                roi.ndim != 2
                or roi.shape[1] != 2
                or len(roi) > 32
                or not np.isfinite(roi).all()
                or (roi < 0).any()
                or (roi > np.asarray(record.video.image_size) - 1).any()
            ):
                raise ValueError("ROI points fall outside the original video or exceed 32 vertices")
            return self.change(roi_px=payload["roi_px"])
        if payload.get("mode") != "calibration":
            raise ValueError("Unknown calibration editor mode")
        vertices = tuple(Vertex.model_validate(v) for v in payload["vertices"])
        size = np.asarray(record.video.image_size) - 1
        if any(v.x < 0 or v.y < 0 or v.x > size[0] or v.y > size[1] for v in vertices):
            raise ValueError("Editor points fall outside the original video")
        return self.change(vertices=[v.model_dump() for v in vertices])


def geometric_quality(record):
    """Heuristic of area and normalized conditioning; no claim about metric accuracy."""
    points = np.asarray(record.points_px, dtype=float)
    if record.geometry_mode == "rectangle":
        validate_quad(points, record.video.image_size)
    if len(points) < 4:
        raise ValueError("Seleziona almeno quattro punti")
    normalized = points / np.asarray(record.video.image_size)
    area = abs(cv2.contourArea(cv2.convexHull(normalized.astype(np.float32))))
    if record.geometry_mode == "rectangle":
        h = cv2.getPerspectiveTransform(
            normalized.astype(np.float32), np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        )
    else:
        destination = np.asarray(record.destination_points, dtype=float)
        if destination.shape != points.shape or not np.isfinite(destination).all():
            raise ValueError("Corrispondenze metriche incomplete")
        span = np.ptp(destination, axis=0)
        if min(span) <= 0:
            raise ValueError("Corrispondenze metriche degeneri")
        h, _ = cv2.findHomography(normalized, (destination - destination.min(axis=0)) / span)
        if h is None:
            raise ValueError("Corrispondenze metriche degeneri")
    condition = np.linalg.cond(h)
    quality = float(0.79 * min(1, area / 0.05) * min(1, 50 / max(condition, 1)))
    selected = record.automation.diagnostics.get("selected_candidate", {})
    if selected.get("reference_type") == "vehicle_reference":
        # Dragging or accepting an estimated plane does not verify its scale.
        quality = min(quality, 0.6)
    return quality


def metric_preview(frame, record, max_size=640):
    """Render at most max_size² pixels, with at most 40 grid lines per axis."""
    if not 64 <= max_size <= 1024:
        raise ValueError("Preview size must be between 64 and 1024")
    if record.geometry_mode == "rectangle":
        validate_quad(record.points_px, record.video.image_size)
        if record.width.value is None or record.length.value is None:
            raise ValueError("Inserisci larghezza e lunghezza per vedere l'anteprima")
        destination = rectangle_destination(record.width.value, record.length.value)
    else:
        destination = record.destination_points
    calibration = estimate_calibration(
        record.camera_id,
        record.video.image_size,
        record.points_px,
        destination,
        accepted=False,
    )
    bounds = np.asarray(destination)
    minimum, maximum = bounds.min(axis=0), bounds.max(axis=0)
    span = maximum - minimum
    if not np.isfinite(span).all() or min(span) <= 0:
        raise ValueError("Estensione metrica non valida")
    scale = (max_size - 40) / max(span)
    size = tuple(int(np.clip(np.ceil(v * scale) + 40, 64, max_size)) for v in span)
    view = np.array(
        [[scale, 0, 20 - scale * minimum[0]], [0, scale, 20 - scale * minimum[1]], [0, 0, 1]]
    )
    image = cv2.warpPerspective(frame, view @ np.asarray(calibration.homography), size)
    step = 10 ** np.ceil(np.log10(max(span) / 15))
    for axis in (0, 1):
        start = np.ceil(minimum[axis] / step) * step
        for value in np.arange(start, maximum[axis] + step * 0.001, step)[:40]:
            pixel = int(round(20 + scale * (value - minimum[axis])))
            ends = (
                ((pixel, 20), (pixel, size[1] - 20))
                if axis == 0
                else ((20, pixel), (size[0] - 20, pixel))
            )
            cv2.line(image, *ends, (100, 180, 100), 1)
    return image, float(step)
