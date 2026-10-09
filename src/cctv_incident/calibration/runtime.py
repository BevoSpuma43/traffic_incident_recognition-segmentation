"""Verified metric input and portable run snapshots, including legacy CLI YAML."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .coordinates import ImageTransform
from .homography import Calibration
from .records import CalibrationRecord, to_runtime_calibration
from .repository import (
    compatibility_reasons,
    file_sha256,
    inspect_video,
    load_record,
    load_reference,
)


def _write_reference(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Cannot encode the run calibration reference")
    path.write_bytes(encoded.tobytes())
    return file_sha256(path)


@dataclass(frozen=True)
class RunCalibration:
    original: Calibration
    record: CalibrationRecord | None
    source_path: Path
    source_bytes: bytes
    source_sha256: str
    reference: np.ndarray | None

    def provenance(self):
        record = self.record
        return {
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "format": "video_record" if record else "legacy",
            "record_id": record.record_id if record else None,
            "revision": record.revision if record else None,
            "video": record.video.model_dump(mode="json") if record else None,
            "reference": record.reference.model_dump(mode="json") if record else None,
            "width": record.width.model_dump(mode="json") if record else None,
            "length": record.length.model_dump(mode="json") if record else None,
            "explicit_scale": (
                record.explicit_scale.model_dump(mode="json")
                if record and record.explicit_scale
                else None
            ),
            "automation": record.automation.model_dump(mode="json") if record else None,
            "automatic_acceptance": record.automatic_acceptance if record else None,
            "source_snapshot": "calibration-source.yaml",
            "original_snapshot": "calibration-original.yaml",
            "record_snapshot": "calibration-record.yaml" if record else None,
            "effective_snapshot": "calibration-effective.yaml",
        }

    def write_original(self, run_dir):
        """Keep exact input bytes plus portable geometry and, if available, record."""
        run_dir = Path(run_dir)
        (run_dir / "calibration-source.yaml").write_bytes(self.source_bytes)
        original = Calibration.model_validate(self.original.model_dump())
        reference_hash = None
        if self.reference is not None:
            original.reference_image = "calibration-reference-original.png"
            reference_hash = _write_reference(run_dir / original.reference_image, self.reference)
        original.save(run_dir / "calibration-original.yaml")
        if self.record:
            data = self.record.model_dump(mode="json")
            data["reference"].update(
                image_path=original.reference_image, image_sha256=reference_hash
            )
            data["runtime"] = original.model_dump(mode="json")
            portable = CalibrationRecord.model_validate(data)
            (run_dir / "calibration-record.yaml").write_text(
                yaml.safe_dump(portable.model_dump(mode="json"), sort_keys=False),
                encoding="utf-8",
            )

    def adapt(self, decoded_size, processed_size):
        decoded_size, processed_size = tuple(decoded_size), tuple(processed_size)
        if self.original.image_size == decoded_size:
            basis = "decoded_pixels"
        elif self.record is None and self.original.image_size == processed_size:
            # Pre-existing CLI configurations can already calibrate resized pixels.
            basis = "legacy_processed_pixels"
        else:
            raise ValueError(
                f"Decoded video size {decoded_size} differs from calibration "
                f"{self.original.image_size} (processed size {processed_size})"
            )
        transform = ImageTransform.resize(self.original.image_size, processed_size)
        calibration = transform.adapt_calibration(self.original)
        reference = (
            transform.adapt_reference(self.reference) if self.reference is not None else None
        )
        return calibration, transform, reference, basis

    @staticmethod
    def write_effective(run_dir, calibration, reference=None):
        if reference is not None:
            calibration.reference_image = "calibration-reference-effective.png"
            _write_reference(Path(run_dir) / calibration.reference_image, reference)
        calibration.save(Path(run_dir) / "calibration-effective.yaml")


def load_run_calibration(config):
    """Reject unsafe scale or wrong video/revision before creating inference models."""
    path = Path(config.calibration.file)
    content = path.read_bytes()
    source_sha256 = hashlib.sha256(content).hexdigest()
    data = yaml.safe_load(content)
    record = None
    if isinstance(data, dict) and "schema_version" in data:
        record = load_record(path)
        original = to_runtime_calibration(record)
    else:
        original = Calibration.model_validate(data)
    if original.camera_id != config.calibration.camera_id:
        raise ValueError("Calibration camera_id differs from configuration")
    if original.units != "m":
        raise ValueError("Metric analysis requires calibration units == 'm'")
    if not original.accepted or not original.valid:
        raise ValueError("Accept or correct the camera calibration before starting")
    if original.confidence < config.calibration.min_confidence:
        raise ValueError("Calibration confidence is below the configured minimum")
    if config.calibration.expected_record_id is not None and (
        record is None or record.record_id != config.calibration.expected_record_id
    ):
        raise ValueError("Selected calibration record changed: review it again")
    if config.calibration.expected_revision is not None and (
        record is None or record.revision != config.calibration.expected_revision
    ):
        raise ValueError("Selected calibration revision changed: review it again")
    if record:
        if config.video.source.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError("A local video calibration record cannot be applied to RTSP")
        reasons = compatibility_reasons(
            record, inspect_video(config.video.source, config.project.root_dir)
        )
        if reasons:
            raise ValueError(
                "Calibration belongs to a different or changed video: " + "; ".join(reasons)
            )
    reference = None
    if record:
        reference = load_reference(record, path)
    elif original.reference_image:
        reference_path = (path.parent / original.reference_image).resolve()
        reference = cv2.imdecode(
            np.frombuffer(reference_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR
        )
        if reference is None:
            raise ValueError(f"Calibration reference image missing or corrupt: {reference_path}")
        if (reference.shape[1], reference.shape[0]) != original.image_size:
            raise ValueError("Calibration reference image size differs from calibration")
    if path.read_bytes() != content:
        raise ValueError("Calibration file changed while preparing analysis: review it again")
    return RunCalibration(original, record, path, content, source_sha256, reference)
