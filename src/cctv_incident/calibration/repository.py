"""Video-bound calibration archive with immutable revisions and atomic publication."""

import hashlib
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import av
import cv2
import numpy as np
import yaml

from .records import (
    Automation,
    CalibrationRecord,
    ExplicitScale,
    ReferenceFrame,
    Vertex,
    VideoIdentity,
    utc_now,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(image):
    image = np.asarray(image)
    digest = hashlib.sha256(f"{image.shape}:{image.dtype}".encode())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DecodedReference:
    image: np.ndarray
    frame_index: int
    timestamp_s: float | None
    pts: int | None
    time_base: str | None
    rotation_degrees: float

    def metadata(self):
        return ReferenceFrame(
            frame_index=self.frame_index,
            timestamp_s=self.timestamp_s,
            pts=self.pts,
            time_base=self.time_base,
            pixel_sha256=pixel_sha256(self.image),
        )


def read_first_frame(source):
    """First decodable frame in native PyAV orientation, without resize or seeking.

    frame_index is the decoded-frame ordinal (not an inferred constant-FPS index).
    timestamp_s retains the media time; missing PTS remain explicitly absent.
    Corrupt packets can be skipped while seeking the first decodable frame.
    """
    path = Path(source)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Video is missing or empty")
    index = 0
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("Video has no video stream")
            stream = container.streams.video[0]
            for packet in container.demux(stream):
                try:
                    frames = packet.decode()
                except av.FFmpegError:
                    continue
                for frame in frames:
                    try:
                        image = frame.to_ndarray(format="bgr24")
                    except av.FFmpegError:
                        index += 1
                        continue
                    rotation = float(
                        getattr(frame, "rotation", 0) or stream.metadata.get("rotate", 0)
                    )
                    return DecodedReference(
                        image=image,
                        frame_index=index,
                        timestamp_s=float(frame.time) if frame.time is not None else None,
                        pts=frame.pts,
                        time_base=str(frame.time_base) if frame.time_base else None,
                        rotation_degrees=rotation % 360,
                    )
    except (av.FFmpegError, OSError) as exc:
        raise ValueError(f"Video cannot be decoded: {path.name}") from exc
    raise ValueError("Video contains no decodable frames")


def _inspect(source, project_root=None):
    path = Path(source).resolve()
    if not path.is_file():
        raise ValueError("Video is missing or empty")
    before = path.stat()
    frame = read_first_frame(path)
    digest = file_sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Video changed during inspection")
    relative = None
    if project_root is not None and path.is_relative_to(Path(project_root).resolve()):
        relative = path.relative_to(Path(project_root).resolve()).as_posix()
    identity = VideoIdentity(
        source_path=str(path),
        relative_path=relative,
        original_name=path.name,
        sha256=digest,
        size_bytes=after.st_size,
        image_size=(frame.image.shape[1], frame.image.shape[0]),
        rotation_degrees=frame.rotation_degrees,
        first_frame_sha256=pixel_sha256(frame.image),
    )
    return identity, frame


def inspect_video(source, project_root=None):
    return _inspect(source, project_root)[0]


def create_draft(source, project_root=None):
    identity, frame = _inspect(source, project_root)
    return CalibrationRecord(
        video=identity,
        reference=frame.metadata(),
        camera_id="video_" + identity.sha256[:24],
    ), frame.image


def _reference_path(record_path, image_path):
    # New records use portable references within the record directory (also in snapshots).
    image = Path(image_path)
    root = Path(record_path).resolve().parent
    target = (root / image).resolve()
    if image.is_absolute() or not target.is_relative_to(root):
        raise ValueError("Reference image must be relative and inside the record directory")
    return target


def load_reference(record, path):
    """Validate the exact encoded bytes and decoded pixels returned to the caller."""
    if record.reference.image_path is None:
        raise ValueError("Saved record has no reference image")
    reference = _reference_path(path, record.reference.image_path)
    if not reference.is_file():
        raise ValueError("Reference image is missing or its hash differs")
    content = reference.read_bytes()
    if hashlib.sha256(content).hexdigest() != record.reference.image_sha256:
        raise ValueError("Reference image is missing or its hash differs")
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None or pixel_sha256(image) != record.reference.pixel_sha256:
        raise ValueError("Reference image pixels differ from the inspected frame")
    if (image.shape[1], image.shape[0]) != record.video.image_size:
        raise ValueError("Reference image size differs from the inspected video")
    return image


def load_record(path, *, verify_reference=True):
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "schema_version" not in data:
        raise ValueError("Legacy calibration requires explicit verified association before import")
    record = CalibrationRecord.model_validate(data)
    if verify_reference:
        load_reference(record, path)
    return record


@dataclass(frozen=True)
class CompatibilityResult:
    status: Literal["missing", "compatible", "incompatible", "ambiguous"]
    path: Path | None = None
    record: CalibrationRecord | None = None
    reasons: tuple[str, ...] = ()


def compatibility_reasons(record, identity):
    reasons = []
    for field, message in (
        ("sha256", "Video content hash differs"),
        ("size_bytes", "Video byte size differs"),
        ("image_size", "Decoded resolution differs"),
        ("rotation_degrees", "Video rotation metadata differs"),
        ("first_frame_sha256", "Decoded first frame differs"),
    ):
        if getattr(record.video, field) != getattr(identity, field):
            reasons.append(message)
    return tuple(reasons)


def _current_paths(root):
    return sorted(
        p for p in Path(root).rglob("*.yaml") if not re.search(r"\.revision-\d+\.yaml$", p.name)
    )


def find_record(identity, root="data/calibration/videos"):
    return find_records([identity], root)[0]


def find_records(identities, root="data/calibration/videos"):
    """Inspect the archive once when building a multi-video preparation table."""
    entries = []
    for path in _current_paths(root):
        try:
            entries.append((path, load_record(path), None))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            entries.append((path, None, str(exc)))
    return [_match_record(identity, entries) for identity in identities]


def _match_record(identity, entries):
    compatible, incompatible = [], []
    for path, record, error in entries:
        if record is None:
            if path.stem == _safe_stem(identity.original_name):
                incompatible.append((path, None, (error,)))
            continue
        reasons = compatibility_reasons(record, identity)
        if not reasons:
            compatible.append((path, record))
        elif (
            record.video.source_path.casefold() == identity.source_path.casefold()
            or record.video.original_name.casefold() == identity.original_name.casefold()
        ):
            incompatible.append((path, record, reasons))
    if len(compatible) > 1:
        return CompatibilityResult(
            "ambiguous", reasons=("Multiple records match the video; choose explicitly",)
        )
    if compatible:
        path, record = compatible[0]
        return CompatibilityResult("compatible", path, record)
    if incompatible:
        path, record, reasons = incompatible[0]
        return CompatibilityResult("incompatible", path, record, reasons)
    return CompatibilityResult("missing")


def _safe_stem(name):
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(name).stem).strip(" .")[:80] or "video"
    if re.search(r"\.revision-\d+$", stem):
        stem += "_video"
    if stem.split(".")[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        stem = "video_" + stem
    return stem


@contextmanager
def _save_lock(root):
    with (root / ".save.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Calibration archive is being saved by another process") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_bytes(path, content):
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _save_path(record, root):
    for path in _current_paths(root):
        try:
            existing = load_record(path, verify_reference=False)
        except (OSError, ValueError, yaml.YAMLError):
            # An unrelated damaged record must not prevent saving other videos.
            # Occupied names still never get overwritten below.
            continue
        if existing.record_id == record.record_id:
            existing = load_record(path)
            if existing.revision != record.revision:
                raise ValueError("Stale record revision: reload before saving")
            if existing.video != record.video:
                raise ValueError("A record cannot be reassigned to another video")
            return path, existing
    if record.revision != 0:
        raise ValueError("Saved revision is missing from the archive")
    filename = _safe_stem(record.video.original_name) + ".yaml"
    path = root / filename
    if path.exists():
        key = hashlib.sha256(
            (record.video.source_path + record.video.original_name + record.video.sha256).encode()
        ).hexdigest()[:16]
        path = root / key / filename
        if path.exists():
            raise ValueError("A record already exists for this video; load it before editing")
    return path, None


def save_record(
    record, root="data/calibration/videos", *, reference_image=None, only_if_missing=False
):
    """Publish image → immutable revision → current YAML. Reload returned path to edit again."""
    record = CalibrationRecord.model_validate(record.model_dump())
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _save_lock(root):
        if only_if_missing and find_record(record.video, root).status != "missing":
            raise ValueError("A calibration already exists for this video; automatic save skipped")
        path, previous = _save_path(record, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        if reference_image is None:
            if previous is None:
                raise ValueError("First save requires the decoded reference image")
            reference = _reference_path(path, previous.reference.image_path)
            reference_image = cv2.imdecode(
                np.frombuffer(reference.read_bytes(), np.uint8), cv2.IMREAD_COLOR
            )
        if (
            reference_image is None
            or pixel_sha256(reference_image) != record.reference.pixel_sha256
            or (reference_image.shape[1], reference_image.shape[0]) != record.video.image_size
        ):
            raise ValueError("Reference image must match the inspected video's first frame")
        revision = record.revision + 1
        history = path.with_name(f"{path.stem}.revision-{revision:04d}.yaml")
        # A crash after staging a revision but before updating the current YAML
        # leaves an immutable orphan. Keep it and publish the next free revision.
        while history.exists():
            revision += 1
            history = path.with_name(f"{path.stem}.revision-{revision:04d}.yaml")
        image_name = f"{path.stem}.reference-r{revision:04d}.png"
        image_path = path.with_name(image_name)
        ok, encoded = cv2.imencode(".png", reference_image)
        if not ok:
            raise ValueError("Cannot encode calibration reference image")
        content = encoded.tobytes()
        data = record.model_dump(mode="json")
        data.update(revision=revision, modified_at=utc_now().isoformat())
        data["reference"].update(
            image_path=image_name, image_sha256=hashlib.sha256(content).hexdigest()
        )
        if data["runtime"] is not None:
            data["runtime"]["reference_image"] = image_name
        published = CalibrationRecord.model_validate(data)
        yaml_content = yaml.safe_dump(
            published.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        ).encode("utf-8")
        _atomic_bytes(image_path, content)
        _atomic_bytes(history, yaml_content)
        load_record(history)  # Verify the staged pair before publishing the current pointer.
        _atomic_bytes(path, yaml_content)
        load_record(path)
        return path


def import_legacy_calibration(
    calibration, identity, reference, *, association_verified, scale_origin, scale_source
):
    """Explicit import only; old camera names never imply association to a new video.

    The result is a draft with original ordered metric correspondences. A separate
    confirmation is required, and the supplied provenance remains an assumption
    when scale_origin is standard/experimental.
    """
    if not association_verified:
        raise ValueError("Legacy calibration must be visually associated with this video")
    if (
        calibration.image_size != identity.image_size
        or calibration.units != "m"
        or not calibration.valid
    ):
        raise ValueError(
            "Legacy calibration must be valid, metric and match the decoded resolution"
        )
    return CalibrationRecord(
        video=identity,
        reference=reference,
        camera_id="video_" + identity.sha256[:24],
        geometry_mode="explicit",
        vertices=tuple(
            Vertex(id=f"P{i + 1}", x=p[0], y=p[1])
            for i, p in enumerate(calibration.source_points_px)
        ),
        destination_points=tuple(tuple(p) for p in calibration.destination_points),
        roi_px=tuple(tuple(p) for p in calibration.roi_px),
        explicit_scale=ExplicitScale(origin=scale_origin, source=scale_source, user_confirmed=True),
        geometric_quality=calibration.confidence,
        automation=Automation(
            method="manual", parameters={"legacy_calibration": calibration.model_dump(mode="json")}
        ),
    )
