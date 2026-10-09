"""Verified, portable calibration inputs for immutable batch experiments."""

from pathlib import Path

import yaml

from .repository import file_sha256, find_records, inspect_video, load_record
from .runtime import load_run_calibration


class InvalidBatchCalibration(ValueError):
    """The experiment needs new calibration snapshots, never an in-place repair."""


def archive_mapping(folder, project_root, relatives):
    """Resolve every video explicitly; never silently omit an uncalibrated video."""
    identities = [inspect_video(Path(folder) / name, project_root) for name in relatives]
    matches = find_records(identities, Path(project_root) / "data/calibration/videos")
    mapping, errors = {}, []
    for name, match in zip(relatives, matches, strict=True):
        if match.status != "compatible" or match.record.status != "confirmed":
            errors.append(f"{name}: calibrazione confermata mancante o incompatibile")
        else:
            mapping[name] = match.path
    if errors:
        raise InvalidBatchCalibration(
            "Revisiona le calibrazioni prima del batch:\n" + "\n".join(errors)
        )
    return mapping


def create_snapshots(config, folder, videos, mapping, destination):
    expected = {v["relative_path"] for v in videos}
    if mapping is None or set(mapping) != expected:
        raise InvalidBatchCalibration(
            "Serve una mappa video → calibrazione completa ed esatta per la cartella."
        )
    for index, video in enumerate(videos):
        name = video["relative_path"]
        try:
            path = Path(mapping[name]).resolve()
            record = load_record(path)
            cfg = config.model_copy(deep=True)
            cfg.video.source = str(Path(folder) / name)
            cfg.calibration.file = path
            cfg.calibration.camera_id = record.camera_id
            cfg.calibration.expected_record_id = record.record_id
            cfg.calibration.expected_revision = record.revision
            run = load_run_calibration(cfg)
            record = run.record
            directory = Path(destination) / "calibrations" / f"{index:06d}"
            directory.mkdir(parents=True)
            run.write_original(directory)
            local = directory / "calibration-record.yaml"
            files = {
                p.relative_to(destination).as_posix(): file_sha256(p)
                for p in sorted(directory.iterdir())
            }
            video["calibration"] = {
                "path": local.relative_to(destination).as_posix(),
                "sha256": file_sha256(local),
                "files": files,
                "camera_id": record.camera_id,
                **run.provenance(),
            }
            video["sha256"] = record.video.sha256
            # Check the portable file, including relative reference paths, before publication.
            configure_snapshot(cfg, destination, video)
            load_run_calibration(cfg)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise InvalidBatchCalibration(f"{name}: {exc}. Prepara un nuovo esperimento.") from exc


def configure_snapshot(config, job, video):
    """Verify all immutable artifacts and select the portable record for this video."""
    try:
        data = video["calibration"]
        root = Path(job).resolve()
        if not data or data["files"].get(data["path"]) != data["sha256"]:
            raise ValueError("Riferimento o hash dello snapshot mancante")
        for relative, expected in data["files"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or file_sha256(path) != expected:
                raise ValueError(f"Snapshot calibrazione modificato: {relative}")
        path = root / data["path"]
        record = load_record(path)
        if (
            record.record_id != data["record_id"]
            or record.revision != data["revision"]
            or record.camera_id != data["camera_id"]
            or record.video.model_dump(mode="json") != data["video"]
            or record.video.sha256 != video["sha256"]
            or record.status != "confirmed"
            or not record.runtime.metric_valid(config.calibration.min_confidence)
        ):
            raise ValueError("Identità, revisione o validità dello snapshot incoerente")
        config.calibration.file = path
        config.calibration.camera_id = record.camera_id
        config.calibration.expected_record_id = record.record_id
        config.calibration.expected_revision = record.revision
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as exc:
        raise InvalidBatchCalibration(
            f"Calibrazione di {video['relative_path']} non valida: {exc}. "
            "Prepara un nuovo esperimento."
        ) from exc


def csv_provenance(result):
    """Absent historical provenance stays absent, including on regenerated exports."""
    data = result.get("calibration") or {}
    return {
        "calibration_path": data.get("path"),
        "calibration_sha256": data.get("sha256"),
        "calibration_record_id": data.get("record_id"),
        "calibration_revision": data.get("revision"),
        "width_origin": (data.get("width") or {}).get("origin"),
        "length_origin": (data.get("length") or {}).get("origin"),
        "explicit_scale_origin": (data.get("explicit_scale") or {}).get("origin"),
    }
