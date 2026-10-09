"""Persistent, frozen video samples shared by image and metric batch experiments."""

import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .batch import read_json, sha256, write_json
from .calibration.batch_snapshot import configure_snapshot, create_snapshots
from .config import AppConfig


def selection_root(project_root):
    return Path(project_root).resolve() / "outputs/batch-selections"


def active_selection(project_root):
    """Missing selection means full-folder mode; a corrupt pointer must fail closed."""
    root = selection_root(project_root)
    pointer = root / "active.json"
    if not pointer.is_file():
        return None
    data = read_json(pointer)
    path = (root / data["selection_id"] / "manifest.json").resolve()
    if not path.is_relative_to(root) or sha256(path) != data["manifest_sha256"]:
        raise ValueError("Campione condiviso modificato: selezionalo nuovamente dal riepilogo")
    load_selection(path)
    return path


def load_selection(path, *, verify_sources=False, verify_calibrations=False):
    path = Path(path).resolve()
    data = read_json(path)
    videos = data.get("videos", [])
    names = [v["relative_path"] for v in videos]
    if data.get("schema_version") != 1 or not names or len(set(names)) != len(names):
        raise ValueError("Campione condiviso vuoto o non valido")
    folder = Path(data["folder"]).resolve()
    cfg = AppConfig()
    cfg.calibration.min_confidence = data["min_quality"]
    for video in videos:
        source = (folder / video["relative_path"]).resolve()
        if not source.is_relative_to(folder) or Path(video["relative_path"]).is_absolute():
            raise ValueError("Percorso video esterno al campione")
        if verify_sources and (not source.is_file() or sha256(source) != video["sha256"]):
            raise ValueError(f"Video del campione modificato o mancante: {video['relative_path']}")
        if verify_calibrations:
            configure_snapshot(cfg, path.parent, video)
    return data


def create_automatic_selection(job):
    """Freeze only usable automatic records, including portable calibration snapshots."""
    from .calibration.preparation import analysis_readiness, validate_session

    manifest = validate_session(job, completed_results=True)
    readiness = analysis_readiness(job)
    selected = [
        r
        for r in readiness["rows"]
        if r["state"] == "confirmed"
        and r["acceptance"] == "automatic"
        and (r["quality"] or 0) >= readiness["min_quality"]
    ]
    if not selected:
        raise ValueError("Nessun video con calibrazione automatica utilizzabile")
    names = {r["video"] for r in selected}
    videos = [
        {"relative_path": v["relative_path"], "sha256": v["sha256"]}
        for v in manifest["videos"]
        if v["relative_path"] in names
    ]
    root = selection_root(manifest["project_root"])
    root.mkdir(parents=True, exist_ok=True)
    identifier = "sample-" + uuid4().hex[:16]
    destination = root / identifier
    cfg = AppConfig()
    cfg.project.root_dir = Path(manifest["project_root"])
    cfg.calibration.min_confidence = readiness["min_quality"]
    with TemporaryDirectory(prefix=".preparing-", dir=root) as staging:
        mapping = {r["video"]: r["record_path"] for r in selected}
        create_snapshots(cfg, manifest["folder"], videos, mapping, staging)
        # A concurrent manual edit must not silently enter the automatic sample.
        expected = {v["relative_path"]: v["sha256"] for v in manifest["videos"]}
        for video in videos:
            if (
                not video["calibration"].get("automatic_acceptance")
                or video["sha256"] != expected[video["relative_path"]]
            ):
                raise ValueError(
                    "Calibrazione o video cambiato durante la selezione: aggiorna il riepilogo"
                )
        data = {
            "schema_version": 1,
            "selection_id": identifier,
            "created_at": time.time(),
            "preparation_session": manifest["session_id"],
            "preparation_manifest_sha256": sha256(Path(job) / "manifest.json"),
            "folder": manifest["folder"],
            "min_quality": readiness["min_quality"],
            "original_count": len(manifest["videos"]),
            "videos": videos,
            "excluded": [
                {
                    "relative_path": r["video"],
                    "reason": r["message"] or "Calibrazione automatica non utilizzabile",
                }
                for r in readiness["rows"]
                if r["video"] not in names
            ],
        }
        write_json(Path(staging) / "manifest.json", data)
        if not Path(staging).resolve().is_relative_to(
            root.resolve()
        ) or not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError("Destinazione del campione esterna alla cartella prevista")
        os.replace(staging, destination)
    path = destination / "manifest.json"
    write_json(root / "active.json", {"selection_id": identifier, "manifest_sha256": sha256(path)})
    return path


def selection_provenance(path, data):
    return {
        "selection_id": data["selection_id"],
        "manifest_sha256": sha256(path),
        "preparation_session": data["preparation_session"],
        "folder": data["folder"],
        "original_count": data["original_count"],
        "excluded": data["excluded"],
        "videos": [
            {"relative_path": v["relative_path"], "sha256": v["sha256"]} for v in data["videos"]
        ],
    }
