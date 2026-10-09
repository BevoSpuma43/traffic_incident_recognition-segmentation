"""Durable calibration proposals, separate from inference jobs and labels."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import psutil

from ..batch import (
    atomic_write,
    exclusive_lock,
    file_signature,
    read_json,
    sha256,
    worker_alive,
    write_json,
)
from ..video_inputs import list_dataset_videos
from .automatic import AutomaticParameters, generate_automatic_proposals
from .proposals import (
    ALGORITHM_VERSION,
    ProposalCandidate,
    ProposalParameters,
    ProposalResult,
    candidate_changes,
    generate_proposals,
)
from .records import Automation, VideoIdentity, accept_automatic_record, edit_record
from .repository import (
    _atomic_bytes,
    compatibility_reasons,
    create_draft,
    find_record,
    find_records,
    load_record,
    save_record,
)


def preparation_signature():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _parameters_hash(parameters):
    return hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()


def prepare_session(
    project_root,
    folder,
    *,
    output_root=None,
    parameters=None,
    automatic_parameters=None,
    model_path=None,
):
    project_root, folder = Path(project_root).resolve(), Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError("Cartella video inesistente")
    names = list_dataset_videos(folder)
    if not names:
        raise ValueError("Nessun video nella cartella")
    videos = []
    for name in names:
        source = folder / name
        signature = file_signature(source)
        digest = sha256(source)
        identity, error = None, None
        try:
            record, _ = create_draft(source, project_root)
            identity = record.video.model_dump(mode="json")
        except (ValueError, OSError) as exc:
            error = str(exc)
        if signature != file_signature(source) or (identity and identity["sha256"] != digest):
            raise ValueError(f"Video cambiato durante la preparazione: {name}")
        videos.append(
            {
                "relative_path": name,
                "sha256": digest,
                "signature": signature,
                "identity": identity,
                "inspection_error": error,
            }
        )
    parameters = ProposalParameters.model_validate(parameters or {}).model_dump()
    automatic = (
        AutomaticParameters.model_validate(automatic_parameters).model_dump(mode="json")
        if automatic_parameters is not None
        else None
    )
    model = Path(model_path).resolve() if model_path else None
    if model is not None and not model.is_file():
        raise ValueError("Pesi locali per la calibrazione non disponibili")
    root = Path(output_root or project_root / "outputs/calibration-preparations").resolve()
    job = root / ("preparation-" + uuid4().hex[:16])
    job.mkdir(parents=True)
    write_json(
        job / "manifest.json",
        {
            "schema_version": 1,
            "session_id": job.name,
            "created_at": time.time(),
            "project_root": str(project_root),
            "folder": str(folder),
            "archive": str(project_root / "data/calibration/videos"),
            "algorithm_version": ALGORITHM_VERSION,
            "code_sha256": preparation_signature(),
            "parameters": parameters,
            "parameters_sha256": _parameters_hash(parameters),
            "automatic_parameters": automatic,
            "automatic_parameters_sha256": _parameters_hash(automatic),
            "model_path": str(model) if model else None,
            "model_sha256": sha256(model) if model else None,
            "videos": videos,
        },
    )
    write_json(
        job / "checkpoint.json",
        {
            "status": "ready",
            "current_index": None,
            "current_video": None,
            "error": None,
            "updated_at": time.time(),
        },
    )
    return job


def list_sessions(root):
    sessions = []
    for path in Path(root).glob("*/manifest.json"):
        try:
            manifest = read_json(path)
            if manifest["schema_version"] == 1 and (path.parent / "checkpoint.json").is_file():
                sessions.append(
                    {
                        "path": str(path.parent),
                        "folder": manifest["folder"],
                        "created_at": manifest["created_at"],
                        "total": len(manifest["videos"]),
                    }
                )
        except (ValueError, KeyError, OSError):
            continue
    return sorted(sessions, key=lambda item: item["created_at"], reverse=True)


def snapshot(job):
    job = Path(job)
    state = read_json(job / "checkpoint.json")
    state["active"] = worker_alive(job)
    state["stop_requested"] = (job / "stop.request").exists()
    state["completed_videos"] = len(list((job / "items").glob("*/result.json")))
    state["total_videos"] = len(read_json(job / "manifest.json")["videos"])
    if not state["active"] and state["status"] in {"starting", "running"}:
        state["status"] = "interrupted"
    return state


def _verify_video(manifest, video):
    source = Path(manifest["folder"]) / video["relative_path"]
    if not source.is_file() or sha256(source) != video["sha256"]:
        raise ValueError(
            f"Video modificato o mancante: {video['relative_path']}. Crea una nuova sessione."
        )


def _verify_result(job, result):
    for relative, expected in result.get("artifacts", {}).items():
        path = (job / relative).resolve()
        if not path.is_relative_to(job.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError("Proposta salvata mancante o modificata: crea una nuova sessione")
    if result.get("record_snapshot"):
        path = Path(result["record_snapshot"])
        if not path.is_file() or sha256(path) != result["record_sha256"]:
            raise ValueError("Revisione archiviata mancante o modificata")
        load_record(path)


def validate_session(job, *, completed_results=False):
    job = Path(job)
    manifest = read_json(job / "manifest.json")
    if completed_results:
        state = snapshot(job)
        if (
            state["active"]
            or state["status"] != "completed"
            or state["completed_videos"] != state["total_videos"]
        ):
            raise ValueError("Completa la preparazione prima di usare i risultati")
    if manifest.get("schema_version") != 1 or (
        not completed_results and manifest["code_sha256"] != preparation_signature()
    ):
        raise ValueError("Formato o codice cambiato: crea una nuova sessione di preparazione")
    if (
        not completed_results and manifest["algorithm_version"] != ALGORITHM_VERSION
    ) or _parameters_hash(manifest["parameters"]) != manifest["parameters_sha256"]:
        raise ValueError("Parametri della proposta cambiati: crea una nuova sessione")
    if manifest.get("automatic_parameters") is not None:
        AutomaticParameters.model_validate(manifest["automatic_parameters"])
        if (
            _parameters_hash(manifest["automatic_parameters"])
            != manifest["automatic_parameters_sha256"]
        ):
            raise ValueError("Ipotesi metriche cambiate: crea una nuova sessione")
        model = manifest.get("model_path")
        if (
            not completed_results
            and model
            and (not Path(model).is_file() or sha256(model) != manifest["model_sha256"])
        ):
            raise ValueError("Pesi della calibrazione modificati: crea una nuova sessione")
    if list_dataset_videos(manifest["folder"]) != [v["relative_path"] for v in manifest["videos"]]:
        raise ValueError("Elenco video cambiato: crea una nuova sessione")
    for video in manifest["videos"]:
        _verify_video(manifest, video)
    for path in (job / "items").glob("*/result.json"):
        _verify_result(job, read_json(path))
    return manifest


def _write_proposal(item, proposal):
    item.mkdir(parents=True, exist_ok=True)
    for name, pixels in (("mask.png", proposal.mask), ("preview.png", proposal.preview)):
        ok, encoded = cv2.imencode(".png", pixels)
        if not ok:
            raise ValueError("Impossibile salvare la diagnostica della proposta")
        _atomic_bytes(item / name, encoded.tobytes())
    evidence = {}
    for digest, pixels in proposal.evidence_images.items():
        if not (
            (len(digest) == 64 and all(c in "0123456789abcdef" for c in digest))
            or (digest.isascii() and digest.isdecimal() and len(digest) <= 16)
        ):
            raise ValueError("Identificativo immagine diagnostica non valido")
        name = f"evidence-{digest}.png"
        ok, encoded = cv2.imencode(".png", pixels)
        if not ok:
            raise ValueError("Impossibile salvare il fotogramma diagnostico")
        _atomic_bytes(item / name, encoded.tobytes())
        evidence[digest] = name
    write_json(
        item / "proposal.json",
        {
            "diagnostics": proposal.diagnostics,
            "candidates": [c.model_dump(mode="json") for c in proposal.candidates],
            "evidence": evidence,
        },
    )


def load_proposal(job, index):
    job = Path(job)
    item = job / "items" / f"{index:06d}"
    if not (item / "proposal.json").is_file() or not (item / "result.json").is_file():
        return None
    result = read_json(item / "result.json")
    _verify_result(job, result)
    if any(
        (item / name).relative_to(job).as_posix() not in result.get("artifacts", {})
        for name in ("proposal.json", "mask.png", "preview.png")
    ):
        return None
    data = read_json(item / "proposal.json")
    mask = cv2.imdecode(
        np.frombuffer((item / "mask.png").read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE
    )
    preview = cv2.imdecode(
        np.frombuffer((item / "preview.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR
    )
    if mask is None or preview is None:
        raise ValueError("Diagnostica della proposta illeggibile")
    evidence = {}
    for digest, name in data.get("evidence", {}).items():
        path = (item / name).resolve()
        if (
            path.parent != item.resolve()
            or path.relative_to(job.resolve()).as_posix() not in result["artifacts"]
        ):
            raise ValueError("Fotogramma diagnostico non verificato")
        pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError("Fotogramma diagnostico illeggibile")
        evidence[digest] = pixels
    return ProposalResult(
        tuple(ProposalCandidate.model_validate(c) for c in data["candidates"]),
        data["diagnostics"],
        mask,
        preview,
        evidence,
    )


def table_rows(job, *, verify_sources=False):
    job = Path(job)
    manifest = read_json(job / "manifest.json")
    identities = [
        VideoIdentity.model_validate(v["identity"]) for v in manifest["videos"] if v["identity"]
    ]
    matches = iter(find_records(identities, manifest["archive"]))
    rows = []
    for index, video in enumerate(manifest["videos"]):
        match = next(matches) if video["identity"] else None
        record = match.record if match else None
        state = (
            (record.status if match.status == "compatible" else match.status) if match else "error"
        )
        if state in {"invalid", "ambiguous"}:
            state = "incompatible"
        error = video["inspection_error"] or ("; ".join(match.reasons) if match else "")
        try:
            source = Path(manifest["folder"]) / video["relative_path"]
            if verify_sources or file_signature(source) != video["signature"]:
                _verify_video(manifest, video)
        except (OSError, ValueError) as exc:
            state, error = "incompatible", str(exc)
        result_path = job / "items" / f"{index:06d}" / "result.json"
        result = read_json(result_path) if result_path.is_file() else {}
        rows.append(
            {
                "index": index,
                "video": video["relative_path"],
                "state": state,
                "width_origin": record.width.origin if record else None,
                "length_origin": record.length.origin if record else None,
                "scale_origin": record.explicit_scale.origin
                if record and record.explicit_scale
                else None,
                "revision": record.revision if record else None,
                "acceptance": ("automatic" if record.automatic_acceptance else "manual")
                if record and record.status == "confirmed"
                else None,
                "reference_type": record.automation.diagnostics.get("selected_candidate", {}).get(
                    "reference_type"
                )
                if record
                else None,
                "width_m": record.width.value if record else None,
                "length_m": record.length.value if record else None,
                "quality": record.geometric_quality if record else None,
                "record_path": str(match.path) if match and match.path else None,
                "preparation": result.get("status", "pending"),
                "message": error or result.get("message", ""),
            }
        )
    return rows


def _record_result(path, status, message=""):
    record = load_record(path)
    history = path.with_name(f"{path.stem}.revision-{record.revision:04d}.yaml")
    # Older archives may have only the current YAML; do not invent a revision snapshot.
    return {
        "status": status,
        "message": message,
        "record_path": str(path),
        "record_id": record.record_id,
        "revision": record.revision,
        "record_snapshot": str(history) if history.is_file() else None,
        "record_sha256": sha256(history) if history.is_file() else None,
    }


def analysis_readiness(job, *, min_quality=0.55, verify_sources=False):
    """The preparation barrier includes failed attempts, then requires manual repairs."""
    state = snapshot(job)
    manifest = read_json(Path(job) / "manifest.json")
    threshold = (manifest.get("automatic_parameters") or {}).get("min_quality", min_quality)
    rows = table_rows(job, verify_sources=verify_sources)
    failed = [r for r in rows if r["state"] != "confirmed" or (r["quality"] or 0) < threshold]
    finished = (
        not state["active"]
        and state["status"] == "completed"
        and state["completed_videos"] == state["total_videos"]
    )
    return {
        "ready": finished and not failed,
        "finished": finished,
        "failed": failed,
        "rows": rows,
        "min_quality": threshold,
    }


def prepared_archive_mapping(job, folder, *, min_quality=0.55):
    from .batch_snapshot import archive_mapping

    manifest = validate_session(job, completed_results=True)
    if Path(folder).resolve() != Path(manifest["folder"]).resolve():
        raise ValueError("La cartella deve coincidere con la preparazione selezionata")
    readiness = analysis_readiness(job, min_quality=min_quality, verify_sources=True)
    if not readiness["finished"]:
        raise ValueError("Completa la calibrazione preliminare di tutti i video prima dell'analisi")
    if readiness["failed"]:
        raise ValueError(
            "Calibra manualmente i video non utilizzabili: "
            + ", ".join(r["video"] for r in readiness["failed"])
        )
    return archive_mapping(
        folder, manifest["project_root"], [v["relative_path"] for v in manifest["videos"]]
    )


def run_preparation(job, *, proposer=None):
    """Commit per video; preserve confirmations, continue after individual failures."""
    job = Path(job).resolve()
    with exclusive_lock(job.parent / ".execution.lock"):
        state = read_json(job / "checkpoint.json")

        def update(**changes):
            state.update(changes, updated_at=time.time())
            write_json(job / "checkpoint.json", state)

        try:
            manifest = validate_session(job)
            automatic = manifest.get("automatic_parameters")

            # Delay loading YOLO until the first vehicle fallback and reuse it.
            class LazySegmenter:
                instance = None

                def predict(self, frame):
                    if self.instance is None:
                        from ..config import Perception
                        from ..segmenter import Segmenter

                        self.instance = Segmenter(
                            Perception(
                                model=Path(manifest["model_path"]),
                                classes=["car", "bus", "truck"],
                                confidence=0.45,
                                max_detections=24,
                            )
                        )
                    return self.instance.predict(frame)

            segmenter = LazySegmenter() if manifest.get("model_path") else None
            update(status="running", error=None)
            for index, video in enumerate(manifest["videos"]):
                item = job / "items" / f"{index:06d}"
                if (item / "result.json").is_file():
                    continue
                if (job / "stop.request").exists():
                    update(status="paused")
                    return
                update(current_index=index + 1, current_video=video["relative_path"])
                _verify_video(manifest, video)
                try:
                    if video["identity"] is None:
                        raise ValueError(video["inspection_error"])
                    identity = VideoIdentity.model_validate(video["identity"])
                    match = find_record(identity, manifest["archive"])
                    if match.status == "compatible" and (
                        not automatic
                        or match.record.status == "confirmed"
                        or match.record.automation.diagnostics.get("preparation_session")
                        == manifest["session_id"]
                    ):
                        result = _record_result(
                            match.path, "kept", "Calibrazione esistente conservata"
                        )
                    elif match.status not in {"missing", "compatible"}:
                        result = {"status": "incompatible", "message": "; ".join(match.reasons)}
                    else:
                        record, image = create_draft(
                            Path(manifest["folder"]) / video["relative_path"],
                            manifest["project_root"],
                        )
                        if compatibility_reasons(record, identity):
                            raise ValueError("Il video è cambiato durante la lettura")
                        if match.status == "compatible":
                            record = match.record
                        if automatic:
                            proposal = (proposer or generate_automatic_proposals)(
                                Path(manifest["folder"]) / video["relative_path"],
                                image,
                                roi=record.roi_px,
                                parameters=manifest["parameters"],
                                automatic_parameters=automatic,
                                model_path=manifest.get("model_path"),
                                segmenter=segmenter,
                            )
                        else:
                            proposal = (proposer or generate_proposals)(
                                image, parameters=manifest["parameters"]
                            )
                        proposal.diagnostics["preparation_session"] = manifest["session_id"]
                        if (job / "stop.request").exists():
                            update(status="paused")
                            return
                        _write_proposal(item, proposal)
                        changes = (
                            candidate_changes(proposal.candidates[0], proposal)
                            if proposal.candidates
                            else {
                                "automation": Automation(
                                    method="auto_assisted",
                                    algorithm_version=manifest["algorithm_version"],
                                    parameters=manifest["parameters"],
                                    diagnostics=proposal.diagnostics,
                                ).model_dump()
                            }
                        )
                        changes["automation"]["diagnostics"]["preparation_session"] = manifest[
                            "session_id"
                        ]
                        record = edit_record(record, **changes)
                        if automatic:
                            rejected = []
                            for candidate in proposal.candidates:
                                try:
                                    if candidate.geometric_quality < automatic["min_quality"]:
                                        raise ValueError("Qualità geometrica insufficiente")
                                    accepted = accept_automatic_record(
                                        edit_record(
                                            record, **candidate_changes(candidate, proposal)
                                        )
                                    )
                                    record = accepted
                                    break
                                except (ValueError, cv2.error) as exc:
                                    rejected.append(str(exc))
                            if record.status != "confirmed":
                                proposal.diagnostics["reason"] = (
                                    "Calibrazione manuale richiesta. "
                                    + (
                                        "; ".join(rejected)
                                        if rejected
                                        else proposal.diagnostics["reason"]
                                    )
                                )
                            _write_proposal(item, proposal)
                        _verify_video(manifest, video)
                        try:
                            path = save_record(
                                record,
                                manifest["archive"],
                                reference_image=image,
                                only_if_missing=match.status == "missing",
                            )
                        except ValueError:
                            # A manual save may have won the archive lock while proposals ran.
                            match = find_record(identity, manifest["archive"])
                            if match.status != "compatible":
                                raise
                            result = _record_result(
                                match.path, "kept", "Salvataggio concorrente conservato"
                            )
                        else:
                            result = _record_result(
                                path,
                                (
                                    "auto_calibrated"
                                    if record.status == "confirmed"
                                    else "manual_required"
                                )
                                if automatic
                                else ("proposed" if proposal.candidates else "no_reference"),
                                proposal.diagnostics["reason"],
                            )
                    # Also recovers a crash after archive save but before result publication.
                    artifacts = [
                        p
                        for p in item.glob("*")
                        if p.name in {"proposal.json", "mask.png", "preview.png"}
                        or p.name.startswith("evidence-")
                    ]
                    result["artifacts"] = {
                        p.relative_to(job).as_posix(): sha256(p) for p in artifacts
                    }
                except (ValueError, OSError, RuntimeError, cv2.error) as exc:
                    result = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
                write_json(item / "result.json", result)
            update(status="completed", current_index=None, current_video=None)
        except Exception as exc:
            update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise


def start_preparation(job):
    job = Path(job).resolve()
    with exclusive_lock(job.parent / ".launch.lock"):
        if any(worker_alive(item["path"]) for item in list_sessions(job.parent)):
            raise RuntimeError("Una preparazione è già attiva: attendi o premi Stop proposte")
        if snapshot(job)["completed_videos"] == snapshot(job)["total_videos"]:
            raise ValueError(
                "Preparazione già completata; revisiona le bozze o crea una nuova sessione"
            )
        try:
            manifest = validate_session(job)
        except (OSError, ValueError) as exc:
            state = read_json(job / "checkpoint.json")
            write_json(job / "checkpoint.json", {**state, "status": "error", "error": str(exc)})
            raise
        (job / "stop.request").unlink(missing_ok=True)
        state = read_json(job / "checkpoint.json")
        write_json(job / "checkpoint.json", {**state, "status": "starting", "error": None})
        try:
            with (job / "worker.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "cctv_incident.calibration.preparation",
                        "--job",
                        str(job),
                    ],
                    cwd=manifest["project_root"],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    start_new_session=os.name != "nt",
                )
            write_json(
                job / "worker.json",
                {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time()},
            )
        except Exception as exc:
            write_json(job / "checkpoint.json", {**state, "status": "error", "error": str(exc)})
            raise
        return process.pid


def stop_preparation(job):
    atomic_write(Path(job) / "stop.request", "stop\n")


def main():
    parser = argparse.ArgumentParser(description="Generate draft calibrations without inference")
    parser.add_argument("--job", required=True)
    job = Path(parser.parse_args().job).resolve()
    for attempt in range(100):
        try:
            with exclusive_lock(job.parent / ".launch.lock"):
                break
        except RuntimeError:
            if attempt == 99:
                raise
            time.sleep(0.05)
    run_preparation(job)


if __name__ == "__main__":
    main()
