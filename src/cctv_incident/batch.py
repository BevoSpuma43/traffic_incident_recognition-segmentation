"""Durable sequential batches. A subprocess owns inference, never the Streamlit session."""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import uuid4

import psutil
import yaml

from .batch_evaluation import aggregate_results, compare_video, load_labels
from .calibration.batch_snapshot import (
    InvalidBatchCalibration,
    configure_snapshot,
    create_snapshots,
    csv_provenance,
)
from .config import AppConfig
from .video_inputs import list_dataset_videos

_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.4, 0.4, 0.4)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write(path, content):
    """Publish atomically, allowing up to 1.95 s for transient Windows file locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        for attempt in range(len(_REPLACE_RETRY_DELAYS) + 1):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                # Readers can briefly prevent replacement on Windows, even after
                # the temporary file has been closed. Keep it intact for retries.
                if getattr(exc, "winerror", None) not in (5, 32) or attempt == len(
                    _REPLACE_RETRY_DELAYS
                ):
                    raise
                time.sleep(_REPLACE_RETRY_DELAYS[attempt])
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, data):
    atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_signature():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        if path.name == "batch_ui.py":
            continue
        digest.update(path.relative_to(Path(__file__).parent).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_signature(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@contextmanager
def exclusive_lock(path):
    """OS lock released even on process death (no stale PID lock deletion races)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
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
            raise RuntimeError(
                "Un'altra operazione batch è già in corso. Riprova tra poco."
            ) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare_job(
    config,
    folder,
    metadata,
    output_root=None,
    tolerance_s=1.0,
    *,
    mode=None,
    config_path=None,
    dataset_directory=None,
    dataset_selection=None,
    calibration_map=None,
):
    folder, metadata = Path(folder).resolve(), Path(metadata).resolve()
    if not folder.is_dir():
        raise ValueError(f"Cartella video inesistente: {folder}")
    labels = load_labels(metadata)
    videos = []
    seen = set()
    for relative in list_dataset_videos(folder):
        path = folder / relative
        if path.name in seen:
            raise ValueError(f"Nome video duplicato nella cartella: {path.name}")
        seen.add(path.name)
        if path.name not in labels:
            raise ValueError(f"Video senza etichetta nei metadati: {path.name}")
        videos.append(
            {
                "relative_path": relative,
                "signature": file_signature(path),
                "label": labels[path.name],
            }
        )
    if not videos:
        raise ValueError("La cartella non contiene video supportati.")
    cfg = config.model_copy(deep=True)
    if mode is None:
        mode = "standard_dataset analisi in batch - " + (
            "con omografia" if cfg.events.coordinate_mode == "metric" else "no omografia"
        )
    cfg.video.source, cfg.video.clip_id, cfg.video.max_frames = "", None, None
    cfg.calibration.camera_id = "batch"
    cfg.project.output_dir = Path("outputs")  # Per-attempt directory assigned only by the worker.
    model = cfg.perception.model.resolve()
    cfg.perception.model = model
    if cfg.perception.backend != "synthetic" and not model.is_file():
        raise ValueError(f"Modello locale mancante: {model}")
    protocol = {
        "schema_version": 2,
        "coordinate_mode": cfg.events.coordinate_mode,
        "folder": str(folder),
        "metadata": str(metadata),
        "metadata_sha256": sha256(metadata),
        "model_name": model.name,
        "model_sha256": sha256(model) if model.is_file() else None,
        "code_sha256": code_signature(),
        "config": cfg.model_dump(mode="json"),
        "tolerance_s": tolerance_s,
        "videos": videos,
    }
    # Validate tolerance even before the first inference.
    compare_video(videos[0]["label"], [], tolerance_s)
    yaml_path = Path(config_path).resolve() if config_path is not None else None
    yaml_content = yaml_path.read_text(encoding="utf-8") if yaml_path else None
    root = Path(output_root or (cfg.project.root_dir / "outputs/batches")).resolve()
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model.stem)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job = root / f"{slug}-{timestamp}-{uuid4().hex[:12]}"
    root.mkdir(parents=True, exist_ok=True)
    # Failed preparation never publishes a discoverable/ready experiment.
    with TemporaryDirectory(prefix=".preparing-", dir=root) as staging:
        if cfg.events.coordinate_mode == "metric":
            create_snapshots(cfg, folder, videos, calibration_map, Path(staging))
        else:
            for video in videos:
                video["sha256"] = sha256(folder / video["relative_path"])
        identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
        with exclusive_lock(root / ".launch.lock"):
            job.mkdir(parents=True, exist_ok=False)
            if (Path(staging) / "calibrations").exists():
                os.replace(Path(staging) / "calibrations", job / "calibrations")
            cfg.project.output_dir = job / "pipeline"
            resolved = cfg.model_dump(mode="json")
            validate_manifest({**protocol, "config": resolved}, job)
            settings = {
                "job_id": job.name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "configuration_yaml": str(yaml_path) if yaml_path else None,
                "configuration_yaml_snapshot": "source_config.yaml" if yaml_path else None,
                "yolo_model": str(model),
                "dataset_directory": str(Path(dataset_directory).resolve())
                if dataset_directory
                else None,
                "dataset_selection": dataset_selection,
                "video_directory": str(folder),
                "labels_csv": str(metadata),
                "analysis_fps": cfg.video.target_fps,
                "coordinate_mode": cfg.events.coordinate_mode,
                "tolerance_s": tolerance_s,
                "output_directory": str(job),
                "resolved_configuration": resolved,
            }
            if yaml_content is not None:
                atomic_write(job / "source_config.yaml", yaml_content)
            atomic_write(
                job / "resolved_config.yaml",
                yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
            )
            write_json(job / "batch_config.json", settings)
            write_json(
                job / "manifest.json",
                {
                    **protocol,
                    "config": resolved,
                    "job_id": job.name,
                    "created_at": time.time(),
                    "protocol_sha256": identity,
                    "batch_settings": settings,
                },
            )
            write_json(
                job / "checkpoint.json",
                {"status": "ready", "completed_videos": 0, "total_videos": len(videos)},
            )
            export_results(job)
    return job


def list_jobs(root):
    jobs = []
    for path in Path(root).glob("*/manifest.json"):
        if not all((path.parent / name).is_file() for name in ("checkpoint.json", "metrics.json")):
            continue
        manifest = read_json(path)
        jobs.append(
            {
                "path": str(path.parent),
                "model_name": manifest["model_name"],
                "coordinate_mode": manifest["config"].get("events", {}).get("coordinate_mode"),
                "model_path": manifest["config"]["perception"]["model"],
                "folder": manifest["folder"],
                "created_at": manifest["created_at"],
                "total_videos": len(manifest["videos"]),
            }
        )
    return sorted(jobs, key=lambda row: row["created_at"], reverse=True)


def worker_alive(job):
    path = Path(job) / "worker.json"
    if not path.exists():
        return False
    info = read_json(path)
    try:
        process = psutil.Process(info["pid"])
        return (
            abs(process.create_time() - info["create_time"]) < 0.01
            and process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def snapshot(job):
    job = Path(job)
    state = read_json(job / "checkpoint.json")
    state["active"] = worker_alive(job)
    state["stop_requested"] = (job / "stop.request").exists()
    state["completed_videos"] = len(list((job / "results").glob("*.json")))
    if not state["active"] and state["status"] in {"starting", "running", "stopping"}:
        state["status"] = "interrupted"
    return state


def start_job(job):
    job = Path(job).resolve()
    with exclusive_lock(job.parent / ".launch.lock"):
        for item in list_jobs(job.parent):
            if worker_alive(item["path"]):
                raise RuntimeError("Un batch è già attivo. Premi Stop prima di avviarne un altro.")
        manifest = read_json(job / "manifest.json")
        state = read_json(job / "checkpoint.json")
        if state.get("requires_new_experiment"):
            raise InvalidBatchCalibration("Calibrazione invalidata: prepara un nuovo esperimento.")
        if state["status"] == "completed":
            raise ValueError("Questo batch è già completo.")
        # Refuse before mutating historical artifacts or launching a doomed worker.
        validate_manifest(manifest, job)
        (job / "stop.request").unlink(missing_ok=True)
        state.update(status="starting", error=None, updated_at=time.time())
        write_json(job / "checkpoint.json", state)
        with (job / "worker.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "cctv_incident.batch", "--job", str(job)],
                cwd=manifest["config"]["project"]["root_dir"],
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
    return process.pid


def stop_job(job):
    atomic_write(Path(job) / "stop.request", "stop\n")


def completed_results(job):
    return [read_json(path) for path in sorted((Path(job) / "results").glob("*.json"))]


def write_csv(path, rows, fields):
    content = io.StringIO(newline="")
    writer = csv.DictWriter(content, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, content.getvalue())


def resume_compatibility(job):
    """Cheap read-only UI check; start_job/worker still verify every input in full."""
    manifest = read_json(Path(job) / "manifest.json")
    version = manifest.get("schema_version", 1)
    if version not in (1, 2):
        return False, "Versione del manifest non supportata per la ripresa."
    if manifest.get("code_sha256") != code_signature():
        return False, (
            "Esperimento consultabile. La ripresa richiede il codice originale: "
            "la versione corrente è diversa. Prepara un nuovo esperimento per usarla."
        )
    return True, "Input e snapshot saranno verificati integralmente all'avvio."


def export_results(job, output_dir=None):
    """Rebuild exports, optionally outside the job to preserve historical artifacts."""
    job = Path(job)
    destination = Path(output_dir) if output_dir is not None else job
    manifest = read_json(job / "manifest.json")
    results = completed_results(job)
    metrics = aggregate_results(results, len(manifest["videos"]), manifest["tolerance_s"])
    common = {
        "job_id": manifest["job_id"],
        "model": manifest["model_name"],
        "model_sha256": manifest["model_sha256"],
        "coordinate_mode": manifest.get("coordinate_mode"),
    }
    video_rows, event_rows = [], []
    for result in results:
        label, comparison = result["label"], result["comparison"]
        shared = {
            **common,
            "video": result["video"],
            "run_id": result["summary"]["run_id"],
            "label_type": label["type"],
            "truth_time_s": label["accident_time_s"],
            **csv_provenance(result),
        }
        video_rows.append(
            {
                **shared,
                "truth_positive": int(label["positive"]),
                "predicted_positive": int(comparison["predicted_positive"]),
                "video_outcome": comparison["video_outcome"],
                "event_tp": comparison["event_tp"],
                "event_fp": comparison["event_fp"],
                "event_fn": comparison["event_fn"],
                "predicted_events": len(comparison["events"]),
                "predicted_times_s": json.dumps([e["impact_time_s"] for e in comparison["events"]]),
                "duration_s": result["summary"]["source_duration_s"],
                "elapsed_s": result["summary"]["elapsed_s"],
                "video_sha256": result["video_sha256"],
                "split_in_distribution": label["split_in_distribution"],
                "split_geo_aware": label["split_geo_aware"],
            }
        )
        for event in comparison["events"]:
            event_rows.append({**shared, **event, "track_ids": json.dumps(event["track_ids"])})
    base = [*common, "video", "run_id", "label_type", "truth_time_s", *csv_provenance({})]
    write_csv(
        destination / "videos.csv",
        video_rows,
        base
        + [
            "truth_positive",
            "predicted_positive",
            "video_outcome",
            "event_tp",
            "event_fp",
            "event_fn",
            "predicted_events",
            "predicted_times_s",
            "duration_s",
            "elapsed_s",
            "video_sha256",
            "split_in_distribution",
            "split_geo_aware",
        ],
    )
    write_csv(
        destination / "events.csv",
        event_rows,
        base
        + [
            "event_id",
            "impact_time_s",
            "confirm_time_s",
            "score",
            "match",
            "time_error_s",
            "track_ids",
        ],
    )
    metric_rows = [
        {
            **common,
            "unit": unit,
            "complete": metrics["complete"],
            "completed_videos": len(results),
            "total_videos": len(manifest["videos"]),
            "tolerance_s": manifest["tolerance_s"],
            **metrics[unit],
        }
        for unit in ("video", "event")
    ]
    write_csv(destination / "metrics.csv", metric_rows, list(metric_rows[0]))
    write_json(destination / "metrics.json", {**common, **metrics})
    return metrics


def validate_manifest(manifest, job=None):
    version = manifest.get("schema_version", 1)
    if version not in (1, 2):
        raise ValueError("Versione manifest batch non supportata")
    config = AppConfig.model_validate(manifest["config"])
    if version >= 2 and manifest.get("coordinate_mode") != config.events.coordinate_mode:
        raise ValueError("Modalità del manifest incoerente con la configurazione")
    if config.events.coordinate_mode == "metric" and version < 2:
        raise InvalidBatchCalibration(
            "Il batch metrico richiede snapshot: prepara un nuovo esperimento."
        )
    job = Path(job) if job is not None else config.project.output_dir.parent
    if code_signature() != manifest["code_sha256"]:
        raise ValueError(
            "Il codice è cambiato: prepara un nuovo batch per non mescolare risultati."
        )
    if sha256(manifest["metadata"]) != manifest["metadata_sha256"]:
        raise ValueError("I metadati sono cambiati: prepara un nuovo batch.")
    if (
        manifest["model_sha256"]
        and sha256(manifest["config"]["perception"]["model"]) != manifest["model_sha256"]
    ):
        raise ValueError("I pesi del modello sono cambiati: prepara un nuovo batch.")
    for video in manifest["videos"]:
        if (
            version >= 2
            and sha256(Path(manifest["folder"]) / video["relative_path"]) != video["sha256"]
        ):
            raise ValueError(f"Video modificato: {video['relative_path']}. Prepara un nuovo batch.")
        if config.events.coordinate_mode == "metric":
            configure_snapshot(config, job, video)
        if file_signature(Path(manifest["folder"]) / video["relative_path"]) != video["signature"]:
            raise ValueError(f"Video modificato: {video['relative_path']}. Prepara un nuovo batch.")


def run_job(job, pipeline_factory=None, segmenter_factory=None):
    """Commit only full videos. Stopped/crashed attempts are excluded and retried from zero."""
    from .pipeline import Pipeline
    from .segmenter import create_segmenter
    from .storage import EventStorage

    job = Path(job).resolve()
    manifest = read_json(job / "manifest.json")
    state = read_json(job / "checkpoint.json")

    def save_state(**changes):
        state.update(changes, updated_at=time.time())
        write_json(job / "checkpoint.json", state)

    try:
        if state.get("requires_new_experiment"):
            raise InvalidBatchCalibration("Calibrazione invalidata: prepara un nuovo esperimento.")
        validate_manifest(manifest, job)
        metrics = export_results(job)  # Recover a crash between result commit and CSV export.
        save_state(status="running", completed_videos=metrics["completed_videos"], error=None)
        config = AppConfig.model_validate(manifest["config"])
        segmenter = None
        for index, video in enumerate(manifest["videos"]):
            result_path = job / "results" / f"{index:06d}.json"
            if result_path.exists():
                continue
            save_state(
                current_index=index + 1,
                current_video=video["relative_path"],
                progress=0.0,
                timestamp_s=0.0,
                status="running",
            )
            if (job / "stop.request").exists():
                save_state(status="paused")
                return
            source = Path(manifest["folder"]) / video["relative_path"]
            if file_signature(source) != video["signature"]:
                raise ValueError(f"Video modificato durante il batch: {source.name}")
            video_hash = sha256(source)
            if video.get("sha256", video_hash) != video_hash:
                raise ValueError(f"Video modificato durante il batch: {source.name}")
            if segmenter is None:
                segmenter = (segmenter_factory or create_segmenter)(config.perception)
                if hasattr(segmenter, "model") and segmenter.model.task != "segment":
                    raise ValueError("Seleziona pesi YOLO di segmentazione (*-seg.pt).")
            cfg = config.model_copy(deep=True)
            cfg.video.source, cfg.video.clip_id = str(source), source.stem
            cfg.calibration.camera_id = "source_" + re.sub(r"[^a-zA-Z0-9_-]", "_", source.stem)[:80]
            cfg.project.output_dir = job / "pipeline"
            if cfg.events.coordinate_mode == "metric":
                configure_snapshot(cfg, job, video)
            last_progress = 0.0
            validity = {"lost": False}

            def progress(
                data,
                fallback_duration=video["label"]["duration_s"],
                metric=cfg.events.coordinate_mode == "metric",
                validity=validity,
            ):
                nonlocal last_progress
                if metric and data.get("calibration_valid") is False:
                    validity["lost"] = True
                if time.monotonic() - last_progress < 0.5:
                    return
                duration = data["duration_s"] or fallback_duration
                save_state(
                    progress=min(0.999, data["timestamp_s"] / duration),
                    timestamp_s=data["timestamp_s"],
                    duration_s=duration,
                    run_id=data["run_id"],
                    processed_frames=data["processed_frames"],
                )
                last_progress = time.monotonic()

            summary = (pipeline_factory or Pipeline)(cfg, segmenter=segmenter).run(
                stop_requested=lambda validity=validity: (
                    validity["lost"] or (job / "stop.request").exists()
                ),
                progress_callback=progress,
            )
            if cfg.events.coordinate_mode == "metric":
                configure_snapshot(cfg, job, video)
                if (
                    validity["lost"]
                    or not summary.get("calibration_valid")
                    or not summary.get("metric_calibration_available")
                ):
                    raise InvalidBatchCalibration(
                        "Calibrazione persa durante l'analisi: "
                        + "; ".join(
                            summary.get("non_evaluable_reasons") or ["geometria non valida"]
                        )
                        + ". Nessun risultato conteggiato; prepara un nuovo esperimento."
                    )
            if summary.get("stopped"):
                save_state(status="paused")
                return
            if not summary.get("completed") or summary.get("error"):
                raise RuntimeError("Video non completato: il risultato non viene conteggiato.")
            if not summary.get("evaluable", cfg.events.coordinate_mode == "image"):
                raise RuntimeError("Video non valutabile: nessun risultato conteggiato.")
            if file_signature(source) != video["signature"] or sha256(source) != video_hash:
                raise ValueError("Video modificato durante l'analisi; risultato non conteggiato.")
            storage = EventStorage(cfg.project.output_dir)
            try:
                predictions = storage.list_events(summary["run_id"], limit=1_000_000)
            finally:
                storage.close()
            comparison = compare_video(video["label"], predictions, manifest["tolerance_s"])
            write_json(
                result_path,
                {
                    "video": video["relative_path"],
                    "video_sha256": video_hash,
                    "coordinate_mode": cfg.events.coordinate_mode,
                    "calibration": video.get("calibration"),
                    "label": video["label"],
                    "comparison": comparison,
                    "summary": summary,
                },
            )
            metrics = export_results(job)
            save_state(completed_videos=metrics["completed_videos"], progress=1.0)
        save_state(status="completed", progress=1.0)
    except Exception as exc:
        save_state(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            requires_new_experiment=isinstance(exc, InvalidBatchCalibration),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description="Resume a prepared sequential ACCIDENT batch")
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = Path(args.job).resolve()
    # Wait for the launcher's short critical section without racing worker.json creation.
    for attempt in range(100):
        try:
            with exclusive_lock(job.parent / ".launch.lock"):
                pass
            break
        except RuntimeError:
            if attempt == 99:
                raise
            time.sleep(0.05)
    with exclusive_lock(job.parent / ".execution.lock"):
        run_job(job)


if __name__ == "__main__":
    main()
