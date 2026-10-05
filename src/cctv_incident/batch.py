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
from pathlib import Path
from tempfile import NamedTemporaryFile

import psutil

from .batch_evaluation import aggregate_results, compare_video, load_labels
from .config import AppConfig
from .video_inputs import list_dataset_videos


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write(path, content):
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
        os.replace(temporary, path)
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


def prepare_job(config, folder, metadata, output_root=None, tolerance_s=1.0):
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
    if cfg.events.coordinate_mode != "image":
        raise ValueError("Il batch ACCIDENT richiede coordinate immagine.")
    cfg.video.source, cfg.video.clip_id, cfg.video.max_frames = "", None, None
    cfg.calibration.camera_id = "batch"
    cfg.project.output_dir = Path("outputs")  # Per-attempt directory assigned only by the worker.
    model = cfg.perception.model.resolve()
    cfg.perception.model = model
    if cfg.perception.backend != "synthetic" and not model.is_file():
        raise ValueError(f"Modello locale mancante: {model}")
    protocol = {
        "schema_version": 1,
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
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(output_root or (cfg.project.root_dir / "outputs/batches")).resolve()
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model.stem)
    job = root / f"{slug}-{identity}"
    with exclusive_lock(root / ".launch.lock"):
        if not (job / "manifest.json").exists():
            write_json(
                job / "manifest.json", {**protocol, "job_id": job.name, "created_at": time.time()}
            )
        if not (job / "checkpoint.json").exists():
            write_json(
                job / "checkpoint.json",
                {"status": "ready", "completed_videos": 0, "total_videos": len(videos)},
            )
        if not worker_alive(job):
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
        if state["status"] == "completed":
            raise ValueError("Questo batch è già completo.")
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


def export_results(job):
    job = Path(job)
    manifest = read_json(job / "manifest.json")
    results = completed_results(job)
    metrics = aggregate_results(results, len(manifest["videos"]), manifest["tolerance_s"])
    common = {
        "job_id": manifest["job_id"],
        "model": manifest["model_name"],
        "model_sha256": manifest["model_sha256"],
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
    base = [*common, "video", "run_id", "label_type", "truth_time_s"]
    write_csv(
        job / "videos.csv",
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
        job / "events.csv",
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
    write_csv(job / "metrics.csv", metric_rows, list(metric_rows[0]))
    write_json(job / "metrics.json", {**common, **metrics})
    return metrics


def validate_manifest(manifest):
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
        validate_manifest(manifest)
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
            if segmenter is None:
                segmenter = (segmenter_factory or create_segmenter)(config.perception)
                if hasattr(segmenter, "model") and segmenter.model.task != "segment":
                    raise ValueError("Seleziona pesi YOLO di segmentazione (*-seg.pt).")
            cfg = config.model_copy(deep=True)
            cfg.video.source, cfg.video.clip_id = str(source), source.stem
            cfg.calibration.camera_id = "source_" + re.sub(r"[^a-zA-Z0-9_-]", "_", source.stem)[:80]
            cfg.project.output_dir = job / "pipeline"
            last_progress = 0.0

            def progress(data, fallback_duration=video["label"]["duration_s"]):
                nonlocal last_progress
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
                stop_requested=lambda: (job / "stop.request").exists(),
                progress_callback=progress,
            )
            if summary.get("stopped"):
                save_state(status="paused")
                return
            if not summary.get("completed") or summary.get("error"):
                raise RuntimeError("Video non completato: il risultato non viene conteggiato.")
            if file_signature(source) != video["signature"]:
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
                    "label": video["label"],
                    "comparison": comparison,
                    "summary": summary,
                },
            )
            metrics = export_results(job)
            save_state(completed_videos=metrics["completed_videos"], progress=1.0)
        save_state(status="completed", progress=1.0)
    except Exception as exc:
        save_state(status="error", error=f"{type(exc).__name__}: {exc}")
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
