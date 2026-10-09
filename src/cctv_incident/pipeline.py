import hashlib
import json
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import psutil

from .calibration.coordinates import ImageTransform
from .calibration.homography import image_reference
from .calibration.quality import CameraMotionGuard
from .calibration.runtime import load_run_calibration
from .clip_buffer import ClipBuffer
from .event_detector import EventDetector
from .features import compute_pairs
from .ground_point import ground_point
from .image_event_detector import ImageEventDetector
from .metrics import Profiler
from .segmenter import create_segmenter
from .storage import EventStorage
from .tracker import VehicleTracker
from .trajectories import TrajectoryStore
from .types import Observation
from .video import VideoSource
from .visualization import render_bird_eye, render_frame


def hardware_info():
    result = {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / 1024**3,
        "python": platform.python_version(),
    }
    try:
        import torch

        result.update(
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_version=torch.version.cuda,
        )
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return result


def hash_file(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Pipeline:
    def __init__(self, config, segmenter=None):
        self.config = config
        self.segmenter = segmenter
        self.summary = None

    def run(self, callback=None, stop_requested=None, progress_callback=None):
        cfg = self.config
        np.random.seed(cfg.project.seed)
        cv2.setRNGSeed(cfg.project.seed)
        image_mode = cfg.events.coordinate_mode == "image"
        calibration_input = None if image_mode else load_run_calibration(cfg)
        calibration = calibration_input.original if calibration_input else None
        reference = calibration_input.reference if calibration_input else None

        run_id = uuid4().hex[:16]
        clip_id = cfg.video.clip_id or (
            cfg.calibration.camera_id
            if cfg.video.source.startswith(("rtsp://", "rtsps://"))
            else Path(cfg.video.source).stem
        )
        output = cfg.project.output_dir
        run_dir = output / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if calibration_input:
            calibration_input.write_original(run_dir)
        profiler = Profiler()
        started = time.perf_counter()
        segmenter = self.segmenter or create_segmenter(cfg.perception)
        tracker = VehicleTracker(cfg.tracking, cfg.video.target_fps)
        trajectories = TrajectoryStore(cfg.tracking, cfg.features, cfg.events.coordinate_mode)
        classifier = None
        if cfg.events.classifier:
            import joblib

            from .classifier import FEATURE_NAMES, TemporalClassifier

            artifact = joblib.load(cfg.events.classifier)
            if artifact["feature_names"] != FEATURE_NAMES:
                raise ValueError("Classifier feature schema mismatch")
            if artifact.get("window_seconds") != cfg.features.window_seconds:
                raise ValueError("Classifier window duration differs from features.window_seconds")
            classifier = TemporalClassifier(artifact["model"], cfg.features.window_seconds)
        detector = (
            ImageEventDetector(cfg.events, cfg.features, cfg.calibration.camera_id)
            if image_mode
            else EventDetector(
                cfg.events,
                cfg.features,
                calibration.camera_id,
                cfg.calibration.min_confidence,
                classifier,
            )
        )
        storage = EventStorage(output)
        clips = ClipBuffer(cfg.events, cfg.storage, output / "clips")
        source = VideoSource(
            cfg.video, on_frame=lambda packet: clips.append(packet.image, packet.timestamp_s)
        )
        metadata = {
            "run_id": run_id,
            "config": cfg.model_dump(mode="json"),
            "calibration": calibration.model_dump(mode="json") if calibration else None,
            "coordinate_mode": cfg.events.coordinate_mode,
            "calibration_provenance": calibration_input.provenance() if calibration_input else None,
            "hardware": hardware_info(),
            "model_sha256": hash_file(cfg.perception.model),
            "versions": {
                name: version(name) for name in ["av", "numpy", "opencv-python", "pydantic"]
            },
            "synthetic": cfg.perception.backend == "synthetic",
        }
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        processed = decoded = event_count = 0
        observation_count = frames_with_detections = segmented_instances = max_tracks = 0
        state_counts = {}
        next_inference = None
        last_motion_check = -float("inf")
        media_start = media_end = 0.0
        peak_rss = 0
        guard = None
        native_size = None
        invalidations = []
        failure = None
        stopped = False
        exhausted = False
        process = psutil.Process()
        cpu_start = process.cpu_times()

        def invalidate_calibration(reason, packet):
            if calibration.valid:
                invalidations.append(
                    {
                        "reason": reason,
                        "timestamp_s": packet.timestamp_s,
                        "frame_index": packet.frame_index,
                    }
                )
                calibration.valid = False
                trajectories.histories.clear()
                detector.reset()

        def save_completed(events):
            for event in events:
                storage.save(event)
            for event_id in clips.enforce_retention():
                row = storage.connection.execute(
                    "SELECT payload FROM events WHERE event_id = ?", (event_id,)
                ).fetchone()
                if row:
                    from .types import Event

                    data = json.loads(row[0])
                    data["clip_status"], data["clip_path"] = "expired", None
                    storage.save(Event(**data))

        try:
            with (
                (run_dir / "trajectories.jsonl").open("w", encoding="utf-8") as trajectory_log,
                (run_dir / "features.jsonl").open("w", encoding="utf-8") as feature_log,
            ):
                iterator = iter(source)
                while True:
                    if stop_requested and stop_requested():
                        stopped = True
                        break
                    with profiler.measure("decode"):
                        packet = next(iterator, None)
                    if packet is None:
                        exhausted = True
                        break
                    if decoded == 0:
                        media_start = packet.timestamp_s
                        actual_size = (packet.image.shape[1], packet.image.shape[0])
                        native_size = (
                            packet.original_image_size or source.original_image_size or actual_size
                        )
                        motion_threshold = cfg.calibration.motion_threshold_px
                        if image_mode:
                            calibration = image_reference(cfg.calibration.camera_id, actual_size)
                            calibration.save(run_dir / "image-reference.yaml")
                        else:
                            calibration, transform, reference, basis = calibration_input.adapt(
                                native_size, actual_size
                            )
                            motion_threshold = transform.adapt_motion_threshold(motion_threshold)
                            calibration_input.write_effective(run_dir, calibration, reference)
                            metadata["calibration_transform"] = {
                                **asdict(transform),
                                "matrix": transform.matrix.tolist(),
                                "inverse_matrix": np.linalg.inv(transform.matrix).tolist(),
                                "coordinate_basis": basis,
                                "motion_threshold_original_px": cfg.calibration.motion_threshold_px,
                                "motion_threshold_processed_px": motion_threshold,
                            }
                            metadata["decoder_transform"] = asdict(
                                ImageTransform.resize(native_size, actual_size)
                            )
                        metadata["calibration"] = calibration.model_dump(mode="json")
                        metadata["original_image_size"] = native_size
                        metadata["processed_image_size"] = actual_size
                        (run_dir / "run.json").write_text(
                            json.dumps(metadata, indent=2), encoding="utf-8"
                        )
                        if actual_size != calibration.image_size:
                            raise ValueError(
                                f"Video size {actual_size} differs from calibration {calibration.image_size}"
                            )
                        if cfg.calibration.detect_camera_motion:
                            guard = CameraMotionGuard(
                                reference if reference is not None else packet.image,
                                motion_threshold,
                            )
                    if (packet.image.shape[1], packet.image.shape[0]) != calibration.image_size:
                        invalidate_calibration("processed_resolution_changed", packet)
                    elif (
                        not image_mode
                        and packet.original_image_size
                        and packet.original_image_size != native_size
                    ):
                        invalidate_calibration("decoded_resolution_changed", packet)
                    if packet.discontinuity:
                        trajectories.histories.clear()
                        detector.reset()
                    decoded += 1
                    media_end = packet.timestamp_s
                    with profiler.measure("clip_buffer"):
                        if not source.live:
                            clips.append(packet.image, packet.timestamp_s)
                        save_completed(clips.collect())
                    if next_inference is None:
                        next_inference = packet.timestamp_s
                    if packet.timestamp_s + 1e-6 < next_inference:
                        continue
                    interval = 1 / cfg.video.target_fps
                    periods = max(
                        1, int((packet.timestamp_s - next_inference + 1e-6) / interval) + 1
                    )
                    next_inference += periods * interval
                    frame_started = time.perf_counter()
                    with profiler.measure("segmentation"):
                        instances = segmenter.predict(packet.image)
                        segmented_instances += len(instances)
                        frames_with_detections += bool(instances)
                    with profiler.measure("tracking"):
                        tracks = tracker.update(
                            instances, packet.timestamp_s, packet.image.shape[:2]
                        )
                    max_tracks = max(max_tracks, len(tracks))
                    with profiler.measure("geometry"):
                        if (
                            guard
                            and packet.timestamp_s - last_motion_check
                            >= cfg.calibration.motion_check_s
                        ):
                            exclusion = np.zeros(packet.image.shape[:2], bool)
                            for instance in instances:
                                exclusion |= instance.mask
                            if guard.update(packet.image, exclusion):
                                invalidate_calibration("camera_motion", packet)
                            last_motion_check = packet.timestamp_s
                        observations = []
                        if calibration.valid:
                            for track in tracks:
                                point, quality = ground_point(
                                    track.instance.mask, track.instance.bbox, calibration.roi_px
                                )
                                if cfg.features.ground_point_method == "box":
                                    box = track.instance.bbox
                                    point = (float((box[0] + box[2]) / 2), float(box[3]))
                                    quality = (
                                        1.0
                                        if cv2.pointPolygonTest(
                                            np.asarray(calibration.roi_px, np.float32), point, False
                                        )
                                        >= 0
                                        else 0.0
                                    )
                                if quality <= 0:
                                    continue
                                try:
                                    world = calibration.transform(point)
                                except ValueError:
                                    continue
                                obs = Observation(
                                    track.track_id,
                                    packet.timestamp_s,
                                    world,
                                    point,
                                    track.instance.bbox,
                                    track.instance.class_id,
                                    track.instance.confidence,
                                    quality,
                                )
                                observations.append(obs)
                                observation_count += 1
                                row = {
                                    "camera_id": calibration.camera_id,
                                    "clip_id": clip_id,
                                    "frame_index": packet.frame_index,
                                    "timestamp_s": packet.timestamp_s,
                                    "track_id": obs.track_id,
                                    "class_id": obs.class_id,
                                    "point_px": point,
                                    "point_world": None if image_mode else world.tolist(),
                                    "position": world.tolist(),
                                    "coordinate_units": calibration.units,
                                    "quality": quality,
                                    "confidence": obs.confidence,
                                    "bbox": obs.bbox.tolist(),
                                    "predicted": False,
                                }
                                trajectory_log.write(json.dumps(row, allow_nan=False) + "\n")
                    with profiler.measure("features_events"):
                        motions = trajectories.update(observations, packet.timestamp_s)
                        pairs = compute_pairs(motions, cfg.features, cfg.events.coordinate_mode)
                        decision = detector.update(motions, pairs, packet.timestamp_s, calibration)
                        state_counts[decision.state] = state_counts.get(decision.state, 0) + 1
                        motion_rows = {
                            key: {
                                "speed": m.speed,
                                "deceleration": m.deceleration,
                                "jerk": m.jerk,
                                "heading_change": m.heading_change,
                                "quality": m.quality,
                                "age_s": m.age_s,
                                "stop_duration_s": m.stop_duration_s,
                                "prior_speed": m.prior_speed,
                            }
                            for key, m in motions.items()
                        }
                        feature_log.write(
                            json.dumps(
                                {
                                    "camera_id": calibration.camera_id,
                                    "clip_id": clip_id,
                                    "frame_index": packet.frame_index,
                                    "timestamp_s": packet.timestamp_s,
                                    "coordinate_mode": cfg.events.coordinate_mode,
                                    "coordinate_units": calibration.units,
                                    "calibration_valid": calibration.valid,
                                    "motions": motion_rows,
                                    "pairs": [asdict(pair) for pair in pairs],
                                    "state": decision.state,
                                    "score": decision.score,
                                    "reasons": decision.reasons,
                                },
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        for event in decision.events:
                            event.run_id = run_id
                            event.clip_id = clip_id
                            clips.schedule(event)
                            storage.save(event)
                            event_count += 1
                    with profiler.measure("rendering"):
                        if callback:
                            rendered = render_frame(
                                packet.image,
                                tracks,
                                motions,
                                trajectories.histories,
                                decision,
                                calibration,
                                cfg.ui.render_masks,
                            )
                            bird = (
                                render_bird_eye(
                                    motions,
                                    trajectories.histories,
                                    calibration,
                                    decision.candidate_ids,
                                )
                                if cfg.ui.render_bird_eye and not image_mode
                                else None
                            )
                            callback(
                                {
                                    "frame": rendered,
                                    "bird_eye": bird,
                                    "decision": decision,
                                    "timestamp_s": packet.timestamp_s,
                                    "processed_frames": processed + 1,
                                    "fps": (processed + 1)
                                    / max(1e-9, time.perf_counter() - started),
                                    "run_id": run_id,
                                    "detections": len(instances),
                                    "tracks": len(tracks),
                                    "coordinate_mode": cfg.events.coordinate_mode,
                                    "calibration_valid": calibration.valid,
                                    "calibration_invalidations": list(invalidations),
                                }
                            )
                    processed += 1
                    if progress_callback:
                        progress_callback(
                            {
                                "timestamp_s": packet.timestamp_s,
                                "duration_s": source.duration_s,
                                "processed_frames": processed,
                                "run_id": run_id,
                                "calibration_valid": calibration.valid,
                                "calibration_invalidations": list(invalidations),
                            }
                        )
                    duration = time.perf_counter() - frame_started
                    profiler.samples["end_to_end"].append(duration)
                    profiler.totals["end_to_end"] += duration
                    profiler.counts["end_to_end"] += 1
                    peak_rss = max(peak_rss, process.memory_info().rss)
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            source.close()
            save_completed(clips.close())
            storage.close()
            elapsed = time.perf_counter() - started
            cpu_end = process.cpu_times()
            completed = exhausted and not failure and decoded > 0
            evaluable = (
                completed
                and calibration is not None
                and calibration.valid
                and (image_mode or calibration.metric_valid(cfg.calibration.min_confidence))
            )
            non_evaluable_reasons = [item["reason"] for item in invalidations]
            if not completed:
                non_evaluable_reasons.append(
                    "analysis_error"
                    if failure
                    else "analysis_stopped"
                    if stopped
                    else "no_decodable_frames"
                )
            if not image_mode and not evaluable and not non_evaluable_reasons:
                non_evaluable_reasons.append("calibration_not_metric_or_invalid")
            metadata["calibration"] = calibration.model_dump(mode="json") if calibration else None
            metadata["calibration_invalidations"] = invalidations
            metadata["evaluable"] = evaluable
            metadata["evaluation_status"] = "evaluable" if evaluable else "not_evaluable"
            metadata["non_evaluable_reasons"] = non_evaluable_reasons
            (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if calibration_input and calibration and "calibration_transform" in metadata:
                calibration_input.write_effective(run_dir, calibration)
            self.summary = {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "processed_frames": processed,
                "decoded_frames": decoded,
                "events": event_count,
                "elapsed_s": elapsed,
                "media_duration_s": max(0, media_end - media_start),
                "effective_fps": processed / elapsed if elapsed else 0,
                "peak_rss_mb": peak_rss / 1024**2,
                "dropped_frames": source.dropped_frames,
                "cpu_seconds": cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system,
                "timestamp_fallbacks": source.timestamp_fallbacks,
                "calibration_valid": calibration.valid if calibration else False,
                "metric_calibration_available": calibration.metric_valid(
                    cfg.calibration.min_confidence
                )
                if calibration
                else False,
                "coordinate_mode": cfg.events.coordinate_mode,
                "source_duration_s": source.duration_s,
                "segmented_instances": segmented_instances,
                "frames_with_detections": frames_with_detections,
                "track_observations": observation_count,
                "max_simultaneous_tracks": max_tracks,
                "state_counts": state_counts,
                "error": failure,
                "stopped": stopped,
                "completed": completed,
                "evaluable": evaluable,
                "evaluation_status": "evaluable" if evaluable else "not_evaluable",
                "non_evaluable_reasons": non_evaluable_reasons,
                "calibration_invalidations": invalidations,
                "timings": profiler.summary(),
                "synthetic": metadata["synthetic"],
            }
            (run_dir / "metrics.json").write_text(
                json.dumps(self.summary, indent=2), encoding="utf-8"
            )
        return self.summary
