"""Create seekable annotated replays from saved analysis, without running inference again."""

import json
from collections import deque
from itertools import groupby
from pathlib import Path

import av
import cv2
import numpy as np

from .calibration.repository import file_sha256
from .config import AppConfig
from .preview import PreviewWriter
from .storage import EventStorage
from .video import VideoSource
from .visualization import color_for


def _json_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _snapshots(run_dir):
    """Join the two ordered logs while retaining just one inference snapshot."""
    observations = iter(_json_rows(run_dir / "trajectories.jsonl"))
    row = next(observations, None)
    for feature in _json_rows(run_dir / "features.jsonl"):
        timestamp = feature["timestamp_s"]
        rows = []
        while row is not None and row["timestamp_s"] <= timestamp + 1e-6:
            if abs(row["timestamp_s"] - timestamp) < 1e-6:
                rows.append(row)
            row = next(observations, None)
        yield timestamp, rows, feature


def locate_impacts(trajectory_path, events, max_gap_s=0.5):
    """Estimate a fixed image point from the involved boxes near the impact time."""
    markers = [
        {
            "event_id": event["event_id"],
            "impact_time_s": event["impact_time_s"],
            "confirm_time_s": event["confirm_time_s"],
            "track_ids": event["track_ids"],
            "point_px": None,
            "sample_time_s": None,
        }
        for event in sorted(events, key=lambda event: event["impact_time_s"])
    ]
    best = {}
    for timestamp, group in groupby(
        _json_rows(trajectory_path), key=lambda row: row["timestamp_s"]
    ):
        rows = {row["track_id"]: row for row in group}
        for marker in markers:
            distance = abs(timestamp - marker["impact_time_s"])
            # Prefer the earlier observation when two samples are equally close.
            rank = (round(distance, 8), timestamp > marker["impact_time_s"])
            if distance > max_gap_s or rank >= best.get(marker["event_id"], (float("inf"), True)):
                continue
            if not marker["track_ids"] or any(key not in rows for key in marker["track_ids"]):
                continue
            boxes = np.asarray([rows[key]["bbox"] for key in marker["track_ids"]], dtype=float)
            if not np.isfinite(boxes).all():
                continue
            # Overlap centre, or midpoint of the gap between the nearest box edges.
            point = (boxes[:, :2].max(axis=0) + boxes[:, 2:].min(axis=0)) / 2
            marker["point_px"] = point.tolist()
            marker["sample_time_s"] = timestamp
            best[marker["event_id"]] = rank
    return markers


def draw_impact_markers(image, markers, timestamp_s):
    """Draw confirmed detections retrospectively from their estimated impact time."""
    height, width = image.shape[:2]
    for index, marker in enumerate(markers, 1):
        point = marker.get("point_px")
        if point is None or timestamp_s + 1e-6 < marker["impact_time_s"]:
            continue
        if not np.isfinite(point).all():
            continue
        x, y = np.rint(point).astype(int)
        if not (0 <= x < width and 0 <= y < height):
            continue
        red = (30, 35, 245)
        cv2.circle(image, (x, y), 23, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.circle(image, (x, y), 23, red, 3, cv2.LINE_AA)
        cv2.drawMarker(image, (x, y), red, cv2.MARKER_CROSS, 32, 3, cv2.LINE_AA)
        label = f"IMPATTO {index} | {marker['impact_time_s']:.2f}s"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = max(4, min(x + 30, width - text_width - 8))
        ty = max(text_height + 8, min(y - 30, height - 10))
        cv2.rectangle(image, (tx - 4, ty - text_height - 5), (tx + text_width + 4, ty + 5), red, -1)
        cv2.putText(
            image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return image


def _render(image, rows, histories, feature, calibration, markers, timestamp):
    canvas = image.copy()
    units = feature.get("coordinate_units", calibration.get("units", ""))
    for row in rows:
        key = row["track_id"]
        color = color_for(key)
        x1, y1, x2, y2 = map(round, row["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"ID {key}"
        motion = feature.get("motions", {}).get(str(key))
        if motion is not None:
            if units == "m":
                label += f" | {motion['speed'] * 3.6:.1f} km/h"
            elif units == "px":
                label += f" | {motion['speed']:.0f} px/s"
        cv2.putText(canvas, label, (x1, max(55, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        points = np.asarray([point for _, point in histories.get(key, ())], np.int32)
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(map(round, row["point_px"])), 4, color, -1)
    if calibration.get("roi_px"):
        cv2.polylines(
            canvas, [np.asarray(calibration["roi_px"], np.int32)], True, (255, 220, 100), 1
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (25, 25, 25), -1)
    label = (
        f"{timestamp:.2f}s | {feature.get('state', 'NORMAL')} | score {feature.get('score', 0):.2f}"
    )
    if feature.get("run_evaluable") is False:
        label = "NON VALUTABILE | " + label
    cv2.putText(canvas, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    return draw_impact_markers(canvas, markers, timestamp)


def list_saved_runs(output_dir):
    runs = []
    for metrics_path in (Path(output_dir) / "runs").glob("*/metrics.json"):
        try:
            summary = json.loads(metrics_path.read_text(encoding="utf-8"))
            metadata = json.loads((metrics_path.parent / "run.json").read_text(encoding="utf-8"))
            if not summary.get("processed_frames") or metadata["config"]["video"][
                "source"
            ].lower().startswith(("rtsp://", "rtsps://")):
                continue
            runs.append(
                {
                    "run_id": summary["run_id"],
                    "run_dir": str(metrics_path.parent),
                    "source_name": Path(metadata["config"]["video"]["source"]).name,
                    "events": summary["events"],
                    "evaluable": summary.get("evaluable", summary.get("completed", True)),
                    "modified_at": metrics_path.stat().st_mtime,
                }
            )
        except (OSError, ValueError, KeyError):
            continue
    return sorted(runs, key=lambda row: row["modified_at"], reverse=True)


def export_replay(run_dir):
    """Render every original frame using saved boxes, trails, speeds and impact markers."""
    run_dir = Path(run_dir).resolve()
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = AppConfig.model_validate(metadata["config"])
    if cfg.video.source.lower().startswith(("rtsp://", "rtsps://")):
        raise ValueError("La riproduzione richiede un file video locale.")
    if not metrics.get("processed_frames"):
        raise ValueError("L'analisi non contiene frame elaborati.")
    source_path = Path(cfg.video.source)
    if not source_path.is_absolute():
        source_path = cfg.project.root_dir / source_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Video originale non disponibile: {source_path}")
    provenance = metadata.get("calibration_provenance") or {}
    video_identity = provenance.get("video") or {}

    def verify_source():
        if video_identity.get("sha256") and file_sha256(source_path) != video_identity["sha256"]:
            raise ValueError("Video originale cambiato rispetto alla calibrazione dell'analisi.")

    verify_source()
    evaluable = metrics.get("evaluable", metrics.get("completed", True))
    expected_size = metadata.get("processed_image_size")
    with av.open(str(source_path)) as container:
        stream = container.streams.video[0]
        fps = stream.average_rate or cfg.video.target_fps
    cfg.video.source = str(source_path)
    # An interrupted run must not present the unanalysed remainder as analysed.
    cfg.video.max_frames = metrics["decoded_frames"]
    storage = EventStorage(run_dir.parent.parent)
    try:
        events = storage.list_events(metadata["run_id"], limit=1_000_000)
    finally:
        storage.close()
    max_gap = max(cfg.features.max_gap_s, 2 / cfg.video.target_fps)
    markers = locate_impacts(run_dir / "trajectories.jsonl", events, max_gap)
    source = VideoSource(cfg.video)
    temporary = run_dir / "annotated.partial.mp4"
    target = run_dir / "annotated.mp4"
    writer = PreviewWriter(temporary, fps)
    snapshots = iter(_snapshots(run_dir))
    upcoming = next(snapshots, None)
    histories = {}
    rows, feature, snapshot_time = [], {}, -float("inf")
    frames, last_timestamp = 0, 0.0
    try:
        for packet in source:
            timestamp = packet.timestamp_s
            if expected_size and list(packet.image.shape[1::-1]) != expected_size:
                raise ValueError("La risoluzione del video non corrisponde all'analisi salvata.")
            while upcoming is not None and upcoming[0] <= timestamp + 1e-6:
                snapshot_time, rows, feature = upcoming
                for row in rows:
                    history = histories.setdefault(
                        row["track_id"], deque(maxlen=cfg.tracking.max_samples)
                    )
                    if history and snapshot_time - history[-1][0] > cfg.features.max_gap_s:
                        history.clear()
                    history.append((snapshot_time, row["point_px"]))
                upcoming = next(snapshots, None)
            for key in list(histories):
                history = histories[key]
                while history and timestamp - history[0][0] > cfg.tracking.history_seconds:
                    history.popleft()
                if not history:
                    del histories[key]
            visible = rows if timestamp - snapshot_time <= max_gap else []
            image = _render(
                packet.image,
                visible,
                histories,
                {**feature, "run_evaluable": evaluable},
                metadata.get("calibration") or {},
                markers,
                timestamp,
            )
            writer.append(image, timestamp)
            frames += 1
            last_timestamp = timestamp
        writer.close()
        if frames != metrics["decoded_frames"]:
            raise ValueError("Il video originale non contiene tutti i frame dell'analisi.")
        verify_source()
        temporary.replace(target)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        source.close()
        close = getattr(snapshots, "close", None)
        if close:
            close()
    result = {
        "run_id": metadata["run_id"],
        "video_path": str(target),
        "source_name": source_path.name,
        "frames": frames,
        "last_timestamp_s": last_timestamp,
        "impacts": markers,
        "evaluable": evaluable,
        "non_evaluable_reasons": metrics.get("non_evaluable_reasons", []),
        "calibration_invalidations": metrics.get("calibration_invalidations", []),
        "position_method": "Estimated from involved vehicle bounding boxes at impact time",
        "overlays": [
            "bounding_boxes",
            "track_ids",
            "trajectories",
            "speeds",
            "detector_state",
            "impacts",
        ],
    }
    report = run_dir / "replay.json"
    report_tmp = report.with_suffix(".tmp")
    report_tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_tmp.replace(report)
    return result
