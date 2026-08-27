"""Raccolta diagnostica e valutazione riproducibile delle collisioni."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TextIO

from src.config import AppConfig
from src.models import CollisionEvent, PairDiagnostic


class CalibrationRecorder:
    """Scrive evidenze ed eventi in formato JSON Lines."""

    def __init__(self, path: str, video_path: str, config: AppConfig) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO = output_path.open("w", encoding="utf-8")
        self._video_path = _normalize_video_path(video_path)
        self._record_diagnostics = bool(
            getattr(config, "calibration_log_diagnostics", True)
        )
        self._write(
            {
                "kind": "metadata",
                "schema_version": 1,
                "video_path": self._video_path,
                "config": asdict(config),
            }
        )
        self._stream.flush()

    def record_frame(
        self,
        frame_index: int,
        diagnostics: Iterable[PairDiagnostic],
        events: Iterable[CollisionEvent],
        timestamp_s: float | None = None,
        detection_count: int | None = None,
        active_track_count: int | None = None,
    ) -> None:
        timestamp_data = (
            {"timestamp_s": float(timestamp_s)} if timestamp_s is not None else {}
        )
        wrote_record = False
        if self._record_diagnostics:
            diagnostic_items = list(diagnostics)
            event_items = list(events)
            self._write(
                {
                    "kind": "frame",
                    "video_path": self._video_path,
                    **timestamp_data,
                    "frame_index": frame_index,
                    "detection_count": detection_count,
                    "active_track_count": active_track_count,
                    "pair_count": len(diagnostic_items),
                    "event_count": len(event_items),
                }
            )
            wrote_record = True
            for diagnostic in diagnostic_items:
                self._write(
                    {
                        "kind": "diagnostic",
                        "video_path": self._video_path,
                        **timestamp_data,
                        **asdict(diagnostic),
                    }
                )
                wrote_record = True
        else:
            event_items = list(events)
        for event in event_items:
            self._write(
                {
                    "kind": "event",
                    "video_path": self._video_path,
                    **timestamp_data,
                    **asdict(event),
                }
            )
            wrote_record = True
        if wrote_record:
            self._stream.flush()

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def _write(self, record: dict[str, object]) -> None:
        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass(frozen=True, slots=True)
class Annotation:
    video_path: str
    start_frame: int
    end_frame: int
    label: str = "collision"
    accident_time_s: float | None = None
    fps: float | None = None
    rollover: bool = False
    region: str = ""
    scene_layout: str = ""
    weather: str = ""
    day_time: str = ""
    quality: str = ""
    split_in_distribution: str = ""
    split_geo_aware: str = ""
    no_frames: int | None = None
    duration_s: float | None = None


def load_annotations(path: str | Path) -> list[Annotation]:
    """Carica il CSV ``metadata-real`` oppure lo schema JSON legacy."""
    annotation_path = Path(path)
    if annotation_path.suffix.casefold() == ".csv":
        return load_real_metadata_csv(annotation_path)

    with annotation_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    annotations: list[Annotation] = []
    for video in payload.get("videos", []):
        video_path = _normalize_video_path(str(video["video_path"]))
        for event in video.get("events", []):
            start_frame = int(event["start_frame"])
            end_frame = int(event.get("end_frame", start_frame))
            if start_frame < 0 or end_frame < start_frame:
                raise ValueError(
                    f"Intervallo non valido per {video_path}: "
                    f"{start_frame}-{end_frame}"
                )
            annotations.append(
                Annotation(
                    video_path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    label=str(event.get("label", "collision")),
                )
            )
    return annotations


def load_real_metadata_csv(path: str | Path) -> list[Annotation]:
    """Converte una riga di ``metadata-real.csv`` in una ground truth."""
    metadata_path = Path(path)
    annotations: list[Annotation] = []
    seen_videos: set[str] = set()
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "type", "accident_time", "accident_frame"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Colonne mancanti in metadata-real: " + ", ".join(sorted(missing))
            )

        for line_number, row in enumerate(reader, start=2):
            video_path = _normalize_video_path(str(row["path"]))
            video_key = _video_key(video_path)
            if video_key in seen_videos:
                raise ValueError(
                    f"Video duplicato in {metadata_path}:{line_number}: {video_path}"
                )
            seen_videos.add(video_key)

            accident_frame = _required_int(row, "accident_frame", line_number)
            if accident_frame < 0:
                raise ValueError(
                    f"accident_frame negativo in {metadata_path}:{line_number}"
                )
            no_frames = _optional_int(row.get("no_frames"))
            duration_s = _optional_float(row.get("duration"))
            fps = (
                no_frames / duration_s
                if no_frames is not None and duration_s is not None and duration_s > 0
                else None
            )
            annotations.append(
                Annotation(
                    video_path=video_path,
                    start_frame=accident_frame,
                    end_frame=accident_frame,
                    label=str(row.get("type") or "collision").strip(),
                    accident_time_s=_optional_float(row.get("accident_time")),
                    fps=fps,
                    rollover=str(row.get("rollover", "0")).strip() == "1",
                    region=str(row.get("region", "")).strip(),
                    scene_layout=str(row.get("scene_layout", "")).strip(),
                    weather=str(row.get("weather", "")).strip(),
                    day_time=str(row.get("day_time", "")).strip(),
                    quality=str(row.get("quality", "")).strip(),
                    split_in_distribution=str(
                        row.get("split_in_distribution", "")
                    ).strip(),
                    split_geo_aware=str(row.get("split_geo_aware", "")).strip(),
                    no_frames=no_frames,
                    duration_s=duration_s,
                )
            )
    return annotations


def load_jsonl_records(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL non valido in {path}:{line_number}") from exc
    return records


def evaluate_records(
    annotations: Iterable[Annotation],
    records: Iterable[dict[str, object]],
    tolerance_frames: int = 0,
    tolerance_seconds: float = 0.0,
) -> dict[str, float | int]:
    """Abbina uno-a-uno eventi previsti e intervalli annotati."""
    all_records = list(records)
    truth = list(annotations)
    recorded_videos = {
        _video_key(str(record.get("video_path", "")))
        for record in all_records
        if record.get("kind") == "metadata" and record.get("video_path")
    }
    if recorded_videos:
        truth = [item for item in truth if _video_key(item.video_path) in recorded_videos]
    predictions = [record for record in all_records if record.get("kind") == "event"]
    matched_truth: set[int] = set()
    delays: list[int] = []
    delay_seconds: list[float] = []
    true_positives = 0

    predictions.sort(
        key=lambda item: (
            _video_key(str(item.get("video_path", ""))),
            _prediction_frame(item),
        )
    )
    tolerance = max(0, int(tolerance_frames))

    for prediction in predictions:
        video_path = _video_key(str(prediction.get("video_path", "")))
        frame_index = _prediction_frame(prediction)
        prediction_time_s = _record_float(prediction.get("timestamp_s"))
        candidates = [
            (index, annotation)
            for index, annotation in enumerate(truth)
            if index not in matched_truth
            and _video_key(annotation.video_path) == video_path
            and _prediction_matches(
                annotation,
                frame_index,
                prediction_time_s,
                tolerance,
                tolerance_seconds,
            )
        ]
        if not candidates:
            continue
        match_index, match = min(
            candidates,
            key=lambda item: abs(frame_index - item[1].start_frame),
        )
        matched_truth.add(match_index)
        true_positives += 1
        delay = frame_index - match.start_frame
        delays.append(delay)
        if prediction_time_s is not None and match.accident_time_s is not None:
            delay_seconds.append(prediction_time_s - match.accident_time_s)
        elif match.fps is not None and match.fps > 0:
            delay_seconds.append(delay / match.fps)

    false_positives = len(predictions) - true_positives
    false_negatives = len(truth) - true_positives
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "evaluated_videos": len({_video_key(item.video_path) for item in truth}),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_delay_frames": statistics.fmean(delays) if delays else 0.0,
        "median_delay_frames": statistics.median(delays) if delays else 0.0,
        "mean_delay_seconds": statistics.fmean(delay_seconds) if delay_seconds else 0.0,
        "median_delay_seconds": statistics.median(delay_seconds) if delay_seconds else 0.0,
    }


def summarize_diagnostics(
    records: Iterable[dict[str, object]],
) -> dict[str, object]:
    diagnostics = [
        record for record in records if record.get("kind") == "diagnostic"
    ]
    frames = [record for record in records if record.get("kind") == "frame"]
    contacts = [record for record in diagnostics if bool(record.get("contact"))]
    emitted = [record for record in diagnostics if bool(record.get("emitted"))]
    return {
        "frame_rows": len(frames),
        "frames_with_detections": sum(
            int(record.get("detection_count") or 0) > 0 for record in frames
        ),
        "frames_with_multiple_tracks": sum(
            int(record.get("active_track_count") or 0) >= 2 for record in frames
        ),
        "max_active_tracks": max(
            (int(record.get("active_track_count") or 0) for record in frames),
            default=0,
        ),
        "diagnostic_rows": len(diagnostics),
        "contact_rows": len(contacts),
        "emitted_rows": len(emitted),
        "contact_overlap_ratio": _distribution(
            float(record.get("overlap_ratio", 0.0)) for record in contacts
        ),
        "contact_closing_speed_px": _distribution(
            float(record.get("closing_speed_px", 0.0)) for record in contacts
        ),
        "contact_spatial_distance_px": _distribution(
            float(record.get("spatial_distance_px", 0.0)) for record in contacts
        ),
    }


def summarize_annotations(annotations: Iterable[Annotation]) -> dict[str, object]:
    """Riassume qualità temporale e composizione della ground truth."""
    items = list(annotations)
    fps_values = [item.fps for item in items if item.fps is not None]
    timestamp_errors = [
        abs((item.start_frame / item.fps) - item.accident_time_s)
        for item in items
        if item.fps is not None
        and item.fps > 0
        and item.accident_time_s is not None
    ]
    return {
        "annotated_videos": len(items),
        "accident_types": dict(sorted(_counts(item.label for item in items).items())),
        "metadata_average_fps": _distribution(float(value) for value in fps_values),
        "frame_timestamp_error_seconds": _distribution(timestamp_errors),
        "out_of_range_accident_frames": sum(
            1
            for item in items
            if item.no_frames is not None
            and (item.start_frame < 0 or item.start_frame >= item.no_frames)
        ),
    }


def _distribution(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    p90_index = min(len(ordered) - 1, math.floor(0.9 * (len(ordered) - 1)))
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[p90_index],
        "max": ordered[-1],
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _normalize_video_path(value: str) -> str:
    return value.replace("\\", "/")


def _video_key(value: str) -> str:
    """Abbina log e CSV anche se usano root relativa/assoluta diversa."""
    return Path(_normalize_video_path(value)).name.casefold()


def _required_int(row: dict[str, str], field: str, line_number: int) -> int:
    value = _optional_int(row.get(field))
    if value is None:
        raise ValueError(f"Valore {field!r} non valido alla riga {line_number}")
    return value


def _optional_int(value: str | None) -> int | None:
    try:
        return int(float(str(value))) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: str | None) -> float | None:
    try:
        return float(str(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _record_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _prediction_frame(record: dict[str, object]) -> int:
    value = record.get("confirmation_frame")
    if value is None:
        value = record.get("frame_index", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _annotation_tolerance(
    annotation: Annotation,
    tolerance_frames: int,
    tolerance_seconds: float,
) -> int:
    seconds_as_frames = 0
    if annotation.fps is not None and annotation.fps > 0:
        seconds_as_frames = round(max(0.0, tolerance_seconds) * annotation.fps)
    return max(tolerance_frames, seconds_as_frames)


def _prediction_matches(
    annotation: Annotation,
    frame_index: int,
    timestamp_s: float | None,
    tolerance_frames: int,
    tolerance_seconds: float,
) -> bool:
    frame_tolerance = _annotation_tolerance(
        annotation,
        tolerance_frames,
        tolerance_seconds if timestamp_s is None else 0.0,
    )
    frame_match = (
        annotation.start_frame - frame_tolerance
        <= frame_index
        <= annotation.end_frame + frame_tolerance
    )
    if timestamp_s is None or annotation.accident_time_s is None:
        return frame_match
    time_match = abs(timestamp_s - annotation.accident_time_s) <= max(
        0.0, tolerance_seconds
    )
    return time_match or (tolerance_frames > 0 and frame_match)


def evaluate_by_group(
    annotations: Iterable[Annotation],
    records: Iterable[dict[str, object]],
    field: str,
    *,
    tolerance_frames: int = 0,
    tolerance_seconds: float = 0.0,
) -> dict[str, dict[str, float | int]]:
    """Calcola le stesse metriche per tipologia o attributo ambientale."""
    truth = list(annotations)
    all_records = list(records)
    groups = sorted({str(getattr(item, field)) for item in truth})
    grouped_metrics: dict[str, dict[str, float | int]] = {}
    for group in groups:
        if not group:
            continue
        group_truth = [
            item for item in truth if str(getattr(item, field)) == group
        ]
        group_video_keys = {_video_key(item.video_path) for item in group_truth}
        group_records = [
            record
            for record in all_records
            if _video_key(str(record.get("video_path", ""))) in group_video_keys
        ]
        grouped_metrics[group] = evaluate_records(
            group_truth,
            group_records,
            tolerance_frames=tolerance_frames,
            tolerance_seconds=tolerance_seconds,
        )
    return grouped_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Valuta i log di calibrazione")
    parser.add_argument("--log", action="append", required=True, help="File JSONL")
    parser.add_argument("--annotations", help="metadata-real.csv o ground truth JSON")
    parser.add_argument("--tolerance-frames", type=int, default=0)
    parser.add_argument("--tolerance-seconds", type=float, default=1.0)
    args = parser.parse_args()

    records = load_jsonl_records(args.log)
    output: dict[str, object] = {"summary": summarize_diagnostics(records)}
    if args.annotations:
        annotations = load_annotations(args.annotations)
        output["ground_truth"] = summarize_annotations(annotations)
        output["evaluation"] = evaluate_records(
            annotations,
            records,
            tolerance_frames=args.tolerance_frames,
            tolerance_seconds=args.tolerance_seconds,
        )
        output["by_type"] = evaluate_by_group(
            annotations,
            records,
            "label",
            tolerance_frames=args.tolerance_frames,
            tolerance_seconds=args.tolerance_seconds,
        )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
