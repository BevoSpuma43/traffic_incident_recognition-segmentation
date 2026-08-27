"""Esecuzione batch e valutazione sul dataset reale annotato."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2

from src.calibration import (
    Annotation,
    evaluate_by_group,
    evaluate_records,
    load_annotations,
    load_jsonl_records,
    summarize_annotations,
    summarize_diagnostics,
)
from src.config import load_default_config
from src.video_pipeline import TrafficAccidentPipeline


def select_annotations(
    annotations: list[Annotation],
    *,
    split_field: str | None = None,
    split: str | None = None,
    accident_types: set[str] | None = None,
    limit: int | None = None,
) -> list[Annotation]:
    selected = annotations
    if split_field and split:
        selected = [
            item
            for item in selected
            if str(getattr(item, split_field)).casefold() == split.casefold()
        ]
    if accident_types:
        normalized_types = {value.casefold() for value in accident_types}
        selected = [
            item for item in selected if item.label.casefold() in normalized_types
        ]
    return selected[: max(0, limit)] if limit is not None else selected


def run_video(
    annotation: Annotation,
    *,
    dataset_root: Path,
    model_path: str,
    log_path: Path,
    diagnostics: bool,
) -> dict[str, object]:
    video_path = _resolve_dataset_path(dataset_root, annotation.video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video non trovato: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    config = replace(
        load_default_config(),
        video_path=str(video_path),
        model_path=model_path,
        calibration_log_path=str(log_path),
        calibration_log_diagnostics=diagnostics,
        draw_masks=False,
        draw_ids=False,
        draw_bbox=False,
    )
    pipeline: TrafficAccidentPipeline | None = None
    processed_frames = 0
    event_count = 0
    decoder_fps = float(capture.get(cv2.CAP_PROP_FPS))
    run_started_at = time.perf_counter()
    processing_started_at: float | None = None
    try:
        pipeline = TrafficAccidentPipeline(config)
        processing_started_at = time.perf_counter()
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if processed_frames > 0 and timestamp_s <= 0.0 and decoder_fps > 0.0:
                timestamp_s = processed_frames / decoder_fps
            _, events = pipeline.process_frame(
                frame,
                processed_frames,
                render=False,
                timestamp_s=timestamp_s,
            )
            event_count += len(events)
            processed_frames += 1
    finally:
        if pipeline is not None:
            pipeline.close()
        capture.release()

    finished_at = time.perf_counter()
    elapsed_seconds = finished_at - run_started_at
    processing_seconds = (
        finished_at - processing_started_at
        if processing_started_at is not None
        else 0.0
    )
    result: dict[str, object] = {
        "kind": "run",
        "completed": True,
        "video_path": str(video_path),
        "label": annotation.label,
        "accident_frame": annotation.start_frame,
        "processed_frames": processed_frames,
        "event_count": event_count,
        "decoder_fps": decoder_fps,
        "elapsed_seconds": elapsed_seconds,
        "startup_seconds": max(0.0, elapsed_seconds - processing_seconds),
        "processing_seconds": processing_seconds,
        "processing_fps": (
            processed_frames / processing_seconds if processing_seconds > 0 else 0.0
        ),
    }
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def _resolve_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    root = dataset_root.resolve()
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path fuori dal dataset: {relative_path}") from exc
    return candidate


def _completed_log(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as stream:
            return any(
                record.get("kind") == "run" and record.get("completed") is True
                for record in (json.loads(line) for line in stream if line.strip())
            )
    except (OSError, json.JSONDecodeError):
        return False


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sul dataset reale")
    parser.add_argument("--metadata", default="dataset/metadata-real.csv")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--output-dir", default="calibration/real")
    parser.add_argument(
        "--split-field",
        choices=("split_in_distribution", "split_geo_aware"),
    )
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--type", action="append", dest="accident_types")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Salva anche ogni coppia/frame (output molto voluminoso)",
    )
    parser.add_argument("--tolerance-frames", type=int, default=0)
    parser.add_argument("--tolerance-seconds", type=float, default=1.0)
    args = parser.parse_args()

    annotations = select_annotations(
        load_annotations(args.metadata),
        split_field=args.split_field,
        split=args.split,
        accident_types=set(args.accident_types or []),
        limit=args.limit,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for index, annotation in enumerate(annotations, start=1):
        log_path = output_dir / f"{Path(annotation.video_path).stem}.jsonl"
        if args.resume and _completed_log(log_path):
            print(f"[{index}/{len(annotations)}] skip {annotation.video_path}")
            continue
        print(f"[{index}/{len(annotations)}] {annotation.video_path}")
        try:
            results.append(
                run_video(
                    annotation,
                    dataset_root=dataset_root,
                    model_path=args.model,
                    log_path=log_path,
                    diagnostics=args.diagnostics,
                )
            )
        except Exception as exc:
            failures.append(
                {"video_path": annotation.video_path, "error": str(exc)}
            )
            print(f"  ERRORE: {exc}")

    completed_logs = [
        output_dir / f"{Path(item.video_path).stem}.jsonl"
        for item in annotations
        if _completed_log(output_dir / f"{Path(item.video_path).stem}.jsonl")
    ]
    records = load_jsonl_records(completed_logs)
    run_records = [record for record in records if record.get("kind") == "run"]
    total_frames = sum(int(item.get("processed_frames", 0)) for item in run_records)
    total_elapsed = sum(float(item.get("elapsed_seconds", 0.0)) for item in run_records)
    total_processing = sum(
        float(item.get("processing_seconds", item.get("elapsed_seconds", 0.0)))
        for item in run_records
    )
    completed_video_keys = {
        Path(str(item.get("video_path", ""))).name.casefold() for item in run_records
    }
    total_video_seconds = sum(
        float(item.duration_s or 0.0)
        for item in annotations
        if Path(item.video_path).name.casefold() in completed_video_keys
    )
    grouped_evaluation = {
        field: evaluate_by_group(
            annotations,
            records,
            field,
            tolerance_frames=args.tolerance_frames,
            tolerance_seconds=args.tolerance_seconds,
        )
        for field in (
            "label",
            "rollover",
            "scene_layout",
            "weather",
            "day_time",
            "quality",
        )
    }
    report = {
        "selection": {
            "metadata": args.metadata,
            "dataset_root": args.dataset_root,
            "model": args.model,
            "split_field": args.split_field,
            "split": args.split,
            "types": args.accident_types or [],
            "selected_videos": len(annotations),
            "completed_videos": len(completed_logs),
        },
        "ground_truth": summarize_annotations(annotations),
        "diagnostics": summarize_diagnostics(records),
        "performance": {
            "processed_frames": total_frames,
            "elapsed_seconds": total_elapsed,
            "processing_seconds": total_processing,
            "end_to_end_fps": total_frames / total_elapsed if total_elapsed else 0.0,
            "processing_fps": (
                total_frames / total_processing if total_processing else 0.0
            ),
            "video_duration_seconds": total_video_seconds,
            "end_to_end_realtime_factor": (
                total_video_seconds / total_elapsed if total_elapsed else 0.0
            ),
            "processing_realtime_factor": (
                total_video_seconds / total_processing if total_processing else 0.0
            ),
        },
        "evaluation": evaluate_records(
            annotations,
            records,
            tolerance_frames=args.tolerance_frames,
            tolerance_seconds=args.tolerance_seconds,
        ),
        "by_group": grouped_evaluation,
        "failures": failures,
        "runs": run_records,
    }
    _write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
