from src.calibration import (
    Annotation,
    evaluate_by_group,
    evaluate_records,
    load_annotations,
    summarize_annotations,
    summarize_diagnostics,
)


def test_evaluation_matches_each_annotation_at_most_once() -> None:
    annotations = [Annotation("dataset/video.mp4", 10, 15)]
    records = [
        {
            "kind": "event",
            "video_path": "dataset/video.mp4",
            "confirmation_frame": 12,
        },
        {
            "kind": "event",
            "video_path": "dataset/video.mp4",
            "confirmation_frame": 13,
        },
    ]

    metrics = evaluate_records(annotations, records)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0


def test_evaluation_respects_frame_tolerance() -> None:
    annotations = [Annotation("dataset/video.mp4", 10, 12)]
    records = [
        {
            "kind": "event",
            "video_path": "dataset/video.mp4",
            "confirmation_frame": 14,
        }
    ]

    without_tolerance = evaluate_records(annotations, records)
    with_tolerance = evaluate_records(annotations, records, tolerance_frames=2)

    assert without_tolerance["true_positives"] == 0
    assert with_tolerance["true_positives"] == 1


def test_summary_reports_contact_distributions() -> None:
    records = [
        {
            "kind": "frame",
            "detection_count": 2,
            "active_track_count": 2,
        },
        {
            "kind": "diagnostic",
            "contact": True,
            "emitted": False,
            "overlap_ratio": 0.1,
            "closing_speed_px": 2.0,
            "spatial_distance_px": 1.0,
        },
        {
            "kind": "diagnostic",
            "contact": True,
            "emitted": True,
            "overlap_ratio": 0.3,
            "closing_speed_px": 4.0,
            "spatial_distance_px": 0.0,
        },
    ]

    summary = summarize_diagnostics(records)

    assert summary["frame_rows"] == 1
    assert summary["frames_with_detections"] == 1
    assert summary["frames_with_multiple_tracks"] == 1
    assert summary["max_active_tracks"] == 2
    assert summary["diagnostic_rows"] == 2
    assert summary["contact_rows"] == 2
    assert summary["emitted_rows"] == 1
    assert summary["contact_overlap_ratio"]["median"] == 0.2


def test_loads_real_metadata_csv(tmp_path) -> None:
    metadata = tmp_path / "metadata-real.csv"
    metadata.write_text(
        "path,type,rollover,accident_time,accident_frame,no_frames,duration,"
        "region,scene_layout,weather,day_time,quality,"
        "split_in_distribution,split_geo_aware\n"
        "real_videos/Crash_A.mp4,t-bone,1,2.0,40,200,10.0,EU,urban,"
        "clear,day,high,test,train\n",
        encoding="utf-8",
    )

    annotations = load_annotations(metadata)

    assert len(annotations) == 1
    assert annotations[0].video_path == "real_videos/Crash_A.mp4"
    assert annotations[0].start_frame == 40
    assert annotations[0].fps == 20.0
    assert annotations[0].rollover is True
    assert annotations[0].label == "t-bone"


def test_evaluation_matches_absolute_log_path_and_seconds_tolerance() -> None:
    annotations = [
        Annotation(
            "real_videos/Crash_A.mp4",
            40,
            40,
            fps=20.0,
            label="t-bone",
        )
    ]
    records = [
        {
            "kind": "metadata",
            "video_path": "C:/dataset/real_videos/crash_a.mp4",
        },
        {
            "kind": "event",
            "video_path": "C:/dataset/real_videos/crash_a.mp4",
            "confirmation_frame": 55,
        },
    ]

    metrics = evaluate_records(annotations, records, tolerance_seconds=1.0)

    assert metrics["true_positives"] == 1
    assert metrics["mean_delay_frames"] == 15.0
    assert metrics["mean_delay_seconds"] == 0.75


def test_group_evaluation_does_not_count_other_types_as_false_positives() -> None:
    annotations = [
        Annotation("real_videos/a.mp4", 10, 10, label="rear-end"),
        Annotation("real_videos/b.mp4", 20, 20, label="t-bone"),
    ]
    records = [
        {"kind": "event", "video_path": "a.mp4", "confirmation_frame": 10},
        {"kind": "event", "video_path": "b.mp4", "confirmation_frame": 20},
    ]

    grouped = evaluate_by_group(annotations, records, "label")

    assert grouped["rear-end"]["false_positives"] == 0
    assert grouped["t-bone"]["false_positives"] == 0


def test_annotation_summary_exposes_timestamp_disagreement() -> None:
    summary = summarize_annotations(
        [
            Annotation(
                "video.mp4",
                50,
                50,
                label="single",
                fps=10.0,
                accident_time_s=4.5,
            )
        ]
    )

    assert summary["annotated_videos"] == 1
    assert summary["accident_types"] == {"single": 1}
    assert summary["frame_timestamp_error_seconds"]["max"] == 0.5


def test_evaluation_prefers_decoder_timestamp_when_available() -> None:
    annotations = [
        Annotation(
            "video.mp4",
            100,
            100,
            fps=10.0,
            accident_time_s=5.0,
        )
    ]
    records = [
        {
            "kind": "event",
            "video_path": "video.mp4",
            "confirmation_frame": 50,
            "timestamp_s": 5.2,
        }
    ]

    metrics = evaluate_records(annotations, records, tolerance_seconds=0.5)

    assert metrics["true_positives"] == 1
    assert abs(metrics["mean_delay_seconds"] - 0.2) < 1e-9
