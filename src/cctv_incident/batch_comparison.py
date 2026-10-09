"""Compare completed paired experiments only when their controlled inputs agree."""

from pathlib import Path

from .batch import completed_results, read_json
from .batch_evaluation import aggregate_results
from .calibration.batch_snapshot import configure_snapshot
from .calibration.repository import load_record
from .config import AppConfig


def compare_jobs(image_job, metric_job):
    jobs = [Path(image_job), Path(metric_job)]
    manifests = [read_json(job / "manifest.json") for job in jobs]
    reasons = []
    reports = {}
    for mode, job, manifest in zip(("image", "metric"), jobs, manifests, strict=True):
        if manifest.get("schema_version") != 2 or manifest.get("coordinate_mode") != mode:
            reasons.append(f"{mode}: requires an explicit version 2 {mode} manifest")
        results = completed_results(job)
        expected = {v["relative_path"]: v for v in manifest["videos"]}
        if len(results) != len(expected) or {r["video"] for r in results} != set(expected):
            reasons.append(f"{mode}: incomplete or duplicate video results")
        for result in results:
            summary = result["summary"]
            if (
                not summary.get("evaluable")
                or not summary.get("completed")
                or summary.get("stopped")
                or summary.get("error")
            ):
                reasons.append(f"{mode}: video without explicit evaluable outcome")
            saved = expected.get(result["video"], {})
            if result.get("label") != saved.get("label"):
                reasons.append(f"{mode}: result labels differ from manifest")
            if mode == "metric" and result.get("calibration") != saved.get("calibration"):
                reasons.append("metric: result calibration differs from manifest")
            if result.get("video_sha256") != expected.get(result["video"], {}).get("sha256"):
                reasons.append(f"{mode}: result source hash differs from manifest")
        reports[mode] = aggregate_results(results, len(expected), manifest["tolerance_s"])
    left, right = manifests
    for field in ("code_sha256", "model_sha256", "metadata_sha256", "tolerance_s"):
        if left.get(field) is None or left.get(field) != right.get(field):
            reasons.append(f"Different or absent {field}")
    signatures = [
        {v["relative_path"]: (v.get("sha256"), v["label"]) for v in m["videos"]} for m in manifests
    ]
    if signatures[0] != signatures[1] or any(not v[0] for v in signatures[0].values()):
        reasons.append("Video contents or saved labels differ")
    configs = [AppConfig.model_validate(m["config"]) for m in manifests]
    if configs[0].project.seed != configs[1].project.seed:
        reasons.append("Different random seeds")
    for field in ("detect_camera_motion", "motion_threshold_px"):
        if getattr(configs[0].calibration, field) != getattr(configs[1].calibration, field):
            reasons.append(f"Different camera guard setting: {field}")
    for section in ("video", "perception", "tracking", "features", "events"):
        values = [getattr(c, section).model_dump(mode="json") for c in configs]
        ignored = {
            "video": ("source", "clip_id"),
            "perception": ("model",),
            "events": ("coordinate_mode",),
        }.get(section, ())
        for value in values:
            for key in ignored:
                value.pop(key, None)
        if values[0] != values[1]:
            reasons.append(f"Different {section} settings; use a shared base configuration")
    for video in right["videos"]:
        try:
            configure_snapshot(configs[1], jobs[1], video)
            record = load_record(configs[1].calibration.file)
            width, height = record.video.image_size
            full = {(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)}
            if record.roi_px is not None and set(record.roi_px) != full:
                reasons.append(
                    f"Different analysis area: {video['relative_path']} has a restricted metric ROI"
                )
        except (ValueError, OSError) as exc:
            reasons.append(str(exc))
    return {
        "comparable": not reasons,
        "reasons": sorted(set(reasons)),
        "jobs": [str(j) for j in jobs],
        "metrics": reports if not reasons else None,
        "scale_provenance": [
            {
                "video": v["relative_path"],
                "width": (v.get("calibration") or {}).get("width"),
                "length": (v.get("calibration") or {}).get("length"),
                "explicit_scale": (v.get("calibration") or {}).get("explicit_scale"),
            }
            for v in right["videos"]
        ],
        "detectors": {
            "image": "ImageEventDetector: pixel motion normalized by vehicle size; contact/occlusion and sideswipe rules",
            "metric": "EventDetector: metric distance, speed, deceleration and TTC; calibrated ground plane",
        },
        "interpretation": "Comparison of two detector workflows, not an isolated homography ablation; assumed scales are not ground truth",
    }
