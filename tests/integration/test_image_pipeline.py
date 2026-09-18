import json

import av

from cctv_incident.demo import generate_demo
from cctv_incident.pipeline import Pipeline
from cctv_incident.preview import PreviewWriter


def test_image_pipeline_without_metric_calibration(tmp_path, config):
    config.project.output_dir = tmp_path / "outputs"
    config.video.source = str(generate_demo(tmp_path, negative=True))
    config.video.max_width = 320
    config.video.max_frames = 30
    config.perception.backend = "synthetic"
    config.events.coordinate_mode = "image"
    config.calibration.file = tmp_path / "does-not-exist.yaml"
    config.calibration.detect_camera_motion = False
    config.ui.render_bird_eye = False
    preview = PreviewWriter(tmp_path / "preview.mp4", config.video.target_fps)
    try:
        summary = Pipeline(config).run(
            lambda data: preview.append(data["frame"], data["timestamp_s"])
        )
    finally:
        preview.close()
    assert summary["error"] is None
    assert summary["coordinate_mode"] == "image"
    assert summary["track_observations"] > 0
    assert summary["calibration_valid"]
    assert not summary["metric_calibration_available"]
    assert "PAUSED" not in summary["state_counts"]
    run = tmp_path / "outputs/runs" / summary["run_id"]
    rows = [json.loads(line) for line in (run / "trajectories.jsonl").read_text().splitlines()]
    assert all(row["point_world"] is None and row["coordinate_units"] == "px" for row in rows)
    with av.open(str(preview.path)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == summary["processed_frames"]
        assert frames[0].width == 320
        assert all(b.time > a.time for a, b in zip(frames, frames[1:], strict=False))
