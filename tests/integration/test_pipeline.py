import json

import av
import numpy as np
import pytest

from cctv_incident.demo import generate_demo
from cctv_incident.pipeline import Pipeline
from cctv_incident.replay import export_replay
from cctv_incident.storage import EventStorage


@pytest.mark.parametrize("negative", [False, True])
def test_video_to_events_and_clips(tmp_path, config, negative):
    config.project.root_dir = tmp_path
    config.project.output_dir = tmp_path / "outputs"
    config.video.source = str(generate_demo(tmp_path, negative))
    config.video.target_fps = 10
    config.perception.backend = "synthetic"
    config.calibration.camera_id = "synthetic"
    config.calibration.file = tmp_path / "data/calibration/synthetic.yaml"
    config.calibration.detect_camera_motion = False
    frames = []

    def preview(data):
        if not frames:
            frames.append(data["frame"].shape)

    summary = Pipeline(config).run(preview)
    assert summary["error"] is None
    assert summary["events"] == (0 if negative else 1)
    assert summary["processed_frames"] == 80
    assert frames == [(360, 640, 3)]
    storage = EventStorage(config.project.output_dir)
    try:
        events = storage.list_events(summary["run_id"])
        if not negative:
            assert len(events) == 1
            event = events[0]
            assert event["clip_status"] == "saved"
            with av.open(event["clip_path"]) as container:
                decoded = list(container.decode(video=0))
                assert len(decoded) >= 50
                assert decoded[-1].time - decoded[0].time >= 2.5
            assert event["clip_start_s"] <= event["impact_time_s"] - 0.9
            assert event["clip_end_s"] >= event["confirm_time_s"] + 0.9
    finally:
        storage.close()
    run_dir = config.project.output_dir / "runs" / summary["run_id"]
    rows = [json.loads(line) for line in (run_dir / "trajectories.jsonl").read_text().splitlines()]
    assert len({row["track_id"] for row in rows[:100]}) == 2
    assert all(np.isfinite(row["point_world"]).all() for row in rows)
    replay = export_replay(run_dir)
    assert replay["frames"] == summary["decoded_frames"]
    assert len(replay["impacts"]) == summary["events"]
    if not negative:
        assert replay["impacts"][0]["point_px"] is not None
