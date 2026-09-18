import json
from pathlib import Path

import av
import numpy as np
import pytest

from cctv_incident.preview import PreviewWriter
from cctv_incident.replay import draw_impact_markers, export_replay, locate_impacts
from cctv_incident.storage import EventStorage
from cctv_incident.types import Event


@pytest.fixture
def saved_analysis(tmp_path, config):
    """A saved run with delayed confirmation and irregular source timestamps."""
    config.project.root_dir = tmp_path
    config.project.output_dir = tmp_path / "outputs"
    config.video.source = str(tmp_path / "source.mp4")
    config.events.coordinate_mode = "image"
    config.video.target_fps = 10
    times = [0, 0.1, 2, 2.2, 2.4, 3]
    writer = PreviewWriter(config.video.source, fps=10)
    for timestamp in times:
        writer.append(np.full((240, 320, 3), 40, np.uint8), timestamp)
    writer.close()
    run_dir = config.project.output_dir / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "config": config.model_dump(mode="json"),
                "processed_image_size": [320, 240],
                "calibration": {"units": "px"},
            }
        )
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "decoded_frames": len(times),
                "processed_frames": 3,
                "events": 1,
            }
        )
    )
    rows, features = [], []
    for timestamp, boxes in [
        (0, [[60, 90, 100, 130], [180, 90, 220, 130]]),
        (2, [[95, 90, 135, 130], [165, 90, 205, 130]]),
        (3, [[190, 90, 230, 130], [260, 90, 300, 130]]),
    ]:
        for key, box in enumerate(boxes, 1):
            rows.append(
                {
                    "track_id": key,
                    "timestamp_s": timestamp,
                    "bbox": box,
                    "point_px": [(box[0] + box[2]) / 2, box[3]],
                }
            )
        features.append(
            {
                "timestamp_s": timestamp,
                "coordinate_units": "px",
                "state": "NORMAL",
                "score": 0,
                "motions": {"1": {"speed": 10}, "2": {"speed": 20}},
            }
        )
    (run_dir / "trajectories.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run_dir / "features.jsonl").write_text("".join(json.dumps(row) + "\n" for row in features))
    event = Event("impact", "camera_01", 1.8, 2, 3, [1, 2], 0.9, 1, ["stop"], run_id="test-run")
    storage = EventStorage(config.project.output_dir)
    try:
        storage.save(event)
    finally:
        storage.close()
    return config, run_dir, times


@pytest.mark.parametrize("with_event", [True, False])
def test_saved_replay_preserves_frames_and_marks_impact_before_confirmation(
    saved_analysis, with_event
):
    config, run_dir, times = saved_analysis
    if not with_event:
        storage = EventStorage(config.project.output_dir)
        try:
            storage.connection.execute("DELETE FROM events")
            storage.connection.commit()
        finally:
            storage.close()
    result = export_replay(run_dir)
    assert result["frames"] == len(times)
    with av.open(result["video_path"]) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        frames = list(container.decode(video=0))
    np.testing.assert_allclose([frame.time for frame in frames], times, atol=0.002)
    assert len(frames) == len(times)
    before = frames[1].to_ndarray(format="bgr24")
    at_impact = frames[2].to_ndarray(format="bgr24")
    # The green box is absent from the uniform source and present in the replay.
    assert np.max(np.abs(before[90, 60:101].astype(int) - 40)) > 40
    for image in [before, at_impact]:
        assert image.shape == (240, 320, 3)
    before_point = before[108:113, 148:153].mean(axis=(0, 1))
    after_point = at_impact[108:113, 148:153].mean(axis=(0, 1))
    assert before_point[2] < 100
    if with_event:
        assert after_point[2] > 180 and after_point[1] < 100
        assert result["impacts"][0]["point_px"] == [150, 110]
        assert result["impacts"][0]["sample_time_s"] == 2
        assert frames[2].time < result["impacts"][0]["confirm_time_s"]
    else:
        assert result["impacts"] == []
        assert after_point[2] < 100
    assert (run_dir / "replay.json").is_file()
    assert not (run_dir / "annotated.partial.mp4").exists()


def test_missing_impact_observations_do_not_invent_a_position(saved_analysis):
    _, run_dir, _ = saved_analysis
    events = [{"event_id": "missing", "impact_time_s": 2, "confirm_time_s": 3, "track_ids": [99]}]
    markers = locate_impacts(run_dir / "trajectories.jsonl", events)
    assert markers[0]["point_px"] is None
    image = np.zeros((240, 320, 3), np.uint8)
    np.testing.assert_array_equal(draw_impact_markers(image.copy(), markers, 3), image)


def test_replay_does_not_publish_a_truncated_source(saved_analysis):
    _, run_dir, _ = saved_analysis
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["decoded_frames"] += 1
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="tutti i frame"):
        export_replay(run_dir)
    assert not (run_dir / "annotated.mp4").exists()
    assert not (run_dir / "annotated.partial.mp4").exists()


def test_review_ui_generates_existing_analysis_and_seeks_after_rerun(saved_analysis, monkeypatch):
    from streamlit.testing.v1 import AppTest

    config, run_dir, _ = saved_analysis
    monkeypatch.setattr(
        "cctv_incident.config.load_config", lambda path: config.model_copy(deep=True)
    )
    app = AppTest.from_file(str(Path("apps/streamlit_app.py").resolve())).run(timeout=20)
    assert not app.exception
    assert "Rivedi analisi" in [tab.label for tab in app.tabs]
    next(button for button in app.button if button.label == "Prepara video annotato").click().run(
        timeout=20
    )
    assert not app.exception
    assert (run_dir / "annotated.mp4").is_file()
    assert len(app.get("video")) == 1
    next(select for select in app.selectbox if select.label == "Vai a un impatto").select(1).run(
        timeout=20
    )
    assert not app.exception
    assert app.get("video")[0].proto.start_time == 1
    assert len(app.get("download_button")) >= 1
    app.run(timeout=20)
    assert not app.exception
    assert len(app.get("video")) == 1
