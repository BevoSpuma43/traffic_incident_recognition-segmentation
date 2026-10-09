import json
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from cctv_incident.config import AppConfig
from cctv_incident.replay import export_replay
from cctv_incident.segmenter import SyntheticSegmenter


@pytest.fixture
def selected_app(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    cfg = AppConfig()
    cfg.project.root_dir = tmp_path
    cfg.project.output_dir = tmp_path / "outputs"
    cfg.events.coordinate_mode = "image"
    cfg.video.source = str(tmp_path / "default.mp4")
    cfg.video.clip_id = "old_clip"
    cfg.calibration.file = tmp_path / "calibration.yaml"
    cfg.calibration.detect_camera_motion = False
    cfg.ui.render_bird_eye = False
    monkeypatch.setattr("cctv_incident.config.load_config", lambda path: cfg.model_copy(deep=True))
    app = AppTest.from_file(str(Path("apps/streamlit_app.py").resolve())).run(timeout=20)
    next(item for item in app.selectbox if item.label == "Modalita").select(
        "Solo Segmentazione"
    ).run(timeout=20)
    assert not app.exception
    return app, cfg


def source_choice(app, choice):
    next(item for item in app.radio if item.label == "Sorgente video").set_value(choice).run(
        timeout=20
    )
    assert not app.exception


def start_button(app):
    return next(item for item in app.button if item.label == "Avvia analisi")


def capture_pipeline(monkeypatch):
    captured = []

    class CapturePipeline:
        def __init__(self, cfg):
            captured.append(cfg)

        def run(self, callback):
            raise RuntimeError("test capture: inference intentionally skipped")

    monkeypatch.setattr("cctv_incident.pipeline.Pipeline", CapturePipeline)
    return captured


def test_upload_is_selected_across_reruns_and_clearing_blocks_analysis(selected_app, monkeypatch):
    app, cfg = selected_app
    captured = capture_pipeline(monkeypatch)
    source_choice(app, "Carica un video")
    assert start_button(app).disabled
    app.file_uploader[0].set_value(("mio_incidente.mp4", b"video content", "video/mp4")).run(
        timeout=20
    )
    assert not app.exception
    assert not start_button(app).disabled
    app.run(timeout=20)
    start_button(app).click().run(timeout=20)
    assert not app.exception
    assert len(captured) == 1
    selected = captured[0]
    uploaded_path = Path(selected.video.source)
    assert uploaded_path.is_relative_to(cfg.project.root_dir / "data/uploads")
    assert uploaded_path.name == "mio_incidente.mp4"
    assert uploaded_path.read_bytes() == b"video content"
    assert selected.video.clip_id == "mio_incidente"
    app.file_uploader[0].clear().run(timeout=20)
    assert not app.exception
    assert start_button(app).disabled
    assert uploaded_path.is_file()


def test_full_dataset_selection_does_not_require_sample_manifest(selected_app, monkeypatch):
    app, cfg = selected_app
    captured = capture_pipeline(monkeypatch)
    video = cfg.project.root_dir / "data/raw/ACCIDENT/real_videos/fuori_campione_00.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"dataset video")
    app.run(timeout=20)
    source_choice(app, "Video del dataset ACCIDENT")
    selector = next(item for item in app.selectbox if item.label == "Video del dataset")
    assert "real_videos/fuori_campione_00.mp4" in selector.options
    selector.select("real_videos/fuori_campione_00.mp4").run(timeout=20)
    start_button(app).click().run(timeout=20)
    assert not app.exception
    assert Path(captured[0].video.source) == video
    assert captured[0].video.clip_id == "fuori_campione_00"
    assert captured[0].calibration.camera_id == "source_fuori_campione"


def test_manual_path_replaces_inherited_clip_and_clears_old_calibration(selected_app, monkeypatch):
    app, cfg = selected_app
    captured = capture_pipeline(monkeypatch)
    source_choice(app, "Percorso o URL RTSP")
    app.session_state["calibration_frame"] = np.zeros((10, 10, 3), np.uint8)
    app.session_state["proposal"] = {"old": True}
    source = cfg.project.root_dir / "altro_video.mkv"
    next(item for item in app.text_input if item.label == "Percorso video o URL RTSP").set_value(
        str(source)
    ).run(timeout=20)
    start_button(app).click().run(timeout=20)
    assert not app.exception
    assert Path(captured[0].video.source) == source
    assert captured[0].video.clip_id == "altro_video"
    assert "calibration_frame" not in app.session_state.filtered_state
    assert "proposal" not in app.session_state.filtered_state


def test_uploaded_video_analysis_and_replay_survive_selecting_another_file(
    selected_app, monkeypatch, tmp_path
):
    app, cfg = selected_app
    source = tmp_path / "camera.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 8, (320, 240))
    assert writer.isOpened()
    try:
        for _ in range(8):
            writer.write(np.zeros((240, 320, 3), np.uint8))
    finally:
        writer.release()
    monkeypatch.setattr(
        "cctv_incident.pipeline.create_segmenter", lambda config: SyntheticSegmenter()
    )
    source_choice(app, "Carica un video")
    app.file_uploader[0].set_value(("camera.mp4", source.read_bytes(), "video/mp4")).run(timeout=20)
    start_button(app).click().run(timeout=40)
    assert not app.exception
    assert not app.error
    summary = app.session_state["last_run"]
    assert summary["error"] is None
    assert summary["decoded_frames"] == 8
    run_dir = Path(summary["run_dir"])
    metadata = json.loads((run_dir / "run.json").read_text())
    stored_source = Path(metadata["config"]["video"]["source"])
    assert stored_source.is_file()
    assert metadata["config"]["video"]["clip_id"] == "camera"
    assert (run_dir / "annotated.mp4").is_file()
    app.file_uploader[0].set_value(("secondo.mp4", b"other content", "video/mp4")).run(timeout=20)
    assert not app.exception
    assert stored_source.is_file()
    # Regenerating the older replay still reads its original, not the current upload.
    replay = export_replay(run_dir)
    assert replay["source_name"] == "camera.mp4"
    with av.open(replay["video_path"]) as container:
        assert len(list(container.decode(video=0))) == 8
