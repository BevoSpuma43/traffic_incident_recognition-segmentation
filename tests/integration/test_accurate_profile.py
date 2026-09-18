from pathlib import Path

import pytest

from cctv_incident.config import AppConfig


@pytest.mark.parametrize("available", [True, False])
def test_accurate_profile_uses_medium_model_and_higher_frame_rate(tmp_path, monkeypatch, available):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    cfg = AppConfig()
    cfg.project.root_dir = tmp_path
    cfg.project.output_dir = tmp_path / "outputs"
    cfg.calibration.file = tmp_path / "calibration.yaml"
    cfg.events.coordinate_mode = "image"
    model = tmp_path / "models/yolo26m-seg.pt"
    if available:
        model.parent.mkdir()
        model.touch()
    monkeypatch.setattr("cctv_incident.config.load_config", lambda path: cfg.model_copy(deep=True))
    captured = []

    class CapturePipeline:
        def __init__(self, selected):
            captured.append(selected)

        def run(self, callback):
            raise RuntimeError("test capture: inference intentionally skipped")

    monkeypatch.setattr("cctv_incident.pipeline.Pipeline", CapturePipeline)
    app = AppTest.from_file(str(Path("apps/streamlit_app.py").resolve())).run(timeout=20)
    next(item for item in app.selectbox if item.label == "Qualità analisi").select("Accurata").run(
        timeout=20
    )
    assert not app.exception
    if not available:
        assert any("non è installato" in item.value for item in app.error)
        assert not any(item.label == "Avvia analisi" for item in app.button)
        assert not captured
        return
    next(item for item in app.button if item.label == "Avvia analisi").click().run(timeout=20)
    assert not app.exception
    assert len(captured) == 1
    assert captured[0].perception.model == model
    assert captured[0].perception.image_size == 640
    assert captured[0].video.target_fps == 15
