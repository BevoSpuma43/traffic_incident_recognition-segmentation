from pathlib import Path

import pytest


def test_streamlit_loads_without_exceptions():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path("apps/streamlit_app.py").resolve())).run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "CCTV · Incident Detection"


def test_streamlit_real_video_preset_without_metric_calibration():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path("apps/streamlit_app.py").resolve())).run(timeout=20)
    next(item for item in app.selectbox if item.label == "Modalita").select(
        "Video reale senza calibrazione"
    ).run(timeout=20)
    assert not app.exception
    config_field = next(item for item in app.text_input if item.label == "Configurazione YAML")
    assert config_field.value == "configs/accident-image.yaml"
    assert any("coordinate immagine" in item.value for item in app.info)
