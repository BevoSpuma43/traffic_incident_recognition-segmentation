from pathlib import Path

from cctv_incident import batch
from cctv_incident.config import AppConfig


def test_folder_model_selection_controls_progress_and_saved_resume(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from cctv_incident import batch_ui

    folder = tmp_path / "dataset/standard_dataset"
    folder.mkdir(parents=True)
    (folder / "a.mp4").write_bytes(b"fixture")
    (folder.parent / "metadata-real.csv").write_text(
        "path,type,accident_time,duration\nreal_videos/a.mp4,t-bone,1,3\n", encoding="utf-8"
    )
    models = tmp_path / "models"
    models.mkdir()
    for name in ("yolo26s-seg.pt", "yolo26m-seg.pt"):
        (models / name).write_bytes(name.encode())
    cfg = AppConfig()
    cfg.project.root_dir = tmp_path
    cfg.project.output_dir = tmp_path / "outputs"
    cfg.calibration.file = tmp_path / "calibration.yaml"
    cfg.events.coordinate_mode = "image"
    monkeypatch.setattr("cctv_incident.config.load_config", lambda path: cfg.model_copy(deep=True))
    active = set()

    def start(job):
        active.add(str(job))
        (job / "stop.request").unlink(missing_ok=True)
        state = batch.read_json(job / "checkpoint.json")
        batch.write_json(
            job / "checkpoint.json",
            {
                **state,
                "status": "running",
                "current_video": "a.mp4",
                "current_index": 1,
                "progress": 0.5,
                "timestamp_s": 1.5,
            },
        )

    def stop(job):
        active.discard(str(job))
        batch.stop_job(job)
        state = batch.read_json(job / "checkpoint.json")
        batch.write_json(job / "checkpoint.json", {**state, "status": "paused"})

    monkeypatch.setattr(batch_ui, "start_job", start)
    monkeypatch.setattr(batch_ui, "stop_job", stop)
    monkeypatch.setattr(batch_ui, "worker_alive", lambda job: str(job) in active)
    monkeypatch.setattr(batch, "worker_alive", lambda job: str(job) in active)

    def open_batch():
        app = AppTest.from_file(str(Path("apps/streamlit_app.py").resolve())).run(timeout=20)
        next(x for x in app.selectbox if x.label == "Modalita").select(
            "standard_dataset analisi in batch - no omografia"
        ).run(timeout=20)
        assert not app.exception
        return app

    def button(app, label):
        return next(x for x in app.button if x.label == label)

    app = open_batch()
    button(app, "Prepara batch").click().run(timeout=20)
    assert not app.exception
    first = app.session_state["batch_selected_job"]
    button(app, "Avvia batch").click().run(timeout=20)
    assert not app.exception
    assert any("Video 1/1" in x.value and "a.mp4" in x.value for x in app.markdown)
    app.run(timeout=20)
    assert not button(app, "Stop").disabled
    button(app, "Stop").click().run(timeout=20)
    app.run(timeout=20)
    assert not button(app, "Riprendi").disabled
    assert app.dataframe[0].value.loc[1, "true_negatives"] != 0  # Undefined, not invented TNs.
    # A new browser session rediscovers the on-disk checkpoint.
    reopened = open_batch()
    assert reopened.session_state["batch_selected_job"] == first
    button(reopened, "Riprendi").click().run(timeout=20)
    assert first in active
    reopened.run(timeout=20)
    button(reopened, "Stop").click().run(timeout=20)
    reopened.selectbox(key="batch_model").select(models / "yolo26m-seg.pt").run(timeout=20)
    button(reopened, "Prepara batch").click().run(timeout=20)
    assert not reopened.exception
    assert reopened.session_state["batch_selected_job"] != first
    assert batch.snapshot(first)["status"] == "paused"
