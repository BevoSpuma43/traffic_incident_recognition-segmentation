from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from cctv_incident.batch import read_json, write_json
from cctv_incident.calibration import preparation as prep
from cctv_incident.calibration.automatic import generate_automatic_proposals
from cctv_incident.calibration.records import (
    CalibrationRecord,
    Distance,
    confirm_record,
    edit_record,
)
from cctv_incident.calibration.repository import load_record, save_record


def video(path, marked=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((240, 320, 3), 45, np.uint8)
    if marked:
        cv2.rectangle(image, (60, 110), (90, 220), (245, 245, 245), -1)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for _ in range(3):
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def job(tmp_path):
    video(tmp_path / "videos/a.mp4")
    video(tmp_path / "videos/b.mp4", False)
    return prep.prepare_session(tmp_path, tmp_path / "videos", automatic_parameters={})


def repair(row):
    record = load_record(row["record_path"])
    record = edit_record(
        record,
        vertices=[
            {"id": f"P{i}", "x": x, "y": y}
            for i, (x, y) in enumerate([(40, 100), (220, 100), (220, 220), (40, 220)], 1)
        ],
        width=Distance(value=4, origin="measured", user_confirmed=True),
        length=Distance(value=6, origin="measured", user_confirmed=True),
        geometric_quality=0.6,
    )
    return confirm_record(record)


def test_automatic_acceptance_failure_table_manual_repair_and_barrier(job):
    folder = read_json(job / "manifest.json")["folder"]
    with pytest.raises(ValueError, match="Completa"):
        prep.prepared_archive_mapping(job, folder)
    prep.run_preparation(job)
    rows = prep.table_rows(job)
    assert [r["preparation"] for r in rows] == ["auto_calibrated", "manual_required"]
    assert prep.snapshot(job)["status"] == "completed"
    record = load_record(rows[0]["record_path"])
    assert record.status == "confirmed" and record.runtime.metric_valid(0.2)
    assert record.automatic_acceptance == "experimental_usa_v1"
    assert not record.width.user_confirmed and not record.length.user_confirmed
    assert not list(job.rglob("metrics*"))
    readiness = prep.analysis_readiness(job)
    assert not readiness["ready"] and [r["video"] for r in readiness["failed"]] == ["b.mp4"]
    with pytest.raises(ValueError, match="b.mp4"):
        prep.prepared_archive_mapping(job, folder)
    save_record(repair(rows[1]), read_json(job / "manifest.json")["archive"])
    assert prep.analysis_readiness(job)["ready"]
    assert set(prep.prepared_archive_mapping(job, folder)) == {"a.mp4", "b.mp4"}
    with pytest.raises(ValueError, match="cartella"):
        prep.prepared_archive_mapping(job, Path(folder).parent)


def test_acceptance_is_revoked_by_edit_and_never_becomes_manual_confirmation(job):
    prep.run_preparation(job)
    record = load_record(prep.table_rows(job)[0]["record_path"])
    changed = edit_record(record, width=Distance(value=10, origin="experimental", source="changed"))
    assert (
        changed.status == "draft"
        and changed.automatic_acceptance is None
        and changed.runtime is None
    )
    with pytest.raises(ValueError, match="confirmed provenance"):
        confirm_record(record)
    tampered = record.model_dump(mode="json")
    tampered["width"]["value"] = 10
    with pytest.raises(ValueError, match="unchanged experimental"):
        CalibrationRecord.model_validate(tampered)


def test_failure_continues_and_vehicle_evidence_is_durable(job):
    calls = []

    def proposer(source, image, **kwargs):
        calls.append(source.name)
        if source.name == "a.mp4":
            raise RuntimeError("first failed")
        result = generate_automatic_proposals(source, image, **kwargs)
        result.evidence_images["0"] = image.copy()
        return result

    prep.run_preparation(job, proposer=proposer)
    assert calls == ["a.mp4", "b.mp4"]
    assert prep.snapshot(job)["status"] == "completed"
    assert prep.table_rows(job)[0]["preparation"] == "error"
    proposal = prep.load_proposal(job, 1)
    assert "0" in proposal.evidence_images
    (job / "items/000001/evidence-0.png").write_bytes(b"bad")
    with pytest.raises(ValueError, match="modificata"):
        prep.load_proposal(job, 1)


def test_existing_drafts_are_retried_but_confirmed_records_are_preserved(job):
    manifest = read_json(job / "manifest.json")
    legacy = prep.prepare_session(manifest["project_root"], manifest["folder"])
    prep.run_preparation(legacy)
    assert prep.table_rows(legacy)[0]["state"] == "draft"
    prep.run_preparation(job)
    rows = prep.table_rows(job)
    assert rows[0]["state"] == "confirmed" and rows[0]["revision"] == 2
    original = Path(rows[0]["record_path"]).read_bytes()
    second = prep.prepare_session(
        manifest["project_root"], manifest["folder"], automatic_parameters={}
    )
    prep.run_preparation(second)
    assert Path(rows[0]["record_path"]).read_bytes() == original
    assert prep.table_rows(second)[0]["preparation"] == "kept"


def test_changed_automatic_policy_prevents_resume(job):
    manifest = read_json(job / "manifest.json")
    manifest["automatic_parameters"]["seconds"] = 5
    write_json(job / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Ipotesi metriche"):
        prep.validate_session(job)


def preparation_page(root):
    from pathlib import Path

    import streamlit as st
    from streamlit.runtime import Runtime

    from cctv_incident.calibration_preparation_ui import render_preparation_page
    from cctv_incident.components.calibration_editor import CSS, HTML, JS
    from cctv_incident.config import AppConfig

    if Runtime.instance().bidi_component_registry.get("road_calibration_editor") is None:
        st.components.v2.component("road_calibration_editor", html=HTML, css=CSS, js=JS)
    cfg = AppConfig()
    cfg.project.root_dir = Path(root)
    render_preparation_page(cfg)


def test_summary_shows_failures_and_manual_repair_enables_analysis(job):
    from streamlit.testing.v1 import AppTest

    prep.run_preparation(job)
    manifest = read_json(job / "manifest.json")
    app = AppTest.from_function(preparation_page, args=(manifest["project_root"],)).run(timeout=20)
    assert not app.exception
    assert app.dataframe[0].value["Accettazione"].tolist() == ["Automatica sperimentale", "—"]
    assert app.dataframe[1].value["Video da calibrare a mano"].tolist() == ["b.mp4"]
    assert app.button(key="prep_analysis").disabled
    assert app.session_state["metric_preparation_job"] == str(job)
    save_record(repair(prep.table_rows(job)[1]), manifest["archive"])
    app.button(key="prep_refresh").click().run(timeout=20)
    assert not app.exception
    assert not app.button(key="prep_analysis").disabled
    assert len(app.dataframe) == 1
    app.button(key="prep_analysis").click().run(timeout=20)
    assert not app.exception
    assert app.session_state["metric_batch_stage"] == "Analisi batch"
    assert not list((Path(manifest["project_root"]) / "outputs/batches").glob("*/manifest.json"))


def test_automatic_snapshots_keep_provenance_and_are_independent_of_archive(job):
    from cctv_incident.calibration.batch_snapshot import (
        configure_snapshot,
        create_snapshots,
        csv_provenance,
    )
    from cctv_incident.calibration.runtime import load_run_calibration
    from cctv_incident.config import AppConfig

    prep.run_preparation(job)
    manifest = read_json(job / "manifest.json")
    rows = prep.table_rows(job)
    save_record(repair(rows[1]), manifest["archive"])
    mapping = prep.prepared_archive_mapping(job, manifest["folder"])
    cfg = AppConfig()
    cfg.project.root_dir = Path(manifest["project_root"])
    cfg.calibration.min_confidence = 0.2
    videos = [{"relative_path": r["video"]} for r in rows]
    destination = job / "snapshot-test"
    create_snapshots(cfg, manifest["folder"], videos, mapping, destination)
    assert videos[0]["calibration"]["automatic_acceptance"] == "experimental_usa_v1"
    assert (
        csv_provenance({"calibration": videos[0]["calibration"]})["automatic_acceptance"]
        == "experimental_usa_v1"
    )
    assert videos[1]["calibration"]["automatic_acceptance"] is None
    # Editing the archive does not mutate the experiment's accepted dimensions.
    original_width = load_record(rows[0]["record_path"]).width.value
    save_record(repair(rows[0]), manifest["archive"])
    cfg.video.source = str(Path(manifest["folder"]) / "a.mp4")
    configure_snapshot(cfg, destination, videos[0])
    run = load_run_calibration(cfg)
    assert run.record.width.value == original_width
    assert run.record.automatic_acceptance == "experimental_usa_v1"


def test_stop_resume_does_not_recalibrate_committed_videos(job):
    calls = []

    def proposer(source, image, **kwargs):
        calls.append(source.name)
        if len(calls) == 2:
            prep.stop_preparation(job)
        return generate_automatic_proposals(source, image, **kwargs)

    prep.run_preparation(job, proposer=proposer)
    assert prep.snapshot(job)["status"] == "paused"
    assert prep.snapshot(job)["completed_videos"] == 1
    first = (job / "items/000000/result.json").read_bytes()
    assert not prep.analysis_readiness(job)["ready"]
    (job / "stop.request").unlink()
    prep.run_preparation(job, proposer=proposer)
    assert calls == ["a.mp4", "b.mp4", "b.mp4"]
    assert (job / "items/000000/result.json").read_bytes() == first
    assert prep.snapshot(job)["status"] == "completed"


def selection_batch_config(job):
    from cctv_incident.config import AppConfig

    manifest = read_json(job / "manifest.json")
    cfg = AppConfig()
    cfg.project.root_dir = Path(manifest["project_root"])
    cfg.perception.backend = "synthetic"
    cfg.perception.model = cfg.project.root_dir / "fixture.pt"
    metadata = cfg.project.root_dir / "metadata.csv"
    # Excluded videos deliberately have no label: they must never enter preflight.
    metadata.write_text("path,type,accident_time,duration\na.mp4,normal,,0.3\n", encoding="utf-8")
    return cfg, Path(manifest["folder"]), metadata


def test_shared_sample_survives_code_change_and_freezes_both_modes(job, monkeypatch):
    from cctv_incident import batch
    from cctv_incident.batch_selection import (
        active_selection,
        create_automatic_selection,
        load_selection,
    )

    prep.run_preparation(job)
    # Completed outputs can be consumed after this update without repeating YOLO.
    monkeypatch.setattr(prep, "preparation_signature", lambda: "updated code")
    with pytest.raises(ValueError, match="codice"):
        prep.validate_session(job)
    path = create_automatic_selection(job)
    cfg, folder, metadata = selection_batch_config(job)
    assert active_selection(cfg.project.root_dir) == path
    sample = load_selection(path)
    assert [v["relative_path"] for v in sample["videos"]] == ["a.mp4"]
    assert [v["relative_path"] for v in sample["excluded"]] == ["b.mp4"]
    # New videos and a later manual repair do not expand the frozen sample.
    save_record(repair(prep.table_rows(job)[1]), read_json(job / "manifest.json")["archive"])
    video(folder / "c.mp4")
    seen = []

    class Pipeline:
        def __init__(self, config, segmenter):
            self.cfg = config

        def run(self, **kwargs):
            seen.append((self.cfg.events.coordinate_mode, Path(self.cfg.video.source).name))
            return {
                "run_id": "fixture",
                "stopped": False,
                "completed": True,
                "error": None,
                "evaluable": True,
                "metric_calibration_available": True,
                "calibration_valid": True,
                "source_duration_s": 0.3,
                "elapsed_s": 0.1,
            }

    manifests = []
    for mode in ("metric", "image"):
        cfg.events.coordinate_mode = mode
        experiment = batch.prepare_job(cfg, folder, metadata, selection_path=path)
        manifest = read_json(experiment / "manifest.json")
        manifests.append(manifest)
        assert [v["relative_path"] for v in manifest["videos"]] == ["a.mp4"]
        batch.run_job(experiment, Pipeline, lambda _: object())
        metrics = read_json(experiment / "metrics.json")
        assert metrics["total_videos"] == metrics["completed_videos"] == 1
        assert metrics["selection_id"] == sample["selection_id"]
    assert seen == [("metric", "a.mp4"), ("image", "a.mp4")]
    assert manifests[0]["video_selection"] == manifests[1]["video_selection"]
    assert manifests[0]["videos"][0]["calibration"]["automatic_acceptance"]


def test_shared_sample_rejects_changed_input_in_both_modes(job):
    from cctv_incident.batch import prepare_job
    from cctv_incident.batch_selection import create_automatic_selection

    prep.run_preparation(job)
    path = create_automatic_selection(job)
    cfg, folder, metadata = selection_batch_config(job)
    (folder / "a.mp4").write_bytes(b"changed")
    for mode in ("image", "metric"):
        cfg.events.coordinate_mode = mode
        with pytest.raises(ValueError, match="modificato o mancante"):
            prepare_job(cfg, folder, metadata, selection_path=path)


def test_manual_records_do_not_enter_automatic_sample_and_empty_is_rejected(job):
    from cctv_incident.batch_selection import create_automatic_selection, load_selection

    with pytest.raises(ValueError, match="Completa"):
        create_automatic_selection(job)
    prep.run_preparation(job)
    archive = read_json(job / "manifest.json")["archive"]
    rows = prep.table_rows(job)
    save_record(repair(rows[1]), archive)
    path = create_automatic_selection(job)
    assert len(load_selection(path)["videos"]) == 1
    save_record(repair(rows[0]), archive)
    with pytest.raises(ValueError, match="Nessun video"):
        create_automatic_selection(job)


def test_selection_button_persists_sample_and_image_batch_recovers_it(job, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from cctv_incident import batch
    from cctv_incident.batch_selection import active_selection

    prep.run_preparation(job)
    cfg, folder, metadata = selection_batch_config(job)
    model = cfg.project.root_dir / "models/fixture-seg.pt"
    model.parent.mkdir()
    model.write_bytes(b"fixture model")
    cfg.perception.model = model
    manifest = read_json(job / "manifest.json")
    app = AppTest.from_function(preparation_page, args=(manifest["project_root"],)).run(timeout=20)
    assert not app.exception and app.button(key="prep_analysis").disabled
    assert (
        app.button(key="prep_use_automatic").label == "Usa solo i 1 video calibrati automaticamente"
    )
    app.button(key="prep_use_automatic").click().run(timeout=20)
    assert not app.exception and not app.error
    assert active_selection(cfg.project.root_dir)
    assert app.session_state["metric_batch_stage"] == "Analisi batch"
    monkeypatch.setattr("cctv_incident.config.load_config", lambda _: cfg.model_copy(deep=True))
    # Fresh browser state must recover the selection from disk in image mode.
    other = AppTest.from_file(str(Path("apps/streamlit_app.py").resolve())).run(timeout=20)
    next(w for w in other.selectbox if w.label == "Modalita").select(
        "standard_dataset analisi in batch - no omografia"
    ).run(timeout=20)
    assert not other.exception
    assert other.dataframe[0].value["Video selezionato"].tolist() == ["a.mp4"]
    assert not any(w.label == "Cartella nel dataset" for w in other.selectbox)
    other.text_input(key="batch_metadata").set_value(str(metadata)).run(timeout=20)
    other.button(key="batch_prepare").click().run(timeout=20)
    assert not other.exception and not other.error
    saved = Path(other.session_state["batch_selected_job"])
    assert [v["relative_path"] for v in read_json(saved / "manifest.json")["videos"]] == ["a.mp4"]
    assert batch.snapshot(saved)["status"] == "ready"  # Selection never starts analysis.
    started = []
    monkeypatch.setattr("cctv_incident.batch_ui.start_job", lambda path: started.append(path))
    other._bidi_component_manager = app._bidi_component_manager
    next(w for w in other.selectbox if w.label == "Modalita").select(
        "standard_dataset analisi in batch - con omografia"
    ).run(timeout=20)
    other.radio(key="metric_batch_stage").set_value("Analisi batch").run(timeout=20)
    assert not other.exception
    other.text_input(key="metric_batch_metadata").set_value(str(metadata)).run(timeout=20)
    assert not other.button(key="metric_batch_prepare").disabled
    other.button(key="metric_batch_prepare").click().run(timeout=20)
    assert not other.exception and not other.error
    metric_job = Path(other.session_state["metric_batch_selected_job"])
    assert started == [metric_job]
    assert (
        read_json(metric_job / "manifest.json")["video_selection"]
        == read_json(saved / "manifest.json")["video_selection"]
    )


def test_corrupt_active_selection_never_falls_back_to_full_folder(job):
    from cctv_incident.batch_selection import active_selection, create_automatic_selection

    prep.run_preparation(job)
    path = create_automatic_selection(job)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Campione condiviso modificato"):
        active_selection(read_json(job / "manifest.json")["project_root"])


def test_image_batch_uses_membership_without_loading_homography(job):
    from cctv_incident.batch import prepare_job
    from cctv_incident.batch_selection import create_automatic_selection, load_selection

    prep.run_preparation(job)
    path = create_automatic_selection(job)
    sample = load_selection(path)
    cfg, folder, metadata = selection_batch_config(job)
    (path.parent / sample["videos"][0]["calibration"]["path"]).write_text(
        "corrupt", encoding="utf-8"
    )
    cfg.events.coordinate_mode = "image"
    experiment = prepare_job(cfg, folder, metadata, selection_path=path)
    assert len(read_json(experiment / "manifest.json")["videos"]) == 1
    cfg.events.coordinate_mode = "metric"
    with pytest.raises(ValueError, match="Snapshot calibrazione modificato"):
        prepare_job(cfg, folder, metadata, selection_path=path)
