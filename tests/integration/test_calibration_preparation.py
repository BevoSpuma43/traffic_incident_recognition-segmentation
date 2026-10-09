import time
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from cctv_incident.batch import exclusive_lock, read_json, sha256, worker_alive, write_json
from cctv_incident.calibration import preparation as prep
from cctv_incident.calibration.proposals import generate_proposals
from cctv_incident.calibration.records import Distance, confirm_record, edit_record
from cctv_incident.calibration.repository import create_draft, load_record, save_record


def video(path, *, marked=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((240, 320, 3), 45, np.uint8)
    if marked:
        cv2.rectangle(image, (45, 110), (85, 210), (245, 245, 245), -1)
        cv2.rectangle(image, (155, 110), (195, 210), (245, 245, 245), -1)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for _ in range(3):
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture
def job(tmp_path):
    video(tmp_path / "videos/a.mp4")
    video(tmp_path / "videos/b.mp4", marked=False)
    return prep.prepare_session(tmp_path, tmp_path / "videos")


def test_durable_candidates_empty_result_and_existing_record_preservation(job):
    assert [r["state"] for r in prep.table_rows(job)] == ["missing", "missing"]
    prep.run_preparation(job)
    state = prep.snapshot(job)
    assert state["status"] == "completed" and state["completed_videos"] == 2
    rows = prep.table_rows(job)
    assert [r["state"] for r in rows] == ["draft", "draft"]
    assert [r["preparation"] for r in rows] == ["proposed", "no_reference"]
    proposal = prep.load_proposal(job, 0)
    assert len(proposal.candidates) == 2
    assert proposal.mask.shape == (240, 320) and proposal.preview.shape == (240, 320, 3)
    record = load_record(rows[0]["record_path"])
    assert record.status == "draft" and record.width.value is None and record.length.value is None
    record = confirm_record(
        edit_record(
            record,
            width=Distance(value=1, origin="measured", user_confirmed=True),
            length=Distance(value=3, origin="experimental", source="fixture", user_confirmed=True),
        )
    )
    archive = Path(read_json(job / "manifest.json")["archive"])
    save_record(record, archive)
    before = {str(p): sha256(p) for p in archive.rglob("*") if p.is_file()}
    manifest = read_json(job / "manifest.json")
    second = prep.prepare_session(manifest["project_root"], manifest["folder"])
    prep.run_preparation(
        second,
        proposer=lambda *a, **k: pytest.fail("Never replace existing drafts or confirmations"),
    )
    assert [r["state"] for r in prep.table_rows(second)] == ["confirmed", "draft"]
    assert prep.table_rows(second)[0]["length_origin"] == "experimental"
    assert before == {str(p): sha256(p) for p in archive.rglob("*") if p.is_file()}
    assert not (job / "metrics.json").exists() and not (job / "pipeline").exists()
    prep.validate_session(job)  # Later manual revisions don't alter the original saved proposal.


def test_stop_inside_proposal_restarts_only_uncommitted_video(job):
    calls = []

    def stop_on_second(image, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            prep.stop_preparation(job)
        return generate_proposals(image, **kwargs)

    prep.run_preparation(job, proposer=stop_on_second)
    assert prep.snapshot(job)["status"] == "paused"
    assert prep.snapshot(job)["completed_videos"] == 1
    first = (job / "items/000000/result.json").read_bytes()
    (job / "stop.request").unlink()
    prep.run_preparation(job, proposer=stop_on_second)
    assert len(calls) == 3
    assert (job / "items/000000/result.json").read_bytes() == first
    assert prep.snapshot(job)["status"] == "completed"


def test_crash_between_archive_and_result_recovers_without_new_revision(job, monkeypatch):
    original_write = prep.write_json

    def crash(path, data):
        if Path(path).name == "result.json":
            raise KeyboardInterrupt("simulated process death")
        return original_write(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(prep, "write_json", crash)
        with pytest.raises(KeyboardInterrupt):
            prep.run_preparation(job)
    rows = prep.table_rows(job)
    assert rows[0]["revision"] == 1 and rows[0]["state"] == "draft"
    assert prep.snapshot(job)["completed_videos"] == 0
    prep.run_preparation(job)
    assert prep.table_rows(job)[0]["revision"] == 1
    assert len(prep.load_proposal(job, 0).candidates) == 2


@pytest.mark.parametrize("change", ["video", "add", "proposal", "code", "parameters"])
def test_resume_checks_input_and_artifact_hashes(job, change, monkeypatch):
    prep.run_preparation(job)
    manifest = read_json(job / "manifest.json")
    if change == "video":
        # Preserve mtime/size: SHA, rather than stat alone, must still reject it.
        import os

        path = Path(manifest["folder"]) / "a.mp4"
        stat = path.stat()
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert prep.table_rows(job, verify_sources=True)[0]["state"] == "incompatible"
    elif change == "add":
        video(Path(manifest["folder"]) / "new.mp4")
    elif change == "proposal":
        (job / "items/000000/preview.png").write_bytes(b"corrupt")
    elif change == "parameters":
        manifest["parameters"]["seed"] += 1
        write_json(job / "manifest.json", manifest)
    else:
        monkeypatch.setattr(prep, "preparation_signature", lambda: "changed")
    with pytest.raises(ValueError):
        prep.validate_session(job)


def test_decode_failure_is_a_preparation_error_not_a_negative(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / "bad.mp4").write_bytes(b"not a video")
    job = prep.prepare_session(tmp_path, folder)
    prep.run_preparation(job)
    row = prep.table_rows(job)[0]
    assert row["state"] == row["preparation"] == "error"
    assert row["message"] and not list(job.rglob("metrics*"))


def test_concurrent_manual_save_wins_without_duplicate_record(job):
    manifest = read_json(job / "manifest.json")
    archive = Path(manifest["archive"])
    saved = []

    def racing_proposer(image, **kwargs):
        if not saved:
            record, reference = create_draft(
                Path(manifest["folder"]) / "a.mp4", manifest["project_root"]
            )
            path = save_record(
                edit_record(record, vertices=[{"id": "P1", "x": 12, "y": 15}]),
                archive,
                reference_image=reference,
            )
            saved.append((path, path.read_bytes()))
        return generate_proposals(image, **kwargs)

    prep.run_preparation(job, proposer=racing_proposer)
    assert saved[0][0].read_bytes() == saved[0][1]
    assert len(list(archive.rglob("a.yaml"))) == 1
    assert prep.table_rows(job)[0]["preparation"] == "kept"


def test_worker_execution_lock(job):
    with exclusive_lock(job.parent / ".execution.lock"), pytest.raises(RuntimeError):
        prep.run_preparation(job)


def test_start_reports_changed_source_without_launching(job):
    manifest = read_json(job / "manifest.json")
    (Path(manifest["folder"]) / "a.mp4").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="modificato"):
        prep.start_preparation(job)
    assert prep.snapshot(job)["status"] == "error"
    assert prep.snapshot(job)["error"]
    assert not (job / "worker.json").exists()


def test_proposal_failure_stays_in_table_and_other_videos_continue(job):
    calls = []

    def proposer(image, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("fixture proposal failure")
        return generate_proposals(image, **kwargs)

    prep.run_preparation(job, proposer=proposer)
    rows = prep.table_rows(job)
    assert rows[0]["preparation"] == "error" and "fixture proposal failure" in rows[0]["message"]
    assert rows[1]["state"] == "draft"
    assert not list(job.rglob("events*")) and not list(job.rglob("metrics*"))


def test_incompatible_archive_is_never_overwritten(tmp_path):
    source = video(tmp_path / "videos/a.mp4")
    record, image = create_draft(source, tmp_path)
    archive = tmp_path / "data/calibration/videos"
    path = save_record(record, archive, reference_image=image)
    original = path.read_bytes()
    video(source, marked=False)
    job = prep.prepare_session(tmp_path, source.parent)
    prep.run_preparation(
        job, proposer=lambda *a, **k: pytest.fail("Incompatible input must be reviewed")
    )
    assert prep.table_rows(job)[0]["state"] == "incompatible"
    assert prep.table_rows(job)[0]["preparation"] == "incompatible"
    assert path.read_bytes() == original


def test_real_subprocess_survives_ui_and_resumes(job):
    def wait_exit():
        deadline = time.monotonic() + 45
        while worker_alive(job) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not worker_alive(job), (job / "worker.log").read_text()

    try:
        prep.start_preparation(job)
        with pytest.raises(RuntimeError, match="già attiva"):
            prep.start_preparation(job)
        prep.stop_preparation(job)
        wait_exit()
        assert prep.snapshot(job)["status"] == "paused", (job / "worker.log").read_text()
        assert prep.snapshot(job)["completed_videos"] == 0
        prep.start_preparation(job)
        wait_exit()
        assert prep.snapshot(job)["status"] == "completed", (job / "worker.log").read_text()
        assert prep.snapshot(job)["completed_videos"] == 2
        assert len(prep.load_proposal(job, 0).candidates) == 2
    finally:
        prep.stop_preparation(job)
        wait_exit()


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


def test_preparation_ui_reopens_proposals_and_persists_review_cursor(job):
    from streamlit.testing.v1 import AppTest

    root = read_json(job / "manifest.json")["project_root"]
    prep.run_preparation(job)

    def open_page():
        return AppTest.from_function(preparation_page, args=(root,)).run(timeout=20)

    app = open_page()
    assert not app.exception
    assert app.dataframe[0].value["Calibrazione"].tolist() == ["Bozza", "Bozza"]
    choices = next(w for w in app.selectbox if w.label == "Proposta da esaminare")
    assert len(choices.options) == 2
    choices.select("candidate_2").run()
    next(b for b in app.button if b.label == "Salva bozza").click().run()
    assert not app.exception
    row = prep.table_rows(job)[0]
    assert row["revision"] == 2
    saved = load_record(row["record_path"])
    assert saved.automation.diagnostics["selected_candidate"]["candidate_id"] == "candidate_2"
    assert saved.width.value is None
    next(b for b in app.button if b.label == "Prossimo video da revisionare").click().run()
    assert not app.exception and read_json(job / "review.json")["index"] == 1
    reopened = open_page()
    assert not reopened.exception
    assert next(w for w in reopened.selectbox if w.label == "Video da revisionare").value == 1
    next(w for w in reopened.selectbox if w.label == "Video da revisionare").select(0).run()
    assert not reopened.exception
    assert next(w for w in reopened.selectbox if w.label == "Proposta da esaminare")
    assert prep.table_rows(job)[0]["revision"] == 2


def test_main_menu_opens_preparation_without_yolo_or_labels(job, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from cctv_incident.config import AppConfig

    cfg = AppConfig()
    cfg.project.root_dir = Path(read_json(job / "manifest.json")["project_root"])
    cfg.project.output_dir = cfg.project.root_dir / "outputs"
    seed = AppTest.from_function(preparation_page, args=(str(cfg.project.root_dir),)).run(
        timeout=20
    )
    assert not seed.exception
    monkeypatch.setattr("cctv_incident.config.load_config", lambda _: cfg.model_copy(deep=True))
    app = AppTest.from_file(str(Path("apps/streamlit_app.py").resolve()))
    app._bidi_component_manager = seed._bidi_component_manager
    app.run(timeout=20)
    next(w for w in app.selectbox if w.label == "Modalita").select(
        "standard_dataset analisi in batch - con omografia"
    ).run(timeout=20)
    assert not app.exception
    assert "Proponi calibrazioni mancanti" in [b.label for b in app.button]
    assert not any(b.label in {"Avvia batch", "Avvia analisi"} for b in app.button)
    assert not any(w.label in {"Modello YOLO", "CSV delle etichette"} for w in app.selectbox)
