import json
from pathlib import Path

import av
import numpy as np
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from cctv_incident.calibration.repository import load_record


@pytest.fixture
def marked_video(tmp_path):
    import cv2

    path = tmp_path / "markings.mp4"
    image = np.full((240, 320, 3), 45, np.uint8)
    cv2.rectangle(image, (45, 110), (85, 210), (245, 245, 245), -1)
    cv2.rectangle(image, (155, 110), (195, 210), (245, 245, 245), -1)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def editor_app(source, root):
    import streamlit as st
    from streamlit.runtime import Runtime

    from cctv_incident.calibration_ui import render_calibration_editor
    from cctv_incident.components.calibration_editor import CSS, HTML, JS

    # AppTest replaces the Runtime between app instances while retaining imported
    # Python modules. Register once in each isolated test runtime, as module import
    # does once when the real Streamlit server starts.
    if Runtime.instance().bidi_component_registry.get("road_calibration_editor") is None:
        st.components.v2.component("road_calibration_editor", html=HTML, css=CSS, js=JS)

    selection = render_calibration_editor(source, root)
    st.button("Analisi di prova", disabled=not selection.ready)
    st.session_state["selection"] = selection


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 160, 96, "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.full((96, 160, 3), 80, np.uint8), format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def button(app, label):
    return next(b for b in app.button if b.label == label)


def fill_geometry(app):
    for label, value in [
        ("P1 · X", 10),
        ("P1 · Y", 10),
        ("P2 · X", 140),
        ("P2 · Y", 10),
        ("P3 · X", 140),
        ("P3 · Y", 80),
        ("P4 · X", 10),
        ("P4 · Y", 80),
    ]:
        next(w for w in app.number_input if w.label == label).set_value(value)
    button(app, "Applica coordinate").click().run()
    next(w for w in app.number_input if w.label == "Larghezza P1→P2 (m)").set_value(5).run()
    next(w for w in app.number_input if w.label == "Lunghezza P2→P3 (m)").set_value(12).run()
    for index in range(2):
        [w for w in app.selectbox if w.label == "Origine della misura"][index].select(
            "measured"
        ).run()
        [w for w in app.checkbox if w.label == "Confermo valore e origine"][index].check().run()
    next(w for w in app.checkbox if w.label.startswith("Ho verificato")).check().run()
    assert not app.exception
    assert not button(app, "Conferma e salva calibrazione").disabled


def test_incomplete_draft_saved_and_reloaded_without_analysis(video, tmp_path):
    app = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run(timeout=20)
    assert not app.exception
    assert button(app, "Conferma e salva calibrazione").disabled
    assert button(app, "Analisi di prova").disabled
    button(app, "Salva bozza").click().run()
    assert not app.exception
    record = load_record(tmp_path / "data/calibration/videos/clip.yaml")
    assert record.status == "draft" and not record.vertices
    second = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run()
    assert not second.exception
    assert second.session_state["single_calibration_editor"].record.revision == 1
    assert button(second, "Analisi di prova").disabled


def test_numeric_confirmation_reuse_measure_change_and_undo(video, tmp_path):
    app = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run()
    fill_geometry(app)
    button(app, "Conferma e salva calibrazione").click().run()
    assert not app.exception
    assert not button(app, "Analisi di prova").disabled
    assert app.session_state["selection"].camera_id.startswith("video_")
    second = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run()
    assert not second.exception
    assert button(second, "Analisi di prova").disabled  # Review required in a new session.
    button(second, "Riutilizza calibrazione").click().run()
    assert not second.exception
    assert not button(second, "Analisi di prova").disabled
    button(second, "Modifica calibrazione").click().run()
    assert button(second, "Analisi di prova").disabled
    next(w for w in second.number_input if w.label == "Larghezza P1→P2 (m)").set_value(6).run()
    state = second.session_state["single_calibration_editor"]
    assert state.record.width.value == 6 and not state.record.width.user_confirmed
    assert button(second, "Conferma e salva calibrazione").disabled
    button(second, "Annulla ultima modifica").click().run()
    assert not second.exception
    assert state.record.width.value == 5 and state.record.status == "draft"
    assert button(second, "Analisi di prova").disabled


def test_new_roi_and_reset_do_not_touch_calibration_points(video, tmp_path):
    app = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run()
    fill_geometry(app)
    state = app.session_state["single_calibration_editor"]
    points = state.record.vertices
    button(app, "Nuova ROI").click().run()
    assert not app.exception
    assert state.record.roi_px == () and state.record.vertices == points
    assert next(w for w in app.radio if w.label == "Modifica sul fotogramma").value == "roi"
    assert button(app, "Conferma e salva calibrazione").disabled
    button(app, "ROI: tutto il frame").click().run()
    assert state.record.roi_px is None and state.record.vertices == points
    button(app, "Azzera punti").click().run()
    assert not app.exception
    assert not state.record.vertices and button(app, "Conferma e salva calibrazione").disabled


def test_source_change_clears_frame_geometry_and_acceptance(video, tmp_path):
    # Change the video at the same path inside the same Streamlit session.
    app = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run()
    fill_geometry(app)
    old = app.session_state["single_calibration_editor"]
    replacement = tmp_path / "replacement.mp4"
    with av.open(str(replacement), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 160, 96, "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.full((96, 160, 3), 180, np.uint8), format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    Path(replacement).replace(video)
    app.run()
    assert not app.exception
    new = app.session_state["single_calibration_editor"]
    assert new.record.video.sha256 != old.record.video.sha256
    assert not new.record.vertices and not new.visually_accepted
    assert not np.array_equal(new.frame, old.frame)
    assert button(app, "Analisi di prova").disabled


def test_missing_source_does_not_reuse_existing_session(video, tmp_path):
    app = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run()
    video.unlink()
    app.run()
    assert not app.exception
    assert "single_calibration_editor" not in app.session_state
    assert button(app, "Analisi di prova").disabled


def test_native_component_release_callback_hydrates_numeric_fields_once(video, tmp_path):
    app = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run()
    component = app.get("bidi_component")[0]
    data = json.loads(component.proto.json)
    payload = {k: data[k] for k in ["record_id", "source_sha256", "revision", "epoch"]}
    payload.update(
        event_id="native-release", mode="calibration", vertices=[{"id": "P1", "x": 24, "y": 18}]
    )
    states = app._tree.get_widget_states()
    state = states.widgets.add()
    state.id = component.proto.id
    state.json_value = "{}"
    trigger = states.widgets.add()
    trigger.id = "$$STREAMLIT_INTERNAL_KEY_" + component.proto.id + "__events"
    trigger.json_trigger_value = json.dumps([{"event": "edit", "value": payload}])
    app._run(states)
    assert not app.exception
    session = app.session_state["single_calibration_editor"]
    assert session.record.vertices[0].x == 24 and session.epoch == 1
    assert next(w for w in app.number_input if w.label == "P1 · X").value == 24
    assert next(w for w in app.number_input if w.label == "P1 · Y").value == 18
    assert not session.visually_accepted
    app.run()
    assert len(session.history) == 1 and session.epoch == 1
    assert json.loads(app.get("bidi_component")[0].proto.json)["vertices"][0]["x"] == 24


def test_advanced_correspondences_require_scale_review(video, tmp_path):
    app = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run()
    next(w for w in app.text_area if w.label == "Punti immagine [x, y] (JSON)").set_value(
        "[[10,10],[140,10],[140,80],[10,80]]"
    )
    next(w for w in app.text_area if w.label == "Punti sul piano [x, y] in metri (JSON)").set_value(
        "[[0,0],[5,0],[5,12],[0,12]]"
    )
    next(w for w in app.text_input if w.label == "Fonte della scala metrica").set_value(
        "Rilievo della strada"
    )
    button(app, "Applica corrispondenze avanzate").click().run()
    assert not app.exception
    assert app.session_state["single_calibration_editor"].record.geometry_mode == "explicit"
    assert button(app, "Conferma e salva calibrazione").disabled
    next(w for w in app.checkbox if w.label.startswith("Confermo la scala")).check().run()
    next(w for w in app.checkbox if w.label.startswith("Ho verificato")).check().run()
    button(app, "Conferma e salva calibrazione").click().run()
    assert not app.exception and not button(app, "Analisi di prova").disabled


def test_main_metric_app_gates_analysis_and_uses_video_record(video, tmp_path, monkeypatch):
    from cctv_incident.config import load_config

    seed = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run()
    cfg = load_config("configs/default.yaml")
    cfg.video.source = str(video)
    cfg.project.root_dir = tmp_path
    cfg.project.output_dir = tmp_path / "outputs"
    configuration = tmp_path / "config.yaml"
    configuration.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    captured = []

    class FakePipeline:
        def __init__(self, config):
            captured.append(config)

        def run(self, update):
            return {"events": 0, "run_dir": str(tmp_path / "test-run")}

    monkeypatch.setattr("cctv_incident.pipeline.Pipeline", FakePipeline)
    app = AppTest.from_file(str(Path("apps/streamlit_app.py").resolve()))
    # The real server has a single registry; isolated AppTest instances do not.
    app._bidi_component_manager = seed._bidi_component_manager
    app.run(timeout=20)
    next(w for w in app.selectbox if w.label == "Modalita").select(
        "Segmentazione + omografia"
    ).run()
    next(w for w in app.text_input if w.label == "Configurazione YAML").set_value(
        str(configuration)
    ).run()
    assert not app.exception and button(app, "Avvia analisi").disabled
    fill_geometry(app)
    button(app, "Conferma e salva calibrazione").click().run()
    assert not app.exception and not button(app, "Avvia analisi").disabled
    button(app, "Avvia analisi").click().run()
    assert not app.exception and len(captured) == 1
    assert captured[0].calibration.file == tmp_path / "data/calibration/videos/clip.yaml"
    assert (
        captured[0].calibration.camera_id
        == app.session_state["single_calibration_editor"].record.camera_id
    )
    record = app.session_state["single_calibration_editor"].record
    assert captured[0].calibration.expected_record_id == record.record_id
    assert captured[0].calibration.expected_revision == record.revision


@pytest.mark.parametrize("invalidate", [False, True])
def test_metric_gui_reuse_real_pipeline_and_replay(video, tmp_path, monkeypatch, invalidate):
    from cctv_incident.config import AppConfig

    seed = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run(timeout=20)
    fill_geometry(seed)
    button(seed, "Conferma e salva calibrazione").click().run()
    record = seed.session_state["single_calibration_editor"].record
    cfg = AppConfig()
    cfg.project.root_dir = tmp_path
    cfg.project.output_dir = tmp_path / "outputs"
    cfg.video.source = str(video)
    cfg.perception.backend = "synthetic"
    # Even a custom image-mode YAML must not bypass the selected metric workflow.
    cfg.events.coordinate_mode = "image"
    monkeypatch.setattr("cctv_incident.config.load_config", lambda _: cfg.model_copy(deep=True))
    monkeypatch.setattr("cctv_incident.pipeline.hardware_info", lambda: {})
    if invalidate:
        monkeypatch.setattr("cctv_incident.pipeline.CameraMotionGuard.update", lambda *args: True)
    app = AppTest.from_file(str(Path("apps/streamlit_app.py").resolve()))
    app._bidi_component_manager = seed._bidi_component_manager
    app.run(timeout=20)
    next(w for w in app.selectbox if w.label == "Modalita").select("Segmentazione + omografia").run(
        timeout=20
    )
    assert not app.exception and button(app, "Avvia analisi").disabled
    button(app, "Riutilizza calibrazione").click().run()
    assert not button(app, "Avvia analisi").disabled
    button(app, "Avvia analisi").click().run(timeout=20)
    assert not app.exception
    summary = app.session_state["last_run"]
    assert summary["evaluable"] is not invalidate
    assert summary["coordinate_mode"] == "metric"
    run = Path(summary["run_dir"])
    metadata = json.loads((run / "run.json").read_text())
    assert metadata["calibration_provenance"]["record_id"] == record.record_id
    assert metadata["calibration_provenance"]["revision"] == record.revision
    assert metadata["config"]["calibration"]["camera_id"] == record.camera_id
    replay = json.loads((run / "replay.json").read_text())
    assert replay["evaluable"] is not invalidate
    assert Path(replay["video_path"]).is_file()
    app.run(timeout=20)
    assert len(app.get("video")) == 1
    if invalidate:
        assert any("Questa analisi non è valutabile" in w.value for w in app.warning)
        assert not any("Nessun impatto rilevato" in w.value for w in app.info)


def generate_and_apply(app, index=0):
    button(app, "Calibrazione automatica").click().run(timeout=20)
    assert not app.exception
    choices = next(w for w in app.selectbox if w.label == "Proposta da esaminare")
    choices.select(f"candidate_{index + 1}").run()
    assert not app.exception


def test_automatic_alternatives_populate_editable_points_and_save_draft(marked_video, tmp_path):
    app = AppTest.from_function(editor_app, args=(str(marked_video), str(tmp_path))).run()
    generate_and_apply(app, index=1)
    state = app.session_state["single_calibration_editor"]
    assert len(state.record.vertices) == 4
    assert state.record.width.value is None and state.record.length.value is None
    assert state.record.status == "draft" and not state.visually_accepted
    selected = state.record.automation.diagnostics["selected_candidate"]
    assert selected["candidate_id"] == "candidate_2"
    assert tuple(map(tuple, state.record.points_px)) == state.record.automation.initial_points_px
    assert button(app, "Conferma e salva calibrazione").disabled
    assert not any(w.label == "P1 · X" for w in app.number_input)
    assert not any(w.label == "Azzera punti" for w in app.button)
    assert not any(w.label == "Modifica sul fotogramma" for w in app.radio)
    old = state.record.vertices[0].x
    component = app.get("bidi_component")[0]
    data = json.loads(component.proto.json)
    payload = {k: data[k] for k in ("record_id", "source_sha256", "revision", "epoch")}
    payload.update(event_id="automatic-drag", mode="calibration", vertices=data["vertices"])
    payload["vertices"][0]["x"] = old + 1
    states = app._tree.get_widget_states()
    widget = states.widgets.add()
    widget.id, widget.json_value = component.proto.id, "{}"
    trigger = states.widgets.add()
    trigger.id = "$$STREAMLIT_INTERNAL_KEY_" + component.proto.id + "__events"
    trigger.json_trigger_value = json.dumps([{"event": "edit", "value": payload}])
    app._run(states)
    app.run()
    assert not app.exception
    assert state.record.vertices[0].x == old + 1 and state.record.status == "draft"
    button(app, "Salva bozza").click().run()
    saved = load_record(tmp_path / "data/calibration/videos/markings.yaml")
    assert saved.automation.initial_points_px == tuple(map(tuple, selected["points_px"]))
    assert saved.vertices[0].x == old + 1 and saved.automation.method == "auto_assisted"


def test_blank_automatic_result_keeps_manual_work_and_explains_fallback(video, tmp_path):
    app = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run()
    next(w for w in app.number_input if w.label == "P1 · X").set_value(20)
    next(w for w in app.number_input if w.label == "P1 · Y").set_value(20)
    button(app, "Applica coordinate").click().run()
    button(app, "Calibrazione automatica").click().run()
    assert not app.exception
    state = app.session_state["single_calibration_editor"]
    assert state.record.vertices[0].x == 20 and not state.proposal_result.candidates
    assert any("Usa i clic" in w.value for w in app.info)
    assert not app.get("bidi_component")
    button(app, "Torna alla selezione manuale").click().run()
    assert not app.exception
    assert next(w for w in app.number_input if w.label == "P1 · X").value == 20
    assert not button(app, "Salva bozza").disabled


def test_preset_requires_explicit_checks_and_keeps_origin_after_confirmation(
    marked_video, tmp_path
):
    app = AppTest.from_function(editor_app, args=(str(marked_video), str(tmp_path))).run()
    generate_and_apply(app)
    assert not next(
        w for w in app.checkbox if w.label.startswith("Ho verificato la provenienza")
    ).value
    next(w for w in app.selectbox if w.label == "Preset dimensionale").select(
        "us_broken_10ft"
    ).run()
    assert button(app, "Applica preset come ipotesi").disabled
    next(
        w for w in app.checkbox if w.label.startswith("Ho verificato la provenienza")
    ).check().run()
    next(w for w in app.selectbox if w.label == "Tipo di segnaletica verificato").select(
        "broken_line"
    ).run()
    next(w for w in app.checkbox if w.label.startswith("Il rettangolo contiene")).check().run()
    button(app, "Applica preset come ipotesi").click().run()
    assert not app.exception
    state = app.session_state["single_calibration_editor"]
    assert state.record.length.value == 3.048 and state.record.length.origin == "standard"
    assert next(w for w in app.number_input if w.label == "Lunghezza P2→P3 (m)").value == 3.048
    assert not state.record.length.user_confirmed and state.record.width.value is None
    assert state.record.length.preset == "us_broken_10ft"
    length_checkbox = [w for w in app.checkbox if w.label == "Confermo valore e origine"][1]
    length_checkbox.check().run()
    assert (
        state.record.length.origin == "standard" and state.record.length.preset == "us_broken_10ft"
    )
    assert button(app, "Conferma e salva calibrazione").disabled
    # Editing the suggested number retains its hypothesis provenance and resets review.
    next(w for w in app.number_input if w.label == "Lunghezza P2→P3 (m)").set_value(3.2).run()
    assert state.record.length.value == 3.2 and not state.record.length.user_confirmed
    assert (
        state.record.length.origin == "standard" and state.record.length.preset == "us_broken_10ft"
    )


def test_automatic_applies_known_distances_immediately_and_preserves_corrections(
    marked_video, tmp_path, monkeypatch
):
    from dataclasses import replace

    from cctv_incident.calibration.proposals import generate_proposals
    from cctv_incident.calibration.records import Distance
    from cctv_incident.calibration.repository import create_draft

    _, image = create_draft(marked_video, tmp_path)
    result = generate_proposals(image)
    candidate = result.candidates[0].model_copy(
        update={
            "width": Distance(value=0.4, origin="experimental", source="fixture known hypothesis"),
            "length": Distance(value=3, origin="experimental", source="fixture known hypothesis"),
        }
    )
    result = replace(result, candidates=(candidate, *result.candidates[1:]))
    monkeypatch.setattr(
        "cctv_incident.calibration_proposal_ui.generate_proposals", lambda *a: result
    )
    app = AppTest.from_function(editor_app, args=(str(marked_video), str(tmp_path))).run()
    button(app, "Calibrazione automatica").click().run()
    assert not app.exception
    state = app.session_state["single_calibration_editor"]
    assert state.record.points_px == [list(p) for p in candidate.points_px]
    assert not any(
        w.label in {"Usa proposta nell'editor", "Applica coordinate", "Azzera punti"}
        for w in app.button
    )
    width = next(w for w in app.number_input if w.label == "Larghezza P1→P2 (m)")
    assert width.value == 0.4
    assert next(w for w in app.number_input if w.label == "Lunghezza P2→P3 (m)").value == 3
    assert not state.record.width.user_confirmed
    width.set_value(0.45).run()
    app.run()
    assert not app.exception and state.record.width.value == 0.45
    button(app, "Salva bozza").click().run()
    reopened = AppTest.from_function(editor_app, args=(str(marked_video), str(tmp_path))).run()
    assert not reopened.exception
    assert reopened.session_state["single_calibration_editor"].workflow == "automatic"
    assert next(w for w in reopened.number_input if w.label == "Larghezza P1→P2 (m)").value == 0.45
    assert not any(w.label == "P1 · X" for w in reopened.number_input)
    assert any(w.label == "Preset dimensionale" for w in reopened.selectbox)
    # Switching candidates must not carry the previous reference's distances across.
    next(w for w in app.selectbox if w.label == "Proposta da esaminare").select("candidate_2").run()
    assert not app.exception
    assert state.record.width.value is None and state.record.length.value is None


def test_automatic_failure_can_return_to_manual_editor(video, tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("invalid ROI")

    monkeypatch.setattr("cctv_incident.calibration_proposal_ui.generate_proposals", fail)
    app = AppTest.from_function(editor_app, args=(str(video), str(tmp_path))).run()
    button(app, "Calibrazione automatica").click().run()
    assert not app.exception and not app.get("bidi_component")
    assert any("ROI" in w.value for w in app.warning)
    button(app, "Torna alla selezione manuale").click().run()
    assert not app.exception and app.get("bidi_component")
    assert any(w.label == "P1 · X" for w in app.number_input)


def test_live_session_from_previous_editor_keeps_existing_automatic_corrections(
    marked_video, tmp_path
):
    from cctv_incident.calibration.editor import EditorSession

    app = AppTest.from_function(editor_app, args=(str(marked_video), str(tmp_path))).run()
    generate_and_apply(app)
    next(w for w in app.number_input if w.label == "Larghezza P1→P2 (m)").set_value(0.7).run()
    current = app.session_state["single_calibration_editor"]
    old_type = type(
        "PreviousEditorSession",
        (),
        {name: getattr(EditorSession, name) for name in ("change", "undo", "receive")},
    )
    previous = old_type()
    previous.__dict__.update(
        {
            key: value
            for key, value in current.__dict__.items()
            if key not in {"workflow", "applied_proposal", "proposal_error"}
        }
    )
    app.session_state["single_calibration_editor"] = previous
    app.run()
    assert not app.exception
    assert previous.workflow == "automatic"
    assert previous.record.width.value == 0.7
    assert previous.record.points_px == current.record.points_px
    assert not any(w.label == "P1 · X" for w in app.number_input)


def test_roi_change_discards_proposals_and_reset_returns_to_manual(marked_video, tmp_path):
    app = AppTest.from_function(editor_app, args=(str(marked_video), str(tmp_path))).run()
    generate_and_apply(app)
    state = app.session_state["single_calibration_editor"]
    assert state.proposal_result is not None
    button(app, "Torna alla selezione manuale").click().run()
    button(app, "Nuova ROI").click().run()
    assert not app.exception and state.proposal_result is None
    button(app, "Azzera punti").click().run()
    assert not app.exception and state.record.automation.method == "manual"
